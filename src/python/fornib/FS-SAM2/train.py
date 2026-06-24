""" FS-SAM2 training code """
import os
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

TASK_DIR = Path(__file__).resolve().parents[2] / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from dltool_task_protocol import TaskClient, TaskStatus


class TaskStopRequested(Exception):
    pass


def create_task_client(args):
    if not args.dltool_task_host or args.dltool_task_port <= 0 or args.dltool_task_id < 0:
        return None
    return TaskClient(args.dltool_task_host, args.dltool_task_port)


def task_progress(args, epoch, batch_index, batch_count, training):
    phase_offset = 0.0 if training else 0.5
    phase_fraction = 0.5 * (batch_index + 1) / max(1, batch_count)
    return min(98, int(98 * (epoch + phase_offset + phase_fraction) / max(1, args.epochs)))


def report_task_status(client, args, status, progress=None, message=""):
    if client is not None and utils.is_main_process():
        client.status(args.dltool_task_id, status, progress, message)


def report_task_progress(client, args, progress, message=""):
    if client is not None and utils.is_main_process():
        client.progress(args.dltool_task_id, progress, message)


def raise_if_task_stopped(client, args, progress=None):
    if client is not None and utils.is_main_process() and client.should_stop(args.dltool_task_id):
        report_task_status(client, args, TaskStatus.STOPPED, progress, "任务已停止")
        raise TaskStopRequested()


def train(args, epoch, sam_model, dataloader, optimizer, scheduler, training, kshot=1, task_client=None):
    """Training or validation of the model"""
    
    sam_model.train() if training else sam_model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    start_time = time.time()

    for idx, batch in enumerate(dataloader):
        progress = task_progress(args, epoch, idx, len(dataloader), training)
        raise_if_task_stopped(task_client, args, progress)

        batch_time = time.time()
        batch = utils.to_cuda(batch)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  # bfloat16 for faster training
            current_out = {}
            for i in range(len(batch['support_names'])):  # accumulate previous outputs
                current_out = sam_model(batch['support_imgs'][:, i], batch['support_masks'][:, i], prev_out=current_out)
            current_out, loss = sam_model(batch['query_img'], prev_out=current_out, query_mask=batch['query_mask'])
        
        logit_mask = current_out["logit_mask"]
        pred_mask = (logit_mask > 0.0).float()  # threshold

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Evaluate prediction
        area_inter, area_union, area_pred, area_gt = Evaluator.classify_prediction(pred_mask.squeeze(1), batch['query_mask'], batch.get('query_ignore_idx'))
        average_meter.update(area_inter, area_union, area_pred, area_gt, batch['class_id'], loss=loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100, dt=time.time()-batch_time)
        report_task_progress(task_client, args, progress, f"Epoch {epoch + 1}/{args.epochs}")
        raise_if_task_stopped(task_client, args, progress)

    # Write evaluation results
    torch.cuda.synchronize()
    dt = time.time() - start_time
    if utils.is_main_process():
        Logger.info(f'Epoch {epoch} | dt: {dt}.\n')
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
 
    return avg_loss, miou, fb_iou


def main():
    """ Run training with validation, save best model """

    # Arguments parsing
    parser = argparse.ArgumentParser(description='FS-SAM2 Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../datasets/')  # CHANGE TO YOUR PATH
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss', 'custom'])
    parser.add_argument('--exp_id', type=str, default='0000')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--kshot', type=int, default=1)  # 1-shot training
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8, help='number of cpu threads to use during batch generation')  # 0 for Windows, 8 for Linux
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during PASCAL training')
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoint/sam2.1_hiera_base_plus.pt')
    parser.add_argument('--sam2_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dltool_task_host', type=str, default='')
    parser.add_argument('--dltool_task_port', type=int, default=0)
    parser.add_argument('--dltool_task_id', type=int, default=-1)
    args = parser.parse_args()

    if args.logpath == '':  # if empty, autogenerate logpath from args
        args.logpath = f'{args.benchmark}/{args.exp_id}/fold{args.fold}'

    # Distributed setting
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:  # to use DDP run with `python -m torch.distributed.run X.py`
        assert torch.cuda.is_available(), 'requires CUDA for DDP'
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_node_local_rank()  # GPU id
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    else:
        local_rank = 0  # just 1 process
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_client = create_task_client(args) if utils.is_main_process() else None
    report_task_status(task_client, args, TaskStatus.RUNNING, 0, "开始 FS-SAM2 训练")
    
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)

    torch.set_float32_matmul_precision('high')  # use Tensor Cores (slightly less precision)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args.kshot, seed=args.seed)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.kshot, seed=args.seed)

    Evaluator.initialize(args.use_ignore)


    # Model initialization
    from peft import LoraConfig, get_peft_model
    from sam2_pred import SAM2_pred
    # from sam2.sam2_video_predictor import SAM2VideoPredictorVOS

    sam_model = SAM2_pred(checkpoint=args.sam2_checkpoint, model_cfg=args.sam2_cfg)
    # sam_model.model.image_encoder.forward = torch.compile(  # compile image encoder
    #             sam_model.model.image_encoder.forward,
    #             mode="max-autotune",
    #             fullgraph=True,
    #             dynamic=False)
    # SAM2VideoPredictorVOS._compile_all_components(sam_model.model)  # compile everything else
    
    # LoRA image_encoder
    peft_config_encoder = LoraConfig(inference_mode=False,
                             r=4,
                             lora_alpha=16,
                             lora_dropout=0.1,
                             target_modules=['qkv', 'proj'],
                             bias="none",
                             )
    peft_encoder = get_peft_model(sam_model.model.image_encoder, peft_config_encoder)
    sam_model.model.image_encoder = peft_encoder

    # LoRA memory_attention
    peft_config_mem = LoraConfig(inference_mode=False,
                             r=32,
                             lora_alpha=16,
                             lora_dropout=0.1,
                             target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                             bias="none",
                             )
    peft_mem = get_peft_model(sam_model.model.memory_attention, peft_config_mem)
    sam_model.model.memory_attention = peft_mem

    # LoRA memory_encoder
    peft_config_mem_enc = LoraConfig(inference_mode=False,
                             r=32,
                             lora_alpha=16,
                             lora_dropout=0.1,
                             target_modules=['out_proj'],
                             bias="none",
                             )
    peft_mem_enc = get_peft_model(sam_model.model.memory_encoder, peft_config_mem_enc)
    sam_model.model.memory_encoder = peft_mem_enc

    # DDP
    if utils.is_dist_avail_and_initialized():
        sam_model = torch.nn.parallel.DistributedDataParallel(sam_model, device_ids=[local_rank], find_unused_parameters=True)  # distributed model
    raw_model = sam_model.module if utils.is_dist_avail_and_initialized() else sam_model

    # Freeze layers
    for name, param in raw_model.named_parameters():
        if not name.startswith('model.image_encoder') and not name.startswith('model.memory_attention') and not name.startswith('model.memory_encoder'):
            param.requires_grad = False  # freeze all other layers

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': (p for p in raw_model.model.parameters() if p.requires_grad and p.dim() >= 2)},
        {'params': (p for p in raw_model.model.parameters() if p.requires_grad and p.dim() < 2), 'weight_decay': 0.0},  # dont decay 1-dim tensors/biases
        ], lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), fused=('cuda' in device.type))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader_trn))

    # Print number of parameters
    if utils.is_main_process():
        Logger.log_params(raw_model)

    # Training
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    val_loss, val_miou, val_fb_iou = 10, 0, 0
    try:
        for epoch in range(args.epochs):
            raise_if_task_stopped(task_client, args)
            if utils.is_dist_avail_and_initialized():
                dataloader_trn.sampler.set_epoch(epoch)  # shuffle dataset
            dataloader_trn.dataset.set_epoch(epoch)  # randomize support-query pairs
            trn_loss, trn_miou, trn_fb_iou = train(args, epoch, sam_model, dataloader_trn, optimizer, scheduler,
                                                   training=True, kshot=args.kshot, task_client=task_client)  # training

            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(args, epoch, sam_model, dataloader_val, optimizer, scheduler,
                                                       training=False, kshot=args.kshot, task_client=task_client)  # validation

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                if utils.is_main_process():
                    Logger.save_model_miou(raw_model, epoch, val_miou)
            if utils.is_main_process():
                Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
                Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
                Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
                Logger.tbd_writer.flush()
    except TaskStopRequested:
        if task_client is not None:
            task_client.close()
        return 130
    except Exception as exc:
        report_task_status(task_client, args, TaskStatus.FAILED, message=str(exc))
        if task_client is not None:
            task_client.close()
        return 1
    finally:
        if utils.is_dist_avail_and_initialized():
            dist.destroy_process_group()

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
    report_task_status(task_client, args, TaskStatus.FINISHED, 100, "训练完成")
    if task_client is not None:
        task_client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
