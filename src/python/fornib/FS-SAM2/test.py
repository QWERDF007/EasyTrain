""" FS-SAM2 testing code """
import os
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def test(args, sam_model, dataloader, kshot):
    """ Test FS-SAM2 model """

    sam_model.eval()

    average_meter = AverageMeter(dataloader.dataset)
    start_time = time.time()

    for idx, batch in enumerate(dataloader):
        batch_time = time.time()
        batch = utils.to_cuda(batch)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            current_out = {}
            for i in range(len(batch['support_names'])):  # for each shot, accumulate previous outputs
                current_out = sam_model(batch['support_imgs'][:, i], batch['support_masks'][:, i], prev_out=current_out)
            current_out = sam_model(batch['query_img'], prev_out=current_out)

        logit_mask = current_out['logit_mask']
        if args.use_original_imgsize:
            size_y, size_x = batch['query_mask'].shape[-2:]
            logit_mask = F.interpolate(logit_mask, (size_y, size_x), mode='bilinear', align_corners=True)
        pred_mask = (logit_mask > 0.0).float()  # threshold

        # Evaluate prediction
        area_inter, area_union, area_pred, area_gt = Evaluator.classify_prediction(pred_mask.squeeze(1), batch['query_mask'], batch.get('query_ignore_idx'))
        iou = area_inter / torch.max(torch.stack([area_union, torch.ones_like(area_union)]), dim=0)[0]
        
        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                batch['query_img'], batch['query_mask'],
                                                logit_mask.squeeze(1),
                                                batch['class_id'], idx, batch['query_name'], batch['support_names'][0],
                                                iou_b=100*iou[1])
        
        average_meter.update(area_inter, area_union, area_pred, area_gt, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1, dt=time.time()-batch_time)

    # Write evaluation results
    torch.cuda.synchronize()
    dt = time.time() - start_time
    if utils.is_main_process():
        Logger.info(f'Total dt: {dt}.\n')

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


def main():
    """ Run testing """

    # Arguments parsing
    parser = argparse.ArgumentParser(description='FS-SAM2 Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../datasets/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss', 'custom'])
    parser.add_argument('--exp_id', type=str, default='0000')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--kshot', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--use_original_imgsize', default=True, action='store_true')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=8, help='number of cpu threads to use during batch generation')  # 0 for Windows, 8 for Linux
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during PASCAL training')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoint/sam2.1_hiera_base_plus.pt')
    parser.add_argument('--sam2_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.logpath == '':  # if empty, autogenerate logpath from args
        args.logpath = f'{args.benchmark}/{args.exp_id}/fold{args.fold}'
    if hasattr(args, 'load') and args.load == '':  # if exists and empty, load pretrained model
        args.load = os.path.join('logs', args.logpath + '.log', 'best_model.pt')

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
    
    if utils.is_main_process():
        Logger.initialize(args, training=False)
    utils.fix_randseed(args.seed)

    torch.set_float32_matmul_precision('high')  # use Tensor Cores (slightly less precision)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.kshot, seed=args.seed)

    Evaluator.initialize(args.use_ignore)
    Visualizer.initialize(args.visualize, path=f'./vis/{args.benchmark}{args.exp_id}_{args.fold}/')


    # Model initialization
    from peft import LoraConfig, get_peft_model
    from sam2_pred import SAM2_pred
    # from sam2.sam2_video_predictor import SAM2VideoPredictorVOS

    sam_model = SAM2_pred(checkpoint=args.sam2_checkpoint, model_cfg=args.sam2_cfg)
    # sam_model.model.image_encoder.forward = torch.compile(  # compile image encoder (only works without LoRA)
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
    
    # Load trained model
    if hasattr(args, 'load'):
        state_dict = torch.load(args.load)['state_dict']
        sam_model.load_state_dict(state_dict)
    
    if utils.is_main_process():
        Logger.log_params(raw_model)

    # Test model
    with torch.inference_mode():
        test_miou, test_fb_iou = test(args, sam_model, dataloader_test, args.kshot)
    
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')

    if utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
