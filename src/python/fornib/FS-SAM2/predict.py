""" FS-SAM2 prediction script: segment query images given support examples """
import os
import sys
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import traceback
import torch
import torch.nn.functional as F
import PIL.Image as Image
import yaml
from torchvision import tv_tensors
from torchvision.transforms import v2

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


def task_progress(args, done, total):
    progress = args.dltool_progress_base + args.dltool_progress_span * done / max(1, total)
    return min(100, max(0, int(progress)))


def estimate_task_eta(args, done, total):
    start_time = getattr(args, "dltool_eta_start_time", None)
    if start_time is None or done <= 0:
        return -1

    elapsed = time.time() - start_time
    completed_span = args.dltool_progress_span * done / max(1, total)
    if elapsed <= 0 or completed_span <= 0:
        return -1

    remaining_span = max(0.0, 100.0 - args.dltool_progress_base - completed_span)
    return int(round(elapsed * remaining_span / completed_span))


def report_task_status(client, args, status, progress, eta_seconds, message):
    if client is not None:
        client.status(args.dltool_task_id, status, progress, eta_seconds, message)


def report_task_progress(client, args, progress, eta_seconds, message):
    if client is not None:
        client.progress(args.dltool_task_id, progress, eta_seconds, message)


def raise_if_task_stopped(client, args, progress=-1, eta_seconds=-1):
    if client is not None and client.should_stop(args.dltool_task_id):
        report_task_status(client, args, TaskStatus.STOPPED, progress, eta_seconds, "任务已停止")
        raise TaskStopRequested()


def load_support(support_dir, img_size, transform, device, task_client=None, args=None):
    """Load all support images and masks from a directory with images/ and masks/ subdirs."""
    img_dir = os.path.join(support_dir, 'images')
    mask_dir = os.path.join(support_dir, 'masks')
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Support images directory not found: {img_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Support masks directory not found: {mask_dir}')

    imgs, masks, names = [], [], []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for img_path in sorted(glob.glob(os.path.join(img_dir, ext))):
            if args is not None:
                raise_if_task_stopped(task_client, args, args.dltool_progress_base, -1)
            name = Path(img_path).stem
            mask_path = os.path.join(mask_dir, name + '.png')
            if not os.path.exists(mask_path):
                continue
            img = Image.open(img_path).convert('RGB')
            mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
            mask[mask < 128] = 0
            mask[mask >= 128] = 1

            img = img.resize((img_size, img_size))
            mask = tv_tensors.Mask(F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (img_size, img_size), mode='nearest').squeeze())
            img, mask = transform(img, mask)
            imgs.append(img)
            masks.append(mask)
            names.append(name)

    if not imgs:
        raise FileNotFoundError(f'No support images found in {img_dir} with matching masks in {mask_dir}')
    print(f'Loaded {len(imgs)} support images')
    return torch.stack(imgs).to(device), torch.stack(masks).to(device), names


def load_queries(query_dir):
    """List all query image paths."""
    queries = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        queries.extend(sorted(glob.glob(os.path.join(query_dir, ext))))
    if not queries:
        raise FileNotFoundError(f'No query images found in {query_dir}')
    return queries


def load_config(path):
    with open(path, 'r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def group(values, *keys):
    current = values
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def text(values, name, default=''):
    value = values.get(name, default) if isinstance(values, dict) else default
    return default if value is None else str(value).strip()


def integer(values, name, default=0):
    try:
        return int(values.get(name, default)) if isinstance(values, dict) else default
    except (TypeError, ValueError):
        return default


def apply_dltool_config(args):
    if not args.config:
        return

    config = load_config(args.config)
    datasets = group(config, 'datasets')
    train_manifest = text(group(datasets, 'train'), 'manifest')
    test_manifest = text(group(datasets, 'test'), 'manifest')
    if not train_manifest:
        raise ValueError('datasets.train.manifest is empty')
    if not test_manifest:
        raise ValueError('datasets.test.manifest is empty')

    args.support_manifest = train_manifest
    args.query_manifest = test_manifest

    inference = group(config, 'test_params', 'inference')
    model = group(config, 'test_params', 'model')
    args.output_dir = text(inference, 'output_dir', text(config, 'result_dir', args.output_dir))
    checkpoint = text(inference, 'checkpoint_path')
    if checkpoint:
        args.checkpoint = checkpoint
    elif text(config, 'weight_dir'):
        args.checkpoint = str(Path(text(config, 'weight_dir')) / 'fs_sam2' / 'best_model.pt')

    args.img_size = integer(inference, 'image_size', integer(model, 'image_size', args.img_size))
    args.sam2_checkpoint = text(model, 'sam2_checkpoint', args.sam2_checkpoint)
    args.sam2_cfg = text(model, 'sam2_cfg', args.sam2_cfg)


def manifest_images(path):
    with open(path, 'r', encoding='utf-8') as handle:
        manifest = yaml.safe_load(handle) or {}
    images = manifest.get('images', [])
    if not isinstance(images, list):
        raise ValueError(f'Manifest images is not a list: {path}')
    return [image for image in images if isinstance(image, dict)]


def load_support_entries_by_class(path):
    entries_by_class = {}
    for image in manifest_images(path):
        image_path = str(image.get('path', '')).strip()
        if not image_path:
            continue
        masks_by_class = {}
        for label in image.get('labels', []) or []:
            class_id = int(label.get('label_class_id', -1))
            class_name = str(label.get('label_class_name') or class_id)
            mask_path = str(label.get('mask_path', '')).strip()
            if class_id < 0:
                continue
            if not mask_path:
                continue
            masks_by_class.setdefault((class_id, class_name), []).append(mask_path)
        for (class_id, class_name), mask_paths in masks_by_class.items():
            entry = {
                'id': str(image.get('id', '')),
                'name': Path(image_path).stem,
                'path': image_path,
                'class_id': class_id,
                'class_name': class_name,
                'mask_paths': mask_paths,
            }
            entries_by_class.setdefault(class_name, []).append(entry)
    return entries_by_class


def load_query_entries(path):
    queries = []
    for image in manifest_images(path):
        image_path = str(image.get('path', '')).strip()
        if image_path:
            queries.append((str(image.get('id', Path(image_path).stem)), image_path))
    if not queries:
        raise FileNotFoundError(f'No query images found in manifest: {path}')
    return queries


def load_mask_paths(mask_paths, image_size):
    merged = None
    for mask_path in mask_paths or []:
        mask = Image.open(mask_path).convert('L')
        if mask.size != image_size:
            mask = mask.resize(image_size, Image.NEAREST)
        values = (np.array(mask, dtype=np.uint8) >= 128).astype(np.uint8)
        merged = values if merged is None else np.maximum(merged, values)
    if merged is None:
        raise FileNotFoundError('No mask_path entries found in manifest support entry')
    return torch.tensor(merged, dtype=torch.uint8)


def load_manifest_support(entries, img_size, transform, device, task_client=None, args=None):
    imgs, masks, names = [], [], []
    for entry in entries[: max(1, args.kshot if args is not None else 1)]:
        if args is not None:
            raise_if_task_stopped(task_client, args, args.dltool_progress_base, -1)
        image = Image.open(entry['path']).convert('RGB')
        mask = load_mask_paths(entry.get('mask_paths', []), image.size)
        image = image.resize((img_size, img_size))
        mask = tv_tensors.Mask(F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (img_size, img_size), mode='nearest').squeeze())
        image, mask = transform(image, mask)
        imgs.append(image)
        masks.append(mask)
        names.append(entry['name'])
    if not imgs:
        raise FileNotFoundError('No support entries found in manifest')
    return torch.stack(imgs).to(device), torch.stack(masks).to(device), names


def setup_model(checkpoint_path, device, sam2_checkpoint=None, sam2_cfg=None):
    """Build FS-SAM2 model with LoRA and load trained weights."""
    from peft import LoraConfig, get_peft_model
    from sam2_pred import SAM2_pred

    sam_model = SAM2_pred(checkpoint=sam2_checkpoint, model_cfg=sam2_cfg)

    # LoRA image_encoder
    peft_config_encoder = LoraConfig(inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1,
                                     target_modules=['qkv', 'proj'], bias="none")
    sam_model.model.image_encoder = get_peft_model(sam_model.model.image_encoder, peft_config_encoder)

    # LoRA memory_attention
    peft_config_mem = LoraConfig(inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
                                 target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj'], bias="none")
    sam_model.model.memory_attention = get_peft_model(sam_model.model.memory_attention, peft_config_mem)

    # LoRA memory_encoder
    peft_config_mem_enc = LoraConfig(inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
                                     target_modules=['out_proj'], bias="none")
    sam_model.model.memory_encoder = get_peft_model(sam_model.model.memory_encoder, peft_config_mem_enc)

    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    sam_model.load_state_dict(state_dict)
    sam_model.to(device)
    sam_model.eval()
    return sam_model


def run_manifest_prediction(args, model, transform, device, task_client):
    support_by_class = load_support_entries_by_class(args.support_manifest)
    query_list = load_query_entries(args.query_manifest)
    if not support_by_class:
        raise FileNotFoundError(f'No support labels found in manifest: {args.support_manifest}')

    total_steps = max(1, len(support_by_class) * len(query_list))
    done = 0
    args.dltool_eta_start_time = time.time()
    for class_name, entries in sorted(support_by_class.items()):
        support_imgs, support_masks, _ = load_manifest_support(entries, args.img_size, transform, device, task_client, args)
        class_output_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        print(f'Encoding support class: {class_name}')
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            current_out = {}
            for i in range(len(support_imgs)):
                raise_if_task_stopped(task_client, args, task_progress(args, done, total_steps), -1)
                current_out = model(support_imgs[i].unsqueeze(0), support_masks[i].unsqueeze(0), prev_out=current_out)

        for qid, qpath in query_list:
            progress = task_progress(args, done, total_steps)
            raise_if_task_stopped(task_client, args, progress, estimate_task_eta(args, done, total_steps))
            print(f'  Processing [{class_name}] [{qid}]: {Path(qpath).stem}')

            img = Image.open(qpath).convert('RGB')
            orig_size = img.size
            img = img.resize((args.img_size, args.img_size))
            img = transform(img).unsqueeze(0).to(device)

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prev = {}
                for k, v in current_out.items():
                    prev[k] = [t.clone() for t in v] if isinstance(v, list) else v.clone() if isinstance(v, torch.Tensor) else v
                out = model(img, prev_out=prev)

            logit_mask = out['logit_mask']
            logit_mask = F.interpolate(logit_mask, (orig_size[1], orig_size[0]), mode='bilinear', align_corners=True)
            pred_mask = (logit_mask.squeeze() > 0.0).float().cpu().numpy() * 255
            out_path = os.path.join(class_output_dir, f'{qid}.png')
            Image.fromarray(pred_mask.astype(np.uint8)).save(out_path)

            done += 1
            progress = task_progress(args, done, total_steps)
            eta_seconds = estimate_task_eta(args, done, total_steps)
            report_task_progress(task_client, args, progress, eta_seconds, f"已推理 {done}/{total_steps}")

    return 0


def main():
    parser = argparse.ArgumentParser(description='FS-SAM2 Prediction')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--support_dir', type=str, default='', help='Dir with images/ and masks/ subdirs (support examples)')
    parser.add_argument('--query_dir', type=str, default='', help='Dir with query images to segment')
    parser.add_argument('--support_manifest', type=str, default='')
    parser.add_argument('--query_manifest', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='', help='Dir to save predicted masks')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to trained .pt checkpoint')
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoint/sam2.1_hiera_base_plus.pt')
    parser.add_argument('--sam2_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--kshot', type=int, default=1, help='Number of support images to use (default 1)')
    parser.add_argument('--img_size', type=int, default=1024, help='Image size for inference (default 1024)')
    parser.add_argument('--dltool_task_host', type=str, default='')
    parser.add_argument('--dltool_task_port', type=int, default=0)
    parser.add_argument('--dltool_task_id', type=int, default=-1)
    parser.add_argument('--dltool_progress_base', type=int, default=0)
    parser.add_argument('--dltool_progress_span', type=int, default=100)
    parser.add_argument('--dltool_finish_on_complete', action='store_true')
    args = parser.parse_args()
    task_client = create_task_client(args)

    try:
        apply_dltool_config(args)

        if not args.output_dir:
            raise ValueError('output_dir is empty')
        if not args.checkpoint:
            raise ValueError('checkpoint is empty')
        if not args.support_manifest and not args.support_dir:
            raise ValueError('support_manifest is empty')
        if not args.query_manifest and not args.query_dir:
            raise ValueError('query_manifest is empty')

        report_task_status(task_client, args, TaskStatus.RUNNING, args.dltool_progress_base, -1, "开始 FS-SAM2 推理")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_float32_matmul_precision('high')
        os.makedirs(args.output_dir, exist_ok=True)

        # Image transform (same normalization as training)
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(img_mean, img_std),
        ])

        # Load model
        raise_if_task_stopped(task_client, args, args.dltool_progress_base, -1)
        print(f'Loading checkpoint: {args.checkpoint}')
        model = setup_model(args.checkpoint, device, args.sam2_checkpoint, args.sam2_cfg)

        if args.support_manifest or args.query_manifest:
            run_manifest_prediction(args, model, transform, device, task_client)
            if args.dltool_finish_on_complete:
                report_task_status(task_client, args, TaskStatus.FINISHED, 100, 0, "推理完成")
            else:
                report_task_progress(task_client, args, 100, 0, "当前类别推理完成")
            return 0

        # Load support images
        support_imgs, support_masks, _ = load_support(args.support_dir, args.img_size, transform, device,
                                                      task_client, args)

        # Use only kshot support images
        total = len(support_imgs)
        if args.kshot < total:
            support_imgs = support_imgs[:args.kshot]
            support_masks = support_masks[:args.kshot]
            print(f'Using {args.kshot} of {total} support images (controlled by --kshot)')

        # Pre-compute support memory once
        print('Encoding support images...')
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            current_out = {}
            for i in range(len(support_imgs)):
                raise_if_task_stopped(task_client, args, args.dltool_progress_base, -1)
                current_out = model(support_imgs[i].unsqueeze(0), support_masks[i].unsqueeze(0), prev_out=current_out)
        print('Support encoding done.')

        # Build query list — use query.txt if present, else all images in dir
        query_txt = os.path.join(args.query_dir, 'query.txt')
        if os.path.exists(query_txt):
            query_list = []
            with open(query_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    qid, qname = line.split(',', 1)
                    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                        p = os.path.join(args.query_dir, qname + ext)
                        if os.path.exists(p):
                            query_list.append((qid, p))
                            break
                    else:
                        print(f'  Skip (no image): {qname}')
        else:
            query_list = [(Path(p).stem, p) for p in load_queries(args.query_dir)]

        print(f'Found {len(query_list)} query images')
        args.dltool_eta_start_time = time.time()
        for index, (qid, qpath) in enumerate(query_list):
            progress = task_progress(args, index, len(query_list))
            raise_if_task_stopped(task_client, args, progress, estimate_task_eta(args, index, len(query_list)))
            print(f'  Processing [{qid}]: {Path(qpath).stem}')

            img = Image.open(qpath).convert('RGB')
            orig_size = img.size  # (W, H)
            img = img.resize((args.img_size, args.img_size))
            img = transform(img).unsqueeze(0).to(device)

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prev = {}
                for k, v in current_out.items():
                    prev[k] = [t.clone() for t in v] if isinstance(v, list) else v.clone() if isinstance(v, torch.Tensor) else v
                out = model(img, prev_out=prev)

            logit_mask = out['logit_mask']
            logit_mask = F.interpolate(logit_mask, (orig_size[1], orig_size[0]), mode='bilinear', align_corners=True)
            pred_mask = (logit_mask.squeeze() > 0.0).float().cpu().numpy() * 255

            out_path = os.path.join(args.output_dir, f'{qid}.png')
            Image.fromarray(pred_mask.astype(np.uint8)).save(out_path)
            progress = task_progress(args, index + 1, len(query_list))
            eta_seconds = estimate_task_eta(args, index + 1, len(query_list))
            report_task_progress(task_client, args, progress, eta_seconds, f"已推理 {index + 1}/{len(query_list)}")
            raise_if_task_stopped(task_client, args, progress, eta_seconds)
            print(f'    Saved: {out_path}')

        finish_progress = task_progress(args, len(query_list), len(query_list))
        finish_eta = 0 if args.dltool_finish_on_complete else estimate_task_eta(args, len(query_list), len(query_list))
        if args.dltool_finish_on_complete:
            report_task_status(task_client, args, TaskStatus.FINISHED, 100, 0, "推理完成")
        else:
            report_task_progress(task_client, args, finish_progress, finish_eta, "当前类别推理完成")
        return 0
    except TaskStopRequested:
        return 130
    except Exception:
        report_task_status(task_client, args, TaskStatus.FAILED, -1, -1, traceback.format_exc())
        return 1
    finally:
        if task_client is not None:
            task_client.close()


if __name__ == '__main__':
    raise SystemExit(main())
