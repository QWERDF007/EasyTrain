r""" box_to_mask.py: Generate segmentation masks from bounding box annotations
using SAM2 with box prompts.

For each image in a class directory, reads bounding boxes from boxes.json,
feeds them as box prompts to SAM2, and saves the resulting masks.
Multiple boxes per image are merged with logical OR.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import traceback
import torch
import PIL.Image as Image

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


def find_image_file(image_dir, alias):
    """Find an image file by alias (stem) in the image directory."""
    for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
        path = os.path.join(image_dir, alias + ext)
        if os.path.exists(path):
            return path
    return None


def load_boxes(support_dir):
    """Load bounding boxes from boxes.json. Returns dict {alias: [box_xywh, ...]}."""
    boxes_path = os.path.join(support_dir, 'boxes.json')
    if not os.path.exists(boxes_path):
        raise FileNotFoundError(f'boxes.json not found in {support_dir}')
    with open(boxes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data:
        print(f'Warning: boxes.json is empty in {support_dir}')
    return data


def setup_sam2_predictor(checkpoint_path, model_cfg, device):
    """Load SAM2 base model and wrap with SAM2ImagePredictor."""
    # Add facebookresearch sam2 to path
    sam2_root = Path(__file__).resolve().parents[2] / "facebookresearch" / "sam2"
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Resolve config path relative to sam2 configs
    cfg_path = os.path.join(sam2_root, 'sam2', 'configs', model_cfg)
    if not os.path.exists(cfg_path):
        cfg_path = model_cfg

    sam2_model = build_sam2(cfg_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def main():
    parser = argparse.ArgumentParser(description='SAM2 Box-to-Mask')
    parser.add_argument('--support_dir', type=str, required=True,
                        help='Class directory with images/ and boxes.json')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--sam2_cfg', type=str, required=True,
                        help='Path to SAM2 config YAML')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Image size (default 1024)')
    parser.add_argument('--dltool_task_host', type=str, default='')
    parser.add_argument('--dltool_task_port', type=int, default=0)
    parser.add_argument('--dltool_task_id', type=int, default=-1)
    parser.add_argument('--dltool_progress_base', type=int, default=0)
    parser.add_argument('--dltool_progress_span', type=int, default=100)
    args = parser.parse_args()

    task_client = create_task_client(args)
    report_task_status(task_client, args, TaskStatus.RUNNING, args.dltool_progress_base,
                       -1, "开始 SAM2 框转Mask")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    image_dir = os.path.join(args.support_dir, 'images')
    mask_dir = os.path.join(args.support_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    if not os.path.isdir(image_dir):
        report_task_status(task_client, args, TaskStatus.FAILED,
                           -1,
                           -1,
                           message=f"Images directory not found: {image_dir}")
        return 1

    try:
        raise_if_task_stopped(task_client, args, args.dltool_progress_base, -1)

        # Load boxes
        boxes_data = load_boxes(args.support_dir)
        if not boxes_data:
            print(f'No boxes found in {args.support_dir}, skipping.')
            report_task_status(task_client, args, TaskStatus.FINISHED,
                               args.dltool_progress_base + args.dltool_progress_span,
                               0,
                               "无需处理的框")
            return 0

        # Load SAM2 predictor
        print(f'Loading SAM2 model: {args.sam2_checkpoint}')
        predictor = setup_sam2_predictor(args.sam2_checkpoint, args.sam2_cfg, device)
        print('SAM2 model loaded.')

        total = len(boxes_data)
        args.dltool_eta_start_time = time.time()
        for idx, (alias, box_list) in enumerate(boxes_data.items()):
            progress = task_progress(args, idx, total)
            raise_if_task_stopped(task_client, args, progress, estimate_task_eta(args, idx, total))

            # Find image file
            img_path = find_image_file(image_dir, alias)
            if img_path is None:
                print(f'  Skip (no image): {alias}')
                done_progress = task_progress(args, idx + 1, total)
                report_task_progress(task_client, args, done_progress,
                                     estimate_task_eta(args, idx + 1, total),
                                     f"跳过 (无图像): {alias}")
                continue

            # Load image
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            print(f'  Processing [{alias}]: {img.size[0]}x{img.size[1]}, {len(box_list)} box(es)')

            # Resize image for SAM2
            predictor.set_image(np.array(img))

            all_masks = []
            for box_idx, box_xywh in enumerate(box_list):
                x, y, w, h = (box_xywh['x'], box_xywh['y'],
                              box_xywh['width'], box_xywh['height'])
                # Convert to XYXY
                box_xyxy = np.array([x, y, x + w, y + h])

                try:
                    masks, ious, _ = predictor.predict(
                        box=box_xyxy,
                        multimask_output=True,
                    )
                    # Select best mask by IoU
                    best_idx = int(np.argmax(ious))
                    mask = masks[best_idx]
                    all_masks.append(mask)
                except Exception as exc:
                    print(f'    Box {box_idx} failed: {exc}')
                    continue

            if not all_masks:
                print(f'    No valid masks generated for {alias}, saving blank mask')
                merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
            else:
                # Merge all masks with logical OR and scale to original resolution
                merged = np.logical_or.reduce(all_masks).astype(np.uint8) * 255

            # Save mask
            out_path = os.path.join(mask_dir, alias + '.png')
            Image.fromarray(merged).save(out_path)

            done_progress = task_progress(args, idx + 1, total)
            report_task_progress(task_client, args, done_progress,
                                 estimate_task_eta(args, idx + 1, total),
                                 f"已处理 {idx + 1}/{total}")
            print(f'    Saved: {out_path}')

        finish_progress = task_progress(args, total, total)
        report_task_status(task_client, args, TaskStatus.FINISHED,
                           finish_progress, 0, f"框转Mask完成 ({total} 张)")
        return 0
    except TaskStopRequested:
        return 130
    except Exception:
        report_task_status(task_client, args, TaskStatus.FAILED,
                           -1,
                           -1,
                           message=traceback.format_exc())
        return 1
    finally:
        if task_client is not None:
            task_client.close()


if __name__ == '__main__':
    raise SystemExit(main())
