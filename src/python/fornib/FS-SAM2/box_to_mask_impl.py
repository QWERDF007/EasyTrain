r"""Generate custom manifest label masks from bounding boxes using SAM2."""
import argparse
import os
import sys
import time

from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
import yaml

TASK_DIR = Path(__file__).resolve().parents[2] / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from dltool_task_protocol import TaskStatus
from dltool_task_reporting import (
    TaskStopRequested,
    create_task_client,
    report_failure,
    report_progress as report_task_progress,
    report_status as report_task_status,
)

FS_SAM2_DIR = Path(__file__).resolve().parent
if str(FS_SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(FS_SAM2_DIR))

from prediction_augmentation import (
    bounded,
    predict_mask_with_augmentation,
    transform_prediction_box,
)


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




def raise_if_task_stopped(client, args, progress=-1, eta_seconds=-1):
    if client is not None and client.should_stop(args.dltool_task_id):
        report_task_status(client, args, TaskStatus.STOPPED, progress, eta_seconds, "任务已停止")
        raise TaskStopRequested()


def load_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a mapping: {manifest_path}")
    images = manifest.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"manifest images is not a list: {manifest_path}")
    return manifest


def save_manifest(path, manifest):
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True, sort_keys=False)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def group(values, *keys):
    current = values
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def text(values, name, default=""):
    value = values.get(name, default) if isinstance(values, dict) else default
    return default if value is None else str(value).strip()


def floating(values, name, default=0.0):
    try:
        return float(values.get(name, default)) if isinstance(values, dict) else default
    except (TypeError, ValueError):
        return default


def boolean(values, name, default=False):
    if not isinstance(values, dict) or name not in values:
        return default
    value = values.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def apply_dltool_config(args):
    if not args.config:
        return []

    config = load_config(args.config)
    datasets = group(config, "datasets")
    manifests = []
    for split_name in ("train", "validation"):
        manifest = text(group(datasets, split_name), "manifest")
        if manifest:
            manifests.append(manifest)
    if not manifests:
        raise ValueError("datasets.train.manifest is empty")

    model = group(config, "train_params", "model")
    args.sam2_checkpoint = text(model, "sam2_checkpoint", args.sam2_checkpoint)
    args.sam2_cfg = text(model, "sam2_cfg", args.sam2_cfg)
    args.box_to_mask_prediction_enhancement_enabled = boolean(
        model,
        "box_to_mask_prediction_enhancement_enabled",
        args.box_to_mask_prediction_enhancement_enabled,
    )
    args.prediction_horizontal_flip = boolean(model, "prediction_horizontal_flip", args.prediction_horizontal_flip)
    args.prediction_vertical_flip = boolean(model, "prediction_vertical_flip", args.prediction_vertical_flip)
    args.prediction_scale = bounded(floating(model, "prediction_scale", args.prediction_scale), -0.9, 1.0)
    args.prediction_brightness = bounded(floating(model, "prediction_brightness", args.prediction_brightness), -1.0, 1.0)
    args.prediction_contrast = bounded(floating(model, "prediction_contrast", args.prediction_contrast), -1.0, 1.0)
    args.prediction_hue = bounded(floating(model, "prediction_hue", args.prediction_hue), -0.5, 0.5)
    args.prediction_rotation = bounded(floating(model, "prediction_rotation", args.prediction_rotation), -180.0, 180.0)
    args.prediction_iou_threshold = bounded(
        floating(model, "prediction_iou_threshold", args.prediction_iou_threshold), 0.0, 1.0
    )
    args.prediction_min_vote_count = int(floating(model, "prediction_min_vote_count", args.prediction_min_vote_count))
    return manifests


def setup_sam2_predictor(checkpoint_path, model_cfg, device):
    sam2_root = Path(__file__).resolve().parents[2] / "facebookresearch" / "sam2"
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg_path = sam2_root / "sam2" / "configs" / model_cfg
    if not cfg_path.exists():
        cfg_path = Path(model_cfg)

    sam2_model = build_sam2(str(cfg_path), checkpoint_path, device=device)
    return SAM2ImagePredictor(sam2_model)


def numeric(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def box_from_mapping(values):
    if not isinstance(values, dict):
        return None
    width = numeric(values.get("width"))
    height = numeric(values.get("height"))
    if width <= 0 or height <= 0:
        return None
    x = numeric(values.get("x"))
    y = numeric(values.get("y"))
    return [x, y, x + width, y + height]


def box_from_points(points):
    if not isinstance(points, list) or len(points) < 2:
        return None

    xs = []
    ys = []
    for point in points:
        if isinstance(point, dict):
            xs.append(numeric(point.get("x")))
            ys.append(numeric(point.get("y")))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(numeric(point[0]))
            ys.append(numeric(point[1]))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def label_box(label):
    data = label.get("data", {}) if isinstance(label, dict) else {}
    box = box_from_mapping(data) or box_from_mapping(label)
    if box is None and isinstance(data, dict):
        box = box_from_points(data.get("points"))
    return box


def clamp_box(box, image_size):
    width, height = image_size
    x1 = min(max(float(box[0]), 0.0), float(width - 1))
    y1 = min(max(float(box[1]), 0.0), float(height - 1))
    x2 = min(max(float(box[2]), 0.0), float(width - 1))
    y2 = min(max(float(box[3]), 0.0), float(height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def default_mask_path(manifest_path, image, label):
    image_id = str(image.get("id", Path(str(image.get("path", "image"))).stem)).strip() or "image"
    label_id = str(label.get("label_id", label.get("label_class_id", "label"))).strip() or "label"
    masks_dir = Path(manifest_path).parent / "masks"
    return str(masks_dir / f"image_{image_id}_label_{label_id}.png")


def collect_label_entries(manifest, manifest_path):
    entries = []
    for image in manifest.get("images", []):
        if not isinstance(image, dict):
            continue
        image_path = str(image.get("path", "")).strip()
        if not image_path:
            continue
        labels = image.get("labels", [])
        if not isinstance(labels, list):
            continue
        for label in labels:
            if not isinstance(label, dict):
                continue
            box = label_box(label)
            if box is None:
                continue
            mask_path = str(label.get("mask_path", "")).strip()
            if not mask_path:
                mask_path = default_mask_path(manifest_path, image, label)
                label["mask_path"] = mask_path
            entries.append((image, label, image_path, mask_path, box))
    return entries


def predict_box_mask(predictor, box):
    """Run the original SAM2 box prompt and select its highest-IoU mask."""
    masks, ious, _ = predictor.predict(box=box, multimask_output=True)
    best_idx = int(np.argmax(ious))
    return masks[best_idx].astype(np.uint8) * 255


def predict_box_mask_with_augmentation(predictor, image_pil, box, args, task_client, progress):
    """Predict a box mask through the shared fixed-TTA workflow."""
    original_image = np.asarray(image_pil)
    try:
        predictor.set_image(original_image)
        return predict_mask_with_augmentation(
            image_pil,
            args,
            lambda _image: predict_box_mask(predictor, box),
            lambda augmented_image, augmentation: _predict_augmented_box_mask(
                predictor, augmented_image, augmentation, box, image_pil.size
            ),
            enabled_attribute="box_to_mask_prediction_enhancement_enabled",
            check_stopped=lambda augmentation_progress: raise_if_task_stopped(
                task_client, args, augmentation_progress, -1
            ),
            progress=progress,
        )
    finally:
        predictor.set_image(original_image)


def _predict_augmented_box_mask(predictor, augmented_image, augmentation, box, image_size):
    predictor.set_image(np.asarray(augmented_image))
    augmented_box = transform_prediction_box(box, augmentation, image_size)
    augmented_box = clamp_box(augmented_box, image_size)
    if augmented_box is None:
        return None
    return predict_box_mask(predictor, augmented_box)


def generate_masks(args, manifest, task_client, finish_on_complete=True):
    entries = collect_label_entries(manifest, args.manifest)
    if not entries:
        if finish_on_complete:
            report_task_status(task_client, args, TaskStatus.FINISHED, task_progress(args, 1, 1), 0, "无需处理的框")
        else:
            report_task_progress(task_client, args, task_progress(args, 1, 1), 0, "无需处理的框")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    print(f"Loading SAM2 model: {args.sam2_checkpoint}")
    predictor = setup_sam2_predictor(args.sam2_checkpoint, args.sam2_cfg, device)
    print("SAM2 model loaded.")

    args.dltool_eta_start_time = time.time()
    done = 0
    current_image_path = None
    current_image_size = None

    for image, label, image_path, mask_path, box in entries:
        progress = task_progress(args, done, len(entries))
        raise_if_task_stopped(task_client, args, progress, estimate_task_eta(args, done, len(entries)))

        if image_path != current_image_path:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
            image_pil = Image.open(image_path).convert("RGB")
            current_image_size = image_pil.size
            predictor.set_image(np.array(image_pil))
            current_image_path = image_path

        box_xyxy = clamp_box(box, current_image_size)
        if box_xyxy is None:
            raise ValueError(f"invalid box for image {image.get('id', image_path)} label {label.get('label_id', '')}")

        mask = predict_box_mask_with_augmentation(predictor, image_pil, box_xyxy, args, task_client, progress)

        output_path = Path(mask_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(output_path)

        done += 1
        done_progress = task_progress(args, done, len(entries))
        eta_seconds = estimate_task_eta(args, done, len(entries))
        report_task_progress(task_client, args, done_progress, eta_seconds, f"已处理 {done}/{len(entries)}")
        print(f"Saved label mask: {output_path}")

    output_manifest = args.output_manifest or args.manifest
    save_manifest(output_manifest, manifest)
    if finish_on_complete:
        report_task_status(task_client, args, TaskStatus.FINISHED, task_progress(args, len(entries), len(entries)), 0,
                           f"BoxToMask 完成 ({len(entries)} 个标注)")
    else:
        report_task_progress(task_client, args, task_progress(args, len(entries), len(entries)), 0,
                             f"BoxToMask 完成 ({len(entries)} 个标注)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="SAM2 box-to-mask for FS-SAM2 custom manifest")
    parser.add_argument("--config", type=str, default="", help="DLTool task config YAML")
    parser.add_argument("--manifest", type=str, default="", help="FS-SAM2 custom manifest YAML")
    parser.add_argument("--output_manifest", type=str, default="", help="Updated manifest path. Defaults to overwrite.")
    parser.add_argument("--sam2_checkpoint", type=str, default="", help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_cfg", type=str, default="", help="Path to SAM2 config YAML")
    parser.add_argument("--box_to_mask_prediction_enhancement_enabled", action="store_true", default=False)
    parser.add_argument("--prediction_horizontal_flip", action="store_true", default=False)
    parser.add_argument("--prediction_vertical_flip", action="store_true", default=False)
    parser.add_argument("--prediction_scale", type=float, default=0.0)
    parser.add_argument("--prediction_brightness", type=float, default=0.0)
    parser.add_argument("--prediction_contrast", type=float, default=0.0)
    parser.add_argument("--prediction_hue", type=float, default=0.0)
    parser.add_argument("--prediction_rotation", type=float, default=0.0)
    parser.add_argument("--prediction_iou_threshold", type=float, default=0.5)
    parser.add_argument("--prediction_min_vote_count", type=int, default=2)
    parser.add_argument("--dltool_task_host", type=str, default="")
    parser.add_argument("--dltool_task_port", type=int, default=0)
    parser.add_argument("--dltool_task_id", type=int, default=-1)
    parser.add_argument("--dltool_progress_base", type=int, default=0)
    parser.add_argument("--dltool_progress_span", type=int, default=100)
    args = parser.parse_args()

    task_client = create_task_client(args)
    try:
        report_task_status(task_client, args, TaskStatus.RUNNING, args.dltool_progress_base, -1, "开始 SAM2 BoxToMask")
        manifests = apply_dltool_config(args)
        if not manifests:
            if not args.manifest:
                raise ValueError("manifest is empty")
            manifests = [args.manifest]
        if not args.sam2_checkpoint:
            raise ValueError("sam2_checkpoint is empty")
        if not args.sam2_cfg:
            raise ValueError("sam2_cfg is empty")

        original_base = args.dltool_progress_base
        original_span = args.dltool_progress_span
        for index, manifest_path in enumerate(manifests):
            split_base = original_base + original_span * index // max(1, len(manifests))
            split_end = original_base + original_span * (index + 1) // max(1, len(manifests))
            args.dltool_progress_base = split_base
            args.dltool_progress_span = max(1, split_end - split_base)
            args.manifest = manifest_path
            args.output_manifest = manifest_path
            manifest = load_manifest(args.manifest)
            generate_masks(args, manifest, task_client, finish_on_complete=index == len(manifests) - 1)
        return 0
    except TaskStopRequested:
        return 130
    except Exception:
        report_failure(task_client, args, "BoxToMask")
        return 1
    finally:
        if task_client is not None:
            task_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
