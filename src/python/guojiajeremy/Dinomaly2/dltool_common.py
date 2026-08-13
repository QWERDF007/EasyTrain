"""Shared integration helpers for the DLTool Dinomaly2 mask-constraint task.

Follows the ``open-edge-platform/anomalib`` integration pattern: parses the
task arguments, loads ``train_params``/``test_params`` from the model/task
SQLite databases, builds the Dinomaly2 training data from the exported file
lists and mask files, and reports progress through the shared task protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT.parents[1] / "task"
for path in (ROOT, TASK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dltool_task_protocol import TaskClient, TaskStatus  # noqa: E402
from dltool_task_reporting import (  # noqa: E402
    TaskStopRequested,
    create_task_client,
    report_failure,
    report_log as log,
    report_progress as progress,
    report_result,
    report_status as status,
)
from dltool_task_utils import (  # noqa: E402
    boolean,
    floating,
    format_hms,
    integer,
    is_character_sequence,
    load_params_table,
    parse_int_list,
    scalar,
    select_device,
    should_stop,
    text,
)


def add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_db", required=True)
    parser.add_argument("--task_db", default="")
    parser.add_argument("--project_db", default="")
    parser.add_argument("--model_root", default="")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--train_dir", default="")
    parser.add_argument("--masks_dir", default="")
    parser.add_argument("--test_file_list", default="")
    parser.add_argument("--weight_dir", default="")
    parser.add_argument("--log_dir", default="")
    parser.add_argument("--model_uuid", default="")
    parser.add_argument("--model_architecture", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--prediction_dir", default="")
    parser.add_argument("--good_values", default="")
    parser.add_argument("--anomaly_values", default="")
    parser.add_argument("--ignore_values", default="")
    parser.add_argument("--dltool_task_host", default="")
    parser.add_argument("--dltool_task_port", type=int, default=0)
    parser.add_argument("--dltool_task_id", type=int, default=-1)


def load_database_config(args: argparse.Namespace, section: str) -> dict[str, Any]:
    """Build the runner view from model.db/task.db and the model file lists."""
    train_params = load_params_table(args.model_db, "train_params")
    test_params = load_params_table(args.task_db, "test_params") if args.task_db else {}
    config: dict[str, Any] = {
        "model_uuid": args.model_uuid,
        "model_architecture": args.model_architecture,
        "method": args.method,
        "model_dir": args.model_root,
        "weight_dir": args.weight_dir,
        "log_dir": args.log_dir,
        "result_dir": args.prediction_dir,
        "prediction_dir": args.prediction_dir,
        "train_params": train_params,
        "test_params": test_params,
        "datasets": {},
    }

    dataset_root = Path(args.dataset_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else dataset_root
    train_dir = Path(args.train_dir) if args.train_dir else dataset_root.parent / "train"
    for split in ("train", "validation"):
        split_file = train_dir / f"{split}.txt"
        if split_file.is_file():
            config["datasets"][split] = {
                "file_list": str(split_file),
                "masks_dir": str(masks_dir),
            }
    test_file_list = Path(args.test_file_list) if args.test_file_list else dataset_root / "test.txt"
    if test_file_list.is_file():
        config["datasets"]["test"] = {
            "file_list": str(test_file_list),
            "masks_dir": str(masks_dir),
        }

    if section == "test_params":
        inference = dict(group(config, "test_params", "inference"))
        if args.prediction_dir:
            inference["output_dir"] = args.prediction_dir
        checkpoint = text(inference, "checkpoint")
        if checkpoint and not Path(checkpoint).is_absolute():
            candidates = [Path(args.model_root) / checkpoint, Path(args.weight_dir) / checkpoint]
            checkpoint = str(next((candidate for candidate in candidates if candidate.is_file()), candidates[-1]))
        if checkpoint:
            inference["checkpoint"] = checkpoint
        config["test_params"]["inference"] = inference
    return config


def group(config: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    section_values = config.get(section, {})
    if not isinstance(section_values, dict):
        return {}
    value = section_values.get(name, {})
    return value if isinstance(value, dict) else {}


def mask_value_lists(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    """Resolve good/anomaly/ignore value lists.

    The project's data management computes these from the label class groups
    and passes them as ``--good_values``/``--anomaly_values``/``--ignore_values``.
    When a task is started without them (e.g. manual python runs), the
    configured ``mask.good_value``/``mask.anomaly_value``/``mask.ignore_value``
    params are used as a fallback.
    """
    mask_values = group(config, "train_params", "mask")
    good = parse_int_list(getattr(args, "good_values", ""), None)
    anomaly = parse_int_list(getattr(args, "anomaly_values", ""), None)
    ignore = parse_int_list(getattr(args, "ignore_values", ""), None)
    if not good:
        good = parse_int_list(text(mask_values, "good_value", ""), [1])
    if not anomaly:
        anomaly = parse_int_list(text(mask_values, "anomaly_value", ""), [255])
    if not ignore:
        ignore = parse_int_list(text(mask_values, "ignore_value", ""), [254])
    return good, anomaly, ignore


def dataset_entry(config: dict[str, Any], split: str) -> dict[str, Any]:
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    entry = datasets.get(split, {})
    return entry if isinstance(entry, dict) else {}


def dataset_file_list_path(config: dict[str, Any], split: str) -> str:
    return text(dataset_entry(config, split), "file_list")


def dataset_masks_dir(config: dict[str, Any], split: str) -> str:
    return text(dataset_entry(config, split), "masks_dir")


def load_file_list(path: str | Path) -> list[dict[str, Any]]:
    """Load one exported ``image_id,image_path`` CSV file list."""
    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"dataset file list not found: {list_path}")

    samples: list[dict[str, Any]] = []
    with list_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower() == "image_id" or len(row) < 2:
                continue
            image_id = row[0].strip()
            image_path = row[1].strip()
            if image_id and image_path:
                samples.append({"id": image_id, "path": image_path})
    if not samples:
        raise ValueError(f"dataset file list has no usable images: {list_path}")
    return samples


def file_list_samples(
    config: dict[str, Any],
    split: str,
    required: bool = True,
) -> list[dict[str, Any]]:
    """Load file-list samples for one dataset split."""
    file_list_path = dataset_file_list_path(config, split)
    if not file_list_path:
        if required:
            raise ValueError(f"datasets.{split}.file_list is empty")
        return []
    samples = load_file_list(file_list_path)
    result: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        image_id = text(sample, "id")
        image_path = text(sample, "path")
        if image_id and image_path:
            result.append({"id": image_id, "path": image_path})
    if required and not result:
        raise ValueError(f"dataset file list has no usable samples: {file_list_path}")
    return result


def build_mask_constraint_model(
    network: dict[str, Any],
    training: dict[str, Any],
    device: str,
):
    """Build the Dinomaly2 model with the noisy bottleneck + decoder."""
    from functools import partial

    import torch
    import torch.nn as nn

    from models import vit_encoder
    from models.uad import Dinomaly
    from models.vision_transformer import Attention, LinearAttention2, Block as VitBlock

    backbone = text(network, "backbone", "dinov2reg_vit_base_14")
    loose_constraint = integer(network, "lc", 2)
    dropout = floating(network, "dropout", 0.4)
    use_linear_attention = boolean(network, "la", True)
    context_recentering = boolean(network, "cr", True)

    groups = _layer_groups(loose_constraint)
    fuse_layer_encoder, fuse_layer_decoder = groups
    encoder = vit_encoder.load(backbone)

    if "small" in backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in backbone:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone}")

    bottleneck = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
            nn.Sequential(
                nn.Linear(256, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(p=dropout),
            ),
        ]
    )
    decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=partial(LinearAttention2, eps=1e-8)
                if use_linear_attention
                else Attention,
            )
            for _ in range(8)
        ]
    )
    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=context_recentering,
    ).to(device)
    model.init_weights()
    trainable = torch.nn.ModuleList([bottleneck, decoder])
    return model, trainable


def _layer_groups(loose_constraint: int):
    if loose_constraint == 0:
        groups = [[i] for i in range(8)]
    elif loose_constraint == 1:
        groups = [list(range(8))]
    elif loose_constraint == 2:
        groups = [list(range(4)), list(range(4, 8))]
    elif loose_constraint == 3:
        groups = [list(range(3)), list(range(3, 6)), [6, 7]]
    elif loose_constraint == 4:
        groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elif loose_constraint == 11:
        groups = [[7]]
    elif loose_constraint == 12:
        groups = [[3], [7]]
    elif loose_constraint == 14:
        groups = [[1], [3], [5], [7]]
    else:
        raise ValueError(f"Unsupported loose constraint: {loose_constraint}")
    return groups, [list(group) for group in groups]


def build_train_dataset(
    config: dict[str, Any],
    image_size: int,
    crop_size: int,
    good_values: list[int] | None = None,
    anomaly_values: list[int] | None = None,
    ignore_values: list[int] | None = None,
):
    """Build the mask training dataset from the exported train split."""
    from dataset import get_data_transforms, get_mask_constraint_train_transform

    mask_values = group(config, "train_params", "mask")
    network = group(config, "train_params", "network")

    data_transform, _gt_transform = get_data_transforms(image_size, crop_size)
    joint_transform = get_mask_constraint_train_transform(
        image_size,
        crop_size,
        hflip_prob=floating(mask_values, "aug_hflip_prob", 0.0),
        brightness=floating(mask_values, "aug_brightness", 0.0),
        contrast=floating(mask_values, "aug_contrast", 0.0),
        hue=floating(mask_values, "aug_hue", 0.0),
    )
    samples = file_list_samples(config, "train", required=True)
    masks_dir = dataset_masks_dir(config, "train")

    from mask_constraint_dataset import DltoolMaskConstraintTrainDataset

    dataset = DltoolMaskConstraintTrainDataset(
        samples=samples,
        image_transform=data_transform,
        image_size=image_size,
        crop_size=crop_size,
        masks_dir=masks_dir,
        good_values=good_values or [1],
        anomaly_values=anomaly_values or [255],
        ignore_values=ignore_values or [254],
        joint_transform=joint_transform,
    )
    return dataset


def build_eval_dataset(
    config: dict[str, Any],
    split: str,
    image_size: int,
    crop_size: int,
    required: bool = True,
    anomaly_values: list[int] | None = None,
):
    """Build an MVTec-style evaluation dataset from an exported split."""
    from dataset import get_data_transforms
    from mask_constraint_dataset import DltoolEvalDataset

    samples = file_list_samples(config, split, required=required)
    if not samples:
        return None
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    return DltoolEvalDataset(
        samples=samples,
        image_transform=data_transform,
        gt_transform=gt_transform,
        masks_dir=dataset_masks_dir(config, split),
        anomaly_values=anomaly_values or [255],
    )


class DltoolPredictDataset:
    """Image-only dataset for prediction; returns tensors, image ids and paths."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        image_path = str(sample["path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, str(sample["id"]), image_path


def build_predict_dataset(
    config: dict[str, Any],
    image_size: int,
    crop_size: int,
):
    """Build the test dataset for prediction from the exported test split."""
    from dataset import get_data_transforms

    samples = file_list_samples(config, "test", required=True)
    data_transform, _gt_transform = get_data_transforms(image_size, crop_size)
    return DltoolPredictDataset(samples, data_transform)


def evaluate_validation(
    model,
    config: dict[str, Any],
    device: str,
    batch_size: int,
    image_size: int,
    crop_size: int,
    client: TaskClient | None,
    task_id: int,
    anomaly_values: list[int] | None = None,
) -> dict[str, float] | None:
    """Run Dinomaly2 evaluation on the exported validation split.

    Returns the mean metrics dict, or ``None`` when no validation split was
    exported (or it contains no usable images).
    """
    import numpy as np

    from utils import evaluation_batch

    dataset = build_eval_dataset(
        config,
        "validation",
        image_size,
        crop_size,
        required=False,
        anomaly_values=anomaly_values,
    )
    if dataset is None or len(dataset) == 0:
        return None

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    results = evaluation_batch(
        model,
        dataloader,
        device,
        max_ratio=0.01,
        resize_mask=256,
    )
    names = ("I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO")
    metrics = {name: float(value) for name, value in zip(names, results)}
    print(
        "[Dinomaly2] validation: "
        + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
        flush=True,
    )
    report_result(
        client,
        task_id,
        "验证评估",
        metrics,
    )
    return metrics


class DltoolProgressReporter:
    """Reports task progress/status and checks stop requests."""

    def __init__(self, client: TaskClient | None, task_id: int, label: str):
        self.client = client
        self.task_id = task_id
        self.label = label
        self.current = 0
        self.last_message = ""
        self.last_payload: dict[str, Any] = {}

    @staticmethod
    def _emit(message: str) -> None:
        print(message, flush=True)

    def start(self, message: str, **payload: Any) -> None:
        status(self.client, self.task_id, TaskStatus.RUNNING, 0, -1, message, **payload)
        self._emit(f"{self.label}: {message}")
        self.check_stop()

    def report(self, done: int, total: int, message: str, **payload: Any) -> None:
        total = max(1, int(total))
        done = max(0, min(total, int(done)))
        value = int(100 * done / total)
        self.current = max(self.current, value)
        self._emit(message)
        progress(self.client, self.task_id, self.current, -1, message, **payload)
        self.check_stop()

    def status(self, message: str, **payload: Any) -> None:
        status(self.client, self.task_id, TaskStatus.RUNNING, self.current, -1, message, **payload)
        self._emit(f"{self.label}: {message}")
        self.check_stop()

    def log(self, message: str) -> None:
        log(self.client, self.task_id, message)
        self._emit(message)

    def check_stop(self) -> None:
        if should_stop(self.client, self.task_id):
            status(self.client, self.task_id, TaskStatus.STOPPED, -1, -1, "任务已停止")
            raise TaskStopRequested()

    def finish(self, message: str, **payload: Any) -> None:
        self._emit(f"{self.label}: {message}")
        status(
            self.client,
            self.task_id,
            TaskStatus.FINISHED,
            100,
            0,
            message,
            **payload,
        )

