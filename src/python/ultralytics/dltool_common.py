"""DLTool integration helpers for the vendored Ultralytics package."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT.parent / "task"
PACKAGE_ROOT = ROOT / "ultralytics"
for path in (ROOT, TASK_DIR, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dltool_task_protocol import TaskStatus  # noqa: E402
from dltool_task_reporting import (  # noqa: E402
    TaskStopRequested,
    create_task_client,
    report_failure,
    report_log,
    report_progress,
    report_result,
    report_status,
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
    parser.add_argument("--dltool_task_host", default="")
    parser.add_argument("--dltool_task_port", type=int, default=0)
    parser.add_argument("--dltool_task_id", type=int, default=-1)


def _insert_value(target: dict[str, Any], name: str, value: Any) -> None:
    parts = [part for part in str(name).split(".") if part]
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    if parts:
        current[parts[-1]] = value


def load_params(database_path: str | Path, table: str) -> dict[str, Any]:
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    if table not in {"train_params", "test_params"}:
        raise ValueError(f"unsupported parameter table: {table}")
    result: dict[str, Any] = {}
    with sqlite3.connect(path) as connection:
        rows = connection.execute(f"SELECT name_en, value FROM {table} ORDER BY name_en").fetchall()
    for name, encoded in rows:
        _insert_value(result, str(name), json.loads(encoded))
    return result


def group(values: dict[str, Any], name: str) -> dict[str, Any]:
    value = values.get(name, {})
    return value if isinstance(value, dict) else {}


def scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, list) and value and all(len(str(item)) == 1 for item in value):
        return "".join(str(item) for item in value)
    return value


def text(values: dict[str, Any], name: str, default: str = "") -> str:
    value = scalar(values.get(name, default), default)
    return default if value is None else str(value).strip()


def integer(values: dict[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(scalar(values.get(name, default), default))
    except (TypeError, ValueError):
        return default


def floating(values: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(scalar(values.get(name, default), default))
    except (TypeError, ValueError):
        return default


def boolean(values: dict[str, Any], name: str, default: bool = False) -> bool:
    value = scalar(values.get(name, default), default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_dataset_yaml(dataset_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    from ultralytics.utils import YAML

    path = Path(dataset_dir) / "dataset.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Ultralytics dataset configuration not found: {path}")
    data = YAML.load(path)
    if not isinstance(data, dict):
        raise ValueError(f"invalid Ultralytics dataset configuration: {path}")
    return path, data


def resolve_model_source(args: argparse.Namespace, train_values: dict[str, Any] | None = None) -> str:
    network = group(train_values or {}, "network")
    configured = text(network, "model_path") or text(network, "checkpoint_path")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = Path(args.model_root) / candidate
        if candidate.is_file():
            return str(candidate)

    architecture = args.model_architecture.strip().lower()
    pretrained = text(network, "pretrained", "COCO").lower()
    if pretrained == "none":
        model_name = "yolov8-seg.yaml" if "seg" in architecture else "yolov8.yaml"
        if "yolov5" in architecture:
            model_name = "yolov5.yaml"
        return str(PACKAGE_ROOT / "ultralytics" / "cfg" / "models" / ("v5" if "yolov5" in architecture else "v8") / model_name)
    if "seg" in architecture:
        return "yolov8n-seg.pt"
    if "yolov5" in architecture:
        return "yolov5n.pt"
    return "yolov8n.pt"


def model_task(architecture: str) -> str:
    return "segment" if "seg" in architecture.lower() else "detect"


def install_custom_dataset() -> None:
    """Install the custom dataset class at Ultralytics' existing factory boundary."""
    from dltool_dataset import DlToolYOLODataset
    from ultralytics.data import build

    build.YOLODataset = DlToolYOLODataset


def train_params(args: argparse.Namespace) -> dict[str, Any]:
    return load_params(args.model_db, "train_params")


def test_params(args: argparse.Namespace) -> dict[str, Any]:
    if not args.task_db:
        return {}
    return load_params(args.task_db, "test_params")


def publish_status(client, args, status: TaskStatus, progress: int, message: str, **payload: Any) -> None:
    report_status(client, args, status, progress, 0 if status == TaskStatus.FINISHED else -1, message, **payload)


__all__ = [
    "TaskStopRequested",
    "TaskStatus",
    "add_task_arguments",
    "boolean",
    "create_task_client",
    "floating",
    "group",
    "install_custom_dataset",
    "integer",
    "load_dataset_yaml",
    "model_task",
    "publish_status",
    "report_failure",
    "report_log",
    "report_progress",
    "report_result",
    "resolve_model_source",
    "test_params",
    "text",
    "train_params",
]
