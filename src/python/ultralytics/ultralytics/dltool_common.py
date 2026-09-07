"""DLTool integration helpers for the vendored Ultralytics package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FRAMEWORK_ROOT = ROOT.parent
TASK_DIR = ROOT.parents[1] / "task"
for path in (ROOT, FRAMEWORK_ROOT, TASK_DIR):
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
from dltool_task_utils import (  # noqa: E402
    boolean,
    estimate_eta,
    floating,
    format_hms,
    format_number,
    integer,
    load_params_table as load_params,
    scalar,
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
    parser.add_argument("--dltool_task_host", default="")
    parser.add_argument("--dltool_task_port", type=int, default=0)
    parser.add_argument("--dltool_task_id", type=int, default=-1)
    parser.add_argument("--dltool_project_id", default="")
    parser.add_argument("--dltool_run_id", default="")


def group(values: dict[str, Any], name: str) -> dict[str, Any]:
    value = values.get(name, {})
    return value if isinstance(value, dict) else {}


def load_dataset_yaml(dataset_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    from ultralytics.utils import YAML

    path = Path(dataset_dir) / "dataset.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Ultralytics dataset configuration not found: {path}")
    data = YAML.load(path)
    if not isinstance(data, dict):
        raise ValueError(f"invalid Ultralytics dataset configuration: {path}")
    return path, data


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
    "test_params",
    "text",
    "train_params",
]

