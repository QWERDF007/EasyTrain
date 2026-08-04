import argparse
import csv
import json
import math
import sys
import sqlite3
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
TASK_DIR = ROOT.parents[1] / "task"
for path in (SRC_DIR, TASK_DIR):
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


def _load_params(database_path: str | Path, table: str) -> dict[str, Any]:
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


def load_database_config(args: argparse.Namespace, section: str) -> dict[str, Any]:
    """Build the runner view from model.db/task.db and the model file lists."""
    train_params = _load_params(args.model_db, "train_params")
    test_params = _load_params(args.task_db, "test_params") if args.task_db else {}
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
    masks_dir = Path(args.masks_dir) if args.masks_dir else dataset_root / "masks"
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
        checkpoint = text(inference, "checkpoint_path")
        if checkpoint and not Path(checkpoint).is_absolute():
            candidates = [Path(args.model_root) / checkpoint, Path(args.weight_dir) / checkpoint]
            checkpoint = str(next((candidate for candidate in candidates if candidate.is_file()), candidates[-1]))
            inference["checkpoint_path"] = checkpoint
        config["test_params"]["inference"] = inference
    return config


def group(config: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    section_values = config.get(section, {})
    if not isinstance(section_values, dict):
        return {}
    value = section_values.get(name, {})
    return value if isinstance(value, dict) else {}


def is_character_sequence(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(len(str(item)) == 1 for item in value)


def scalar(value: Any, default: str = "") -> Any:
    if value is None:
        return default
    if is_character_sequence(value):
        return "".join(str(item) for item in value)
    return value


def text(values: dict[str, Any], name: str, default: str = "") -> str:
    value = scalar(values.get(name, default), default)
    return default if value is None else str(value).strip()


def optional_text(values: dict[str, Any], name: str) -> str | None:
    value = text(values, name)
    return value or None


def integer(values: dict[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(scalar(values.get(name, default), str(default)))
    except (TypeError, ValueError):
        return default


def floating(values: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(scalar(values.get(name, default), str(default)))
    except (TypeError, ValueError):
        return default


def boolean(values: dict[str, Any], name: str, default: bool = False) -> bool:
    value = scalar(values.get(name, default), str(default))
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def string_list(values: dict[str, Any], name: str, default: list[str] | None = None) -> list[str]:
    fallback = list(default or [])
    if not isinstance(values, dict) or name not in values or values.get(name) is None:
        return fallback

    value = scalar(values.get(name))
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        return result or fallback
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
        return result or fallback
    return fallback


def square_size(values: dict[str, Any], name: str, default: int) -> tuple[int, int]:
    size = integer(values, name, default)
    return size, size


def batch_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            try:
                total += max(0, int(item))
            except (TypeError, ValueError, OverflowError):
                continue
        return total
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def dataloader_batch_count(value: Any) -> int:
    try:
        loader = value() if callable(value) else value
        return batch_count(len(loader))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0




def should_stop(client: TaskClient | None, task_id: int) -> bool:
    return client is not None and client.should_stop(task_id)


def build_datamodule(config: dict[str, Any], section: str):
    data_group = "training" if section == "train_params" else "data"
    data = group(config, section, data_group)
    batch_size = integer(data, "batch_size", 32)
    eval_batch_size = batch_size
    if section == "train_params":
        train_samples = dltool_file_list_samples(config, "train", split="train", normal_only=True)
        validation_samples = dltool_file_list_samples(config, "validation", split="test", required=False)
        return DltoolCustomDataModule(
            name=text(data, "name", "dltool"),
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=validation_samples,
            train_batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=integer(data, "num_workers", 8),
        )

    test_samples = dltool_file_list_samples(config, "test", split="test")
    return DltoolCustomDataModule(
        name=text(data, "name", "dltool"),
        train_samples=[],
        validation_samples=[],
        test_samples=test_samples,
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=integer(data, "num_workers", 8),
    )


class DltoolCustomDataModule:
    def __new__(
        cls,
        name: str,
        train_samples: list[dict[str, Any]],
        validation_samples: list[dict[str, Any]],
        test_samples: list[dict[str, Any]],
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
    ):
        from anomalib.data.datamodules.base.image import AnomalibDataModule
        from anomalib.data.datasets.image.tabular import TabularDataset
        from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
        from pandas import DataFrame

        class DataModuleImpl(AnomalibDataModule):
            def __init__(self) -> None:
                self._name = name
                self._train_samples = train_samples
                self._validation_samples = validation_samples
                self._test_samples = test_samples
                super().__init__(
                    train_batch_size=train_batch_size,
                    eval_batch_size=eval_batch_size,
                    num_workers=num_workers,
                    test_split_mode=TestSplitMode.FROM_DIR,
                    val_split_mode=ValSplitMode.FROM_DIR,
                )

            @property
            def name(self) -> str:
                return self._name

            def _dataset(self, samples: list[dict[str, Any]], split):
                if samples:
                    table = DataFrame(samples)
                else:
                    table = DataFrame(columns=["id", "image_path", "label_index", "split", "mask_path"])
                return TabularDataset(name=self.name, samples=table, split=split, root=None)

            @property
            def test_image_ids(self) -> dict[str, str]:
                """Return database image IDs keyed by their exported image paths."""
                return {
                    str(sample["image_path"]): str(sample["id"])
                    for sample in self._test_samples
                    if sample.get("image_path") and sample.get("id") is not None
                }

            def _setup(self, _stage: str | None = None) -> None:
                self.train_data = self._dataset(self._train_samples, Split.TRAIN)
                self.test_data = self._dataset(self._test_samples, Split.TEST)
                self.val_data = self._dataset(self._validation_samples, Split.TEST)

            def val_dataloader(self):
                if not self._validation_samples:
                    return None
                return super().val_dataloader()

            def _create_test_split(self) -> None:
                return

            def _create_val_split(self) -> None:
                return

        return DataModuleImpl()


def dataset_entry(config: dict[str, Any], split: str) -> dict[str, Any]:
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    entry = datasets.get(split, {})
    if not isinstance(entry, dict):
        return {}
    return entry


def dataset_file_list_path(config: dict[str, Any], split: str) -> str:
    return text(dataset_entry(config, split), "file_list")


def dataset_masks_dir(config: dict[str, Any], split: str) -> str:
    return text(dataset_entry(config, split), "masks_dir")


def load_file_list(path: str | Path) -> dict[str, Any]:
    """Load one exported CSV image list (used for the test split)."""
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
    return {"samples": samples}


def dltool_file_list_samples(
    config: dict[str, Any],
    dataset_split: str,
    split: str,
    required: bool = True,
    normal_only: bool = False,
) -> list[dict[str, Any]]:
    file_list_path = dataset_file_list_path(config, dataset_split)
    if not file_list_path:
        if required:
            raise ValueError(f"datasets.{dataset_split}.file_list is empty")
        return []

    masks_dir = dataset_masks_dir(config, dataset_split)
    # Test labels are deliberately not exported.  Ground truth is rebuilt by
    # the C++ evaluator from task.db and the project database.
    file_list = load_file_list(file_list_path)
    samples = file_list.get("samples", [])

    result: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        image_id = text(sample, "id")
        image_path = text(sample, "path")
        if not image_id or not image_path:
            continue
        if dataset_split == "test":
            label_index = integer(sample, "label_index", 0)
            mask_name = text(sample, "mask")
            mask_path = str(Path(masks_dir) / mask_name) if mask_name and masks_dir else ""
        else:
            # 异常由掩膜是否存在决定：找到 {image_id}.png 即为异常图，否则为正常图。
            # 掩膜像素 0 表示背景，255 表示异常区域。
            mask_path = str(Path(masks_dir) / f"{image_id}.png") if masks_dir else ""
            if mask_path and not Path(mask_path).is_file():
                mask_path = ""
            label_index = 1 if mask_path else 0
        if normal_only and label_index != 0:
            continue
        result.append(
            {
                "id": image_id,
                "image_path": image_path,
                "label_index": label_index,
                "split": split,
                "mask_path": mask_path,
            }
        )

    if required and not result:
        raise ValueError(f"dataset file list has no usable samples: {file_list_path}")
    return result


def build_model(
    config: dict[str, Any],
    section: str = "train_params",
    visualizer: bool = True,
):
    architecture = str(config.get("model_architecture", "")).strip().lower()
    model_params = group(config, "train_params", "network")
    training_params = group(config, "train_params", "training")

    if architecture == "patchcore":
        from anomalib.models import Patchcore

        if integer(model_params, "image_channels", 3) != 3:
            raise ValueError("PatchCore currently supports RGB images with 3 channels only")
        crop = integer(model_params, "center_crop_size", 0)
        pre_processor = Patchcore.configure_pre_processor(
            image_size=square_size(model_params, "image_size", 256),
            center_crop_size=(crop, crop) if crop > 0 else None,
        )
        return Patchcore(
            backbone=text(model_params, "backbone", "wide_resnet50_2"),
            layers=string_list(model_params, "layers", ["layer2", "layer3"]),
            pre_trained=boolean(model_params, "pre_trained", True),
            coreset_sampling_ratio=floating(
                training_params,
                "coreset_sampling_ratio",
                floating(model_params, "coreset_sampling_ratio", 0.1),
            ),
            num_neighbors=integer(
                training_params,
                "num_neighbors",
                integer(model_params, "num_neighbors", 9),
            ),
            precision=text(training_params, "precision", text(model_params, "precision", "float32")),
            pre_processor=pre_processor,
            visualizer=visualizer,
        )

    if architecture == "dinomaly2":
        from anomalib.models import Dinomaly
        from anomalib.metrics import AUPRO, Evaluator

        if integer(model_params, "image_channels", 3) != 3:
            raise ValueError("Dinomaly currently supports RGB images with 3 channels only")
        pre_processor = Dinomaly.configure_pre_processor(
            image_size=square_size(model_params, "image_size", 448),
            crop_size=integer(model_params, "crop_size", 392),
        )
        default_evaluator = Dinomaly.configure_evaluator()
        evaluator = Evaluator(
            val_metrics=list(default_evaluator.val_metrics),
            test_metrics=[
                *list(default_evaluator.test_metrics),
                AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
            ],
            compute_on_cpu=default_evaluator.compute_on_cpu,
        )
        return Dinomaly(
            encoder_name=text(model_params, "encoder_name", "dinov2reg_vit_base_14"),
            decoder_depth=integer(training_params, "decoder_depth", integer(model_params, "decoder_depth", 8)),
            bottleneck_dropout=floating(
                training_params,
                "bottleneck_dropout",
                floating(model_params, "bottleneck_dropout", 0.2),
            ),
            use_context_recentering=boolean(
                training_params,
                "use_context_recentering",
                boolean(model_params, "use_context_recentering", True),
            ),
            precision=text(training_params, "precision", text(model_params, "precision", "float32")),
            learning_rate=floating(training_params, "learning_rate", 2e-3),
            pre_processor=pre_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

    raise ValueError(f"Unsupported anomalib architecture: {architecture}")


def build_engine(config: dict[str, Any], section: str, callback):
    from anomalib.callbacks.checkpoint import ModelCheckpoint
    from anomalib.engine import Engine
    from anomalib.loggers import AnomalibTensorBoardLogger

    runtime_group = "training" if section == "train_params" else "inference"
    runtime_params = group(config, section, runtime_group)
    accelerator = "auto"
    devices: int | list[int] = 1
    selected_device = text(runtime_params, "device", "auto").lower()
    if selected_device.startswith(("cuda:", "gpu:")):
        try:
            gpu_index = int(selected_device.split(":", 1)[1])
            if gpu_index >= 0:
                accelerator = "cuda"
                devices = [gpu_index]
        except (TypeError, ValueError):
            pass
    elif selected_device in {"cuda", "gpu"}:
        accelerator = "cuda"
    elif selected_device == "cpu" or selected_device.startswith("cpu:"):
        accelerator = "cpu"
        devices = 1

    log_dir = text(config, "log_dir", "logs")
    tensorboard_logger = AnomalibTensorBoardLogger(save_dir=log_dir, name="", version="")
    callbacks = [callback]
    default_root_dir = text(runtime_params, "output_dir", "results")
    if section == "train_params":
        # Keep the checkpoint at the model-level weights directory. Anomalib's
        # default callback puts it below the trainer workspace in results.
        weight_dir = text(config, "weight_dir")
        if not weight_dir:
            model_dir = text(config, "model_dir")
            if model_dir:
                weight_dir = str(Path(model_dir) / "weights")
            else:
                output_dir = text(runtime_params, "output_dir", "results")
                weight_dir = str(Path(output_dir).parent / "weights")
        # Keep the trainer workspace out of results during training as well.
        default_root_dir = weight_dir
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=Path(weight_dir),
                filename="model",
                auto_insert_metric_name=False,
                save_top_k=1,
                save_last=False,
                enable_version_counter=False,
            ),
        )
    kwargs: dict[str, Any] = {
        "callbacks": callbacks,
        "logger": tensorboard_logger,
        "default_root_dir": default_root_dir,
        "accelerator": accelerator,
        "devices": devices,
        "num_sanity_val_steps": integer(runtime_params, "num_sanity_val_steps", 0),
    }
    max_epochs = integer(runtime_params, "max_epochs", 0)
    max_steps = integer(runtime_params, "max_steps", 0)
    if max_epochs > 0:
        kwargs["max_epochs"] = max_epochs
    if max_steps > 0:
        kwargs["max_steps"] = max_steps
    return Engine(**kwargs)


def metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_value(item) for item in value]
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def metrics_payload(results: Any) -> dict[str, Any]:
    if isinstance(results, list):
        merged: dict[str, Any] = {}
        for result in results:
            normalized = metric_value(result)
            if isinstance(normalized, dict):
                merged.update(normalized)
        return merged
    normalized = metric_value(results)
    return normalized if isinstance(normalized, dict) else {"value": normalized}


def metrics_text(results: Any) -> str:
    values = metrics_payload(results)
    return "\n".join(f"{key}: {value}" for key, value in values.items())


def status_value_text(value: Any, digits: int = 6) -> str:
    normalized = metric_value(value)
    if normalized is None:
        return "-"
    if isinstance(normalized, float):
        return f"{normalized:.{digits}f}"
    return str(normalized)


def elapsed_text(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if value < 0:
        return "-"
    total = max(0, int(round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class DltoolProgressCallback:
    REPORT_INTERVAL_SECONDS = 0.0
    FIT_PROGRESS_END = 90
    TEST_PROGRESS_END = 98

    def __init__(self, client: TaskClient | None, task_id: int, label: str):
        from lightning.pytorch.callbacks import Callback

        class CallbackImpl(Callback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def on_fit_start(self, trainer, pl_module):
                self.outer.configure_fit(trainer)
                self.outer.report_runtime(trainer, "训练")
                self.outer.report_status("开始训练", **self.outer.training_payload(-1))

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                self.outer.check_stop()

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                done = max(0, int(getattr(trainer, "global_step", 0)))
                self.outer.update_fit(trainer, done, outputs, "训练中", batch_idx=batch_idx)

            def on_validation_start(self, trainer, pl_module):
                self.outer.report_status("开始验证", phase="train", started=True)

            def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.outer.check_stop()

            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                self.outer.validation_done += 1
                train_done = max(0, int(getattr(trainer, "global_step", 0)))
                self.outer.update_fit(trainer, train_done, outputs, "验证中")

            def on_fit_end(self, trainer, pl_module):
                self.outer.update_fit(trainer, self.outer.train_total, None, "训练完成", force=True)

            def on_test_start(self, trainer, pl_module):
                self.outer.configure_test(trainer)
                self.outer.report_runtime(trainer, "测试")
                self.outer.report_status("开始测试", phase="test", started=True, phase_progress=0)

            def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.outer.check_stop()

            def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                self.outer.test_done += 1
                self.outer.update_test(trainer, self.outer.test_done, "测试中")

            def on_test_end(self, trainer, pl_module):
                self.outer.update_test(trainer, self.outer.test_total, "测试完成", force=True)
                self.outer.report_summary("测试")

            def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.outer.check_stop()

            def on_predict_start(self, trainer, pl_module):
                self.outer.configure_predict(trainer)
                self.outer.report_runtime(trainer, "预测")
                self.outer.report_status("开始预测", phase="test", started=True, phase_progress=0)

            def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                self.outer.predict_done += 1
                self.outer.update_predict(trainer, self.outer.predict_done, "预测中")

            def on_predict_end(self, trainer, pl_module):
                self.outer.update_predict(trainer, self.outer.predict_total, "预测完成", force=True)
                self.outer.report_summary("预测")

        self.client = client
        self.task_id = task_id
        self.label = label
        self.start_time = time.monotonic()
        self.phase_start_time = self.start_time
        self.current_progress = 0
        self.train_total = 1
        self.validation_total = 0
        self.validation_done = 0
        self.fit_total = 1
        self.epoch_total = 1
        self.train_batches_per_epoch = 1
        self.steps_per_epoch = 1
        self.iter_total = 1
        self.reported_train_batches: Any = 0
        self.train_batches_source = "回退值"
        self.accumulation = 1
        self.max_epochs = -1
        self.max_steps = -1
        self.estimated_stepping_batches: Any = 0
        self.last_epoch_current = 0
        self.last_iter_current = 0
        self.last_phase_progress = 0
        self.last_lr = None
        self.last_loss = None
        self.test_total = 1
        self.test_done = 0
        self.predict_total = 1
        self.predict_done = 0
        self.last_report_time = 0.0
        self.last_report_message = ""
        self.report_count = 0
        self.fit_start_time = self.start_time
        self.callback = CallbackImpl(self)

    def configure_fit(self, trainer) -> None:
        self.reported_train_batches = getattr(trainer, "num_training_batches", 0)
        train_batches = batch_count(self.reported_train_batches)
        self.train_batches_source = "trainer.num_training_batches"
        if train_batches <= 0:
            for name, value in (
                ("trainer.train_dataloader", getattr(trainer, "train_dataloader", None)),
                (
                    "datamodule.train_dataloader",
                    getattr(getattr(trainer, "datamodule", None), "train_dataloader", None),
                ),
            ):
                train_batches = dataloader_batch_count(value)
                if train_batches > 0:
                    self.train_batches_source = name
                    break
        train_batches = max(1, train_batches)
        self.train_batches_per_epoch = train_batches
        self.accumulation = batch_count(getattr(trainer, "accumulate_grad_batches", 1)) or 1
        self.steps_per_epoch = max(1, math.ceil(train_batches / self.accumulation))
        self.max_epochs = getattr(trainer, "max_epochs", -1)
        self.max_steps = getattr(trainer, "max_steps", -1)
        self.estimated_stepping_batches = getattr(trainer, "estimated_stepping_batches", 0)
        self.train_total = batch_count(self.estimated_stepping_batches)
        if self.train_total <= 0:
            self.train_total = batch_count(self.max_steps)
        if self.train_total <= 0:
            self.train_total = batch_count(self.max_epochs) * self.steps_per_epoch
        self.train_total = max(1, self.train_total)
        estimated_epochs = max(1, math.ceil(self.train_total / self.steps_per_epoch))
        self.epoch_total = estimated_epochs
        self.iter_total = self.train_total
        validation_batches = batch_count(getattr(trainer, "num_val_batches", 0))
        check_every = getattr(trainer, "check_val_every_n_epoch", 1)
        if isinstance(check_every, int) and check_every > 0:
            validation_runs = math.ceil(estimated_epochs / check_every)
        else:
            validation_runs = estimated_epochs
        self.validation_total = validation_batches * validation_runs
        self.validation_done = 0
        self.fit_total = max(1, self.train_total + self.validation_total)
        self.fit_start_time = time.monotonic()
        self.phase_start_time = self.fit_start_time
        self.last_epoch_current = 0
        self.last_iter_current = 0
        self.last_phase_progress = 0

    def configure_test(self, trainer) -> None:
        self.test_total = max(1, batch_count(getattr(trainer, "num_test_batches", 1)))
        self.test_done = 0
        self.phase_start_time = time.monotonic()

    def configure_predict(self, trainer) -> None:
        self.predict_total = max(1, batch_count(getattr(trainer, "num_predict_batches", 1)))
        self.predict_done = 0
        self.phase_start_time = time.monotonic()

    def report_runtime(self, trainer, phase: str) -> None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
            gpu_name = torch.cuda.get_device_name(0) if gpu_count else "无"
        except Exception as exc:
            cuda_available = False
            gpu_count = 0
            gpu_name = f"查询失败: {exc}"

        strategy = getattr(trainer, "strategy", None)
        root_device = getattr(strategy, "root_device", "未知")
        datamodule = getattr(trainer, "datamodule", None)
        workers = getattr(datamodule, "num_workers", "未知")
        train_batch_size = getattr(datamodule, "train_batch_size", "未知")
        eval_batch_size = getattr(datamodule, "eval_batch_size", "未知")
        report_interval = (
            "每个batch"
            if self.REPORT_INTERVAL_SECONDS <= 0
            else f"{self.REPORT_INTERVAL_SECONDS:.1f}秒"
        )
        log(
            self.client,
            self.task_id,
            f"{self.label}: {phase}设备={root_device}, CUDA可用={cuda_available}, "
            f"GPU数量={gpu_count}, GPU={gpu_name}, 加载线程={workers}, "
            f"训练批量={train_batch_size}, 评估批量={eval_batch_size}, "
            f"进度上报间隔={report_interval}（每个batch检查停止）",
        )
        if phase == "训练":
            log(
                self.client,
                self.task_id,
                f"{self.label}: 原始训练批次={self.reported_train_batches!r}, "
                f"有效每轮训练批次={self.train_batches_per_epoch}（来源={self.train_batches_source}），"
                f"梯度累积={self.accumulation}，每轮优化步数={self.steps_per_epoch}，"
                f"max_epochs={self.max_epochs}，max_steps={self.max_steps}，"
                f"estimated_stepping_batches={self.estimated_stepping_batches!r}；"
                f"状态 Epoch 总数={self.epoch_total}，状态 Iter 总数={self.iter_total}，"
                f"预计验证批次={self.validation_total}，总工作量={self.fit_total}；"
                f"Iter 与 TensorBoard 使用同一 global_step，训练阶段最多报告{self.FIT_PROGRESS_END}%",
            )

    def update_fit(self, trainer, train_done: int, outputs: Any, message: str, batch_idx: int | None = None,
                   force: bool = False) -> None:
        if batch_idx is not None:
            self.last_epoch_current = min(
                self.epoch_total, max(1, int(getattr(trainer, "current_epoch", 0)) + 1)
            )
            global_step = batch_count(getattr(trainer, "global_step", train_done))
            self.last_iter_current = min(
                self.iter_total,
                max(0, global_step),
            )
            self.last_phase_progress = int(100 * self.last_iter_current / max(1, self.iter_total))
        elif force:
            self.last_epoch_current = self.epoch_total
            self.last_iter_current = self.iter_total
            self.last_phase_progress = 100

        self.last_loss = self.extract_loss(outputs, self.last_loss)
        self.last_lr = self.extract_lr(trainer, self.last_lr)
        done = max(0, min(self.train_total, int(train_done))) + min(self.validation_done, self.validation_total)
        eta_seconds = self.estimate_eta(done, self.fit_total)
        self.update(
            done,
            self.fit_total,
            message,
            0,
            self.FIT_PROGRESS_END,
            force,
            self.training_payload(eta_seconds),
        )

    def update_test(self, trainer, done: int, message: str, force: bool = False) -> None:
        self.update(
            done,
            self.test_total,
            message,
            self.FIT_PROGRESS_END,
            self.TEST_PROGRESS_END,
            force,
            self.evaluation_payload("test", done, self.test_total),
        )

    def update_predict(self, trainer, done: int, message: str, force: bool = False) -> None:
        self.update(
            done,
            self.predict_total,
            message,
            0,
            self.TEST_PROGRESS_END,
            force,
            self.evaluation_payload("test", done, self.predict_total),
        )

    @staticmethod
    def extract_loss(outputs: Any, fallback: Any = None) -> Any:
        value = outputs
        if isinstance(outputs, dict):
            value = next((outputs[key] for key in ("loss", "train_loss", "val_loss") if key in outputs), None)
        if value is None:
            return fallback
        normalized = metric_value(value)
        return normalized if isinstance(normalized, (int, float)) else fallback

    @staticmethod
    def extract_lr(trainer, fallback: Any = None) -> Any:
        try:
            optimizers = getattr(trainer, "optimizers", [])
            if optimizers:
                return float(optimizers[0].param_groups[0]["lr"])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        return fallback

    def training_payload(self, eta_seconds: int = -1) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase": "train",
            "started": True,
            "phase_progress": self.last_phase_progress,
            "epoch": f"{self.last_epoch_current} / {self.epoch_total}",
            "iter": f"{self.last_iter_current} / {self.iter_total}",
            "lr": status_value_text(self.last_lr),
            "loss": status_value_text(self.last_loss),
            "elapsed": elapsed_text(time.monotonic() - self.fit_start_time),
            "eta": elapsed_text(eta_seconds),
        }
        return payload

    def estimate_eta(self, done: int, total: int) -> int:
        total = max(1, int(total))
        done = max(0, min(total, int(done)))
        if done >= total:
            return 0

        elapsed = time.monotonic() - self.phase_start_time
        if done <= 0 or elapsed <= 0:
            return -1
        return int(round(elapsed * (total - done) / done))

    @staticmethod
    def evaluation_payload(phase: str, done: int, total: int) -> dict[str, Any]:
        bounded_done = max(0, min(int(total), int(done)))
        return {
            "phase": phase,
            "started": True,
            "phase_progress": int(100 * bounded_done / max(1, int(total))),
        }

    def report_summary(self, phase: str) -> None:
        report_interval = (
            "每个batch"
            if self.REPORT_INTERVAL_SECONDS <= 0
            else f"{self.REPORT_INTERVAL_SECONDS:.1f}秒"
        )
        log(
            self.client,
            self.task_id,
            f"{self.label}: {phase}阶段进度上报完成，实际发送{self.report_count}次，"
            f"间隔规则为{report_interval}或阶段切换/完成时立即发送",
        )

    def update(
        self,
        done: int,
        total: int,
        message: str,
        progress_start: int,
        progress_end: int,
        force: bool = False,
        payload: dict[str, Any] | None = None,
    ) -> None:
        total = max(1, int(total))
        done = max(0, min(total, int(done)))
        value = progress_start + int((progress_end - progress_start) * done / total)
        value = min(progress_end, max(progress_start, value))
        self.current_progress = max(self.current_progress, value)
        eta = self.estimate_eta(done, total)
        now = time.monotonic()
        should_report = (
            force
            or self.last_report_time <= 0
            or self.REPORT_INTERVAL_SECONDS <= 0
            or now - self.last_report_time >= self.REPORT_INTERVAL_SECONDS
            or message != self.last_report_message
            or done >= total
        )
        if should_report:
            progress(self.client, self.task_id, self.current_progress, eta, message, **(payload or {}))
            self.last_report_time = now
            self.last_report_message = message
            self.report_count += 1
        self.check_stop()

    def report_status(self, message: str, **payload: Any) -> None:
        status(self.client, self.task_id, TaskStatus.RUNNING, self.current_progress, -1, message, **payload)
        self.check_stop()

    def check_stop(self) -> None:
        if should_stop(self.client, self.task_id):
            status(self.client, self.task_id, TaskStatus.STOPPED, -1, -1, "任务已停止")
            raise TaskStopRequested()
