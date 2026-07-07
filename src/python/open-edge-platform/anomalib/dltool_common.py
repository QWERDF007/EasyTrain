import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
TASK_DIR = ROOT.parents[1] / "task"
for path in (SRC_DIR, TASK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dltool_task_protocol import TaskClient, TaskStatus  # noqa: E402


class TaskStopRequested(Exception):
    pass


def add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--dltool_task_host", default="")
    parser.add_argument("--dltool_task_port", type=int, default=0)
    parser.add_argument("--dltool_task_id", type=int, default=-1)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def group(config: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    value = config.get(section, {}).get(name, {})
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
    value = scalar(values.get(name))
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return default or []


def square_size(values: dict[str, Any], name: str, default: int) -> tuple[int, int]:
    size = integer(values, name, default)
    return size, size


def create_task_client(args: argparse.Namespace) -> TaskClient | None:
    if not args.dltool_task_host or args.dltool_task_port <= 0 or args.dltool_task_id < 0:
        return None
    return TaskClient(args.dltool_task_host, args.dltool_task_port)


def status(client: TaskClient | None, task_id: int, task_status: TaskStatus, progress: int, eta: int, message: str) -> None:
    if client is not None:
        client.status(task_id, task_status, progress, eta, message)


def progress(client: TaskClient | None, task_id: int, value: int, eta: int, message: str) -> None:
    if client is not None:
        client.progress(task_id, value, eta, message)


def log(client: TaskClient | None, task_id: int, message: str) -> None:
    if client is not None and message:
        client.log(task_id, message)


def should_stop(client: TaskClient | None, task_id: int) -> bool:
    return client is not None and client.should_stop(task_id)


def build_datamodule(config: dict[str, Any], section: str):
    data = group(config, section, "data")
    if section == "train_params":
        train_samples = dltool_file_list_samples(config, "train", split="train", normal_only=True)
        validation_samples = dltool_file_list_samples(config, "validation", split="test", required=False)
        return DltoolCustomDataModule(
            name=text(data, "name", "dltool"),
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=validation_samples,
            train_batch_size=integer(data, "train_batch_size", 32),
            eval_batch_size=integer(data, "eval_batch_size", 32),
            num_workers=integer(data, "num_workers", 8),
        )

    test_samples = dltool_file_list_samples(config, "test", split="test")
    return DltoolCustomDataModule(
        name=text(data, "name", "dltool"),
        train_samples=[],
        validation_samples=[],
        test_samples=test_samples,
        train_batch_size=integer(data, "train_batch_size", 32),
        eval_batch_size=integer(data, "eval_batch_size", 32),
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
    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"dataset file list not found: {list_path}")
    with list_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"dataset file list is not a mapping: {list_path}")
    return loaded


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

    file_list = load_file_list(file_list_path)
    masks_dir = dataset_masks_dir(config, dataset_split) or text(file_list, "masks_dir")
    samples = file_list.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"dataset file list samples is not a list: {file_list_path}")

    result: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        image_id = text(sample, "id")
        image_path = text(sample, "path")
        if not image_path:
            continue
        label_index = integer(sample, "label_index", 0)
        if normal_only and label_index != 0:
            continue

        mask_name = text(sample, "mask")
        mask_path = str(Path(masks_dir) / mask_name) if mask_name and masks_dir else ""
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


def build_model(config: dict[str, Any], section: str = "train_params"):
    architecture = str(config.get("model_architecture", "")).strip().lower()
    model_params = group(config, section, "model") or group(config, "train_params", "model")

    if architecture == "patchcore":
        from anomalib.models import Patchcore

        crop = integer(model_params, "center_crop_size", 0)
        pre_processor = Patchcore.configure_pre_processor(
            image_size=square_size(model_params, "image_size", 256),
            center_crop_size=(crop, crop) if crop > 0 else None,
        )
        return Patchcore(
            backbone=text(model_params, "backbone", "wide_resnet50_2"),
            layers=string_list(model_params, "layers", ["layer2", "layer3"]),
            pre_trained=boolean(model_params, "pre_trained", True),
            coreset_sampling_ratio=floating(model_params, "coreset_sampling_ratio", 0.1),
            num_neighbors=integer(model_params, "num_neighbors", 9),
            precision=text(model_params, "precision", "float32"),
            pre_processor=pre_processor,
        )

    if architecture == "dinomaly2":
        from anomalib.models import Dinomaly

        pre_processor = Dinomaly.configure_pre_processor(
            image_size=square_size(model_params, "image_size", 448),
            crop_size=integer(model_params, "crop_size", 392),
        )
        return Dinomaly(
            encoder_name=text(model_params, "encoder_name", "dinov2reg_vit_base_14"),
            decoder_depth=integer(model_params, "decoder_depth", 8),
            bottleneck_dropout=floating(model_params, "bottleneck_dropout", 0.2),
            use_context_recentering=boolean(model_params, "use_context_recentering", True),
            precision=text(model_params, "precision", "float32"),
            pre_processor=pre_processor,
        )

    raise ValueError(f"Unsupported anomalib architecture: {architecture}")


def build_engine(config: dict[str, Any], section: str, callback):
    from anomalib.engine import Engine

    trainer = group(config, section, "trainer") or group(config, section, "inference")
    kwargs: dict[str, Any] = {
        "callbacks": [callback],
        "default_root_dir": text(trainer, "output_dir", "results"),
        "accelerator": text(trainer, "accelerator", "auto"),
        "devices": integer(trainer, "devices", 1),
        "num_sanity_val_steps": integer(trainer, "num_sanity_val_steps", 0),
    }
    max_epochs = integer(trainer, "max_epochs", 0)
    max_steps = integer(trainer, "max_steps", 0)
    if max_epochs > 0:
        kwargs["max_epochs"] = max_epochs
    if max_steps > 0:
        kwargs["max_steps"] = max_steps
    return Engine(**kwargs)


class DltoolProgressCallback:
    def __init__(self, client: TaskClient | None, task_id: int, label: str):
        from lightning.pytorch.callbacks import Callback

        class CallbackImpl(Callback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def on_train_start(self, trainer, pl_module):
                self.outer.report_status("开始训练")

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                self.outer.check_stop()

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                total = max(1, int(getattr(trainer, "estimated_stepping_batches", 1) or 1))
                done = max(0, int(getattr(trainer, "global_step", 0)))
                self.outer.update(done, total, "训练中")

            def on_validation_start(self, trainer, pl_module):
                self.outer.validation_base = self.outer.completed
                self.outer.report_status("开始验证")

            def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.outer.check_stop()

            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                batches = getattr(trainer, "num_val_batches", [1]) or [1]
                total = max(1, int(batches[0] or 1))
                self.outer.update(self.outer.validation_base + batch_idx + 1, self.outer.validation_base + total, "验证中")

            def on_test_start(self, trainer, pl_module):
                self.outer.report_status("开始测试")

            def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.outer.check_stop()

            def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                total = max(1, int(getattr(trainer, "num_test_batches", [1])[0] or 1))
                self.outer.update(batch_idx + 1, total, "测试中")

            def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.outer.check_stop()

            def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                total = max(1, int(getattr(trainer, "num_predict_batches", [1])[0] or 1))
                self.outer.update(batch_idx + 1, total, "预测中")

        self.client = client
        self.task_id = task_id
        self.label = label
        self.start_time = time.time()
        self.completed = 0
        self.validation_base = 0
        self.current_progress = 0
        self.callback = CallbackImpl(self)

    def update(self, done: int, total: int, message: str) -> None:
        done = max(self.completed, done)
        self.completed = done
        value = min(98, max(0, int(98 * done / max(1, total))))
        self.current_progress = max(self.current_progress, value)
        eta = -1
        elapsed = time.time() - self.start_time
        if done > 0 and elapsed > 0 and total > done:
            eta = int(round(elapsed * (total - done) / done))
        progress(self.client, self.task_id, self.current_progress, eta, message)
        self.check_stop()

    def report_status(self, message: str) -> None:
        status(self.client, self.task_id, TaskStatus.RUNNING, self.current_progress, -1, message)
        self.check_stop()

    def check_stop(self) -> None:
        if should_stop(self.client, self.task_id):
            status(self.client, self.task_id, TaskStatus.STOPPED, -1, -1, "任务已停止")
            raise TaskStopRequested()
