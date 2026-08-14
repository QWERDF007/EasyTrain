"""Train a detection or instance-segmentation model through Ultralytics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dltool_common import (
    TaskStatus,
    TaskStopRequested,
    add_task_arguments,
    create_task_client,
    estimate_eta,
    format_hms,
    format_number,
    group,
    install_custom_dataset,
    model_task,
    publish_status,
    report_failure,
    report_log,
    report_progress,
    report_result,
    report_status,
    text,
    train_params,
)

# 允许透传给 YOLO.train() 的参数名（name_en 与 ultralytics 参数名一致）。
TRAIN_KWARG_WHITELIST = {
    "epochs",
    "batch",
    "workers",
    "imgsz",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "cos_lr",
    "optimizer",
    "patience",
    "close_mosaic",
    "seed",
    "amp",
    "device",
    "val",
    "mosaic",
    "mixup",
    "copy_paste",
    "cutmix",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
}


def _flatten_params(values: dict) -> dict:
    flat: dict = {}
    for group_values in (values or {}).values():
        if isinstance(group_values, dict):
            flat.update(group_values)
    return flat


def _metrics_text(results) -> str:
    """从 ultralytics train() 返回值提取评估指标文本（仅 metrics/ 前缀项）。"""
    try:
        if hasattr(results, "results_dict"):
            data = results.results_dict
        elif isinstance(results, dict):
            data = results
        else:
            return str(results)
        lines = [f"{key}: {format_number(value)}" for key, value in data.items() if str(key).startswith("metrics/")]
        return "\n".join(lines)
    except Exception:
        return str(results)


class UltralyticsProgressReporter:
    """通过 ultralytics 回调上报训练进度（epoch/iter/loss/lr/进度百分比）。"""

    REPORT_INTERVAL_SECONDS = 0.5

    def __init__(self, client, args: argparse.Namespace, label: str):
        self.client = client
        self.args = args
        self.label = label
        self.start_time = time.monotonic()
        self.last_report_time = 0.0
        self.epoch_total = 1
        self.steps_per_epoch = 1
        self.train_total = 1
        self.last_epoch = 0
        self.last_iter = 0
        self.last_loss = "-"
        self.last_lr = "-"

    def install(self, model) -> None:
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_batch_end", self.on_train_batch_end)
        model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

    def _estimate_eta(self) -> int:
        return estimate_eta(time.monotonic() - self.start_time, self.last_iter, self.train_total)

    def _payload(self, phase_progress: int) -> dict:
        return {
            "phase": "train",
            "started": True,
            "phase_progress": phase_progress,
            "epoch": f"{self.last_epoch} / {self.epoch_total}",
            "iter": f"{self.last_iter} / {self.train_total}",
            "lr": str(self.last_lr),
            "loss": str(self.last_loss),
            "elapsed": format_hms(time.monotonic() - self.start_time),
            "eta": format_hms(self._estimate_eta()),
        }

    def _report(self, progress: int, message: str, phase_progress: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report_time < self.REPORT_INTERVAL_SECONDS:
            return
        self.last_report_time = now
        report_progress(self.client, self.args, progress, -1, message, **self._payload(phase_progress))

    def on_train_start(self, trainer) -> None:
        self.epoch_total = max(1, int(getattr(trainer, "epochs", 1)))
        try:
            self.steps_per_epoch = max(1, int(len(trainer.train_loader)))
        except Exception:
            self.steps_per_epoch = 1
        self.train_total = max(1, self.epoch_total * self.steps_per_epoch)
        report_log(
            self.client,
            self.args,
            f"{self.label}: epochs={self.epoch_total}, 每轮批数={self.steps_per_epoch}",
        )
        report_status(
            self.client,
            self.args,
            TaskStatus.RUNNING,
            0,
            -1,
            "开始 Ultralytics 训练",
            phase="train",
            started=True,
            phase_progress=0,
        )
        self._report(0, "训练中", 0, force=True)

    def on_train_batch_end(self, trainer) -> None:
        if self.client is not None and self.client.should_stop(self.args.dltool_task_id):
            raise TaskStopRequested
        self.last_iter = min(self.train_total, self.last_iter + 1)
        self.last_epoch = min(self.epoch_total, max(1, int(getattr(trainer, "epoch", 0)) + 1))
        try:
            loss = getattr(trainer, "loss", None)
            if loss is not None:
                self.last_loss = f"{loss.item():.6g}" if hasattr(loss, "item") else f"{loss:.6g}"
        except Exception:
            pass
        try:
            self.last_lr = f"{trainer.optimizer.param_groups[0]['lr']:.6g}"
        except Exception:
            pass
        progress = int(100 * self.last_iter / self.train_total)
        phase_progress = int(100 * (self.last_iter % self.steps_per_epoch) / self.steps_per_epoch)
        self._report(progress, "训练中", phase_progress)

    def on_train_epoch_end(self, trainer) -> None:
        self.last_epoch = min(self.epoch_total, max(1, int(getattr(trainer, "epoch", 0)) + 1))
        self.last_iter = min(self.train_total, self.last_epoch * self.steps_per_epoch)
        progress = int(100 * self.last_iter / self.train_total)
        self._report(progress, f"Epoch {self.last_epoch}/{self.epoch_total}", 100, force=True)


def _resolve_checkpoint(args: argparse.Namespace, values: dict) -> str:
    """解析训练初始权重：取 network.checkpoint（官方名或用户权重完整路径）。"""
    network = group(values, "network")
    checkpoint = text(network, "checkpoint")
    if not checkpoint:
        raise ValueError("network.checkpoint is empty")
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = Path(args.model_root) / candidate
    if candidate.is_file():
        return str(candidate)
    return checkpoint


def _collect_finished_weights(weight_dir: str) -> str:
    """返回训练直接落盘的权重：优先 best.pt，其次 last.pt。"""
    output_dir = Path(weight_dir)
    best = output_dir / "best.pt"
    last = output_dir / "last.pt"
    if best.is_file():
        return str(best)
    if last.is_file():
        return str(last)
    raise FileNotFoundError("Ultralytics training did not produce a checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool Ultralytics training entry")
    add_task_arguments(parser)
    args = parser.parse_args()
    client = create_task_client(args)
    try:
        values = train_params(args)
        install_custom_dataset()

        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        # 确保 TensorBoard 回调启用（settings.json 的 tensorboard 开关可能被关闭）。
        try:
            if not SETTINGS["tensorboard"]:
                SETTINGS.update({"tensorboard": True})
        except Exception:
            pass

        source = _resolve_checkpoint(args, values)
        task = model_task(args.model_architecture)
        data_yaml = Path(args.dataset_dir) / "dataset.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"dataset configuration not found: {data_yaml}")

        publish_status(client, args, TaskStatus.RUNNING, 0, "开始 Ultralytics 训练")
        report_log(client, args, f"model={source}, data={data_yaml}, task={task}")
        model = YOLO(source, task=task, verbose=False)

        reporter = UltralyticsProgressReporter(client, args, "Ultralytics")
        reporter.install(model)

        flat = _flatten_params(values)
        kwargs = {key: flat[key] for key in TRAIN_KWARG_WHITELIST if key in flat}
        log_dir = Path(args.log_dir or str(Path(args.weight_dir).parent / "logs"))
        results = model.train(
            data=str(data_yaml),
            task=task,
            **kwargs,
            # 权重直接落 <模型>/train/weights；results.csv、TensorBoard 等日志仍在 <模型>/train/logs。
            weights_dir=str(args.weight_dir),
            project=str(log_dir.parent),
            name=log_dir.name,
            exist_ok=True,
            verbose=False,
        )
        checkpoint = _collect_finished_weights(args.weight_dir)

        metrics = _metrics_text(results)
        report_result(client, args, "训练结果", {"checkpoint": checkpoint, "results": metrics})
        publish_status(client, args, TaskStatus.FINISHED, 100, "Ultralytics 训练完成", checkpoint=checkpoint,
                       metrics=metrics)
        return 0
    except TaskStopRequested:
        return 2
    except Exception:
        report_failure(client, args, "Ultralytics 训练")
        return 1
    finally:
        if client is not None:
            client.close()

