"""Train a detection or instance-segmentation model through Ultralytics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dltool_common import (
    TaskStatus,
    add_task_arguments,
    boolean,
    create_task_client,
    floating,
    group,
    install_custom_dataset,
    integer,
    model_task,
    publish_status,
    report_failure,
    report_log,
    report_result,
    resolve_model_source,
    text,
    train_params,
)


def _copy_finished_weights(model, weight_dir: str) -> str:
    output_dir = Path(weight_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(getattr(model, "trainer", None).save_dir) if getattr(model, "trainer", None) else None
    candidates = []
    if save_dir:
        candidates.extend([save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"])
    candidates.extend([output_dir / "best.pt", output_dir / "last.pt"])
    source = next((item for item in candidates if item.is_file()), None)
    if source is None:
        raise FileNotFoundError("Ultralytics training did not produce a checkpoint")
    target = output_dir / "model.pt"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool Ultralytics training entry")
    add_task_arguments(parser)
    args = parser.parse_args()
    client = create_task_client(args)
    try:
        values = train_params(args)
        network = group(values, "network")
        training = group(values, "training")
        augmentation = group(values, "augmentation")
        install_custom_dataset()

        from ultralytics import YOLO

        source = resolve_model_source(args, values)
        task = model_task(args.model_architecture)
        data_yaml = Path(args.dataset_dir) / "dataset.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"dataset configuration not found: {data_yaml}")

        publish_status(client, args, TaskStatus.RUNNING, 0, "开始 Ultralytics 训练", task=task)
        report_log(client, args, f"model={source}, data={data_yaml}, task={task}")
        model = YOLO(source, task=task, verbose=False)
        results = model.train(
            data=str(data_yaml),
            task=task,
            epochs=integer(training, "epochs", 100),
            batch=integer(training, "batch_size", 16),
            imgsz=integer(network, "image_size", 640),
            lr0=floating(training, "learning_rate", 0.01),
            optimizer=text(training, "optimizer", "auto"),
            mosaic=1.0 if boolean(augmentation, "mosaic", True) else 0.0,
            mixup=1.0 if boolean(augmentation, "mixup", False) else 0.0,
            project=args.log_dir or str(Path(args.weight_dir).parent / "logs"),
            name="ultralytics",
            exist_ok=True,
            verbose=False,
        )
        checkpoint = _copy_finished_weights(model, args.weight_dir)

        export_format = text(network, "export_format")
        if export_format and export_format.lower() not in {"none", "null", "off"}:
            export_path = model.export(format=export_format, project=args.weight_dir, name="export", exist_ok=True)
            report_result(client, args, "导出结果", {"format": export_format, "path": str(export_path)})
        report_result(client, args, "训练结果", {"checkpoint": checkpoint, "results": str(results)})
        publish_status(client, args, TaskStatus.FINISHED, 100, "Ultralytics 训练完成", checkpoint=checkpoint)
        return 0
    except Exception:
        report_failure(client, args, "Ultralytics 训练")
        return 1
    finally:
        if client is not None:
            client.close()
