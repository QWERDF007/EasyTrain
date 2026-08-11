"""Run Ultralytics inference and write the common DLTool prediction protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from dltool_common import (
    TaskStatus,
    add_task_arguments,
    create_task_client,
    floating,
    group,
    model_task,
    publish_status,
    report_failure,
    report_result,
    resolve_model_source,
    test_params,
    text,
    train_params,
    integer,
    load_dataset_yaml,
)


def read_test_records(path: str | Path) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row or not any(value.strip() for value in row):
                continue
            if row[0].strip().lower() == "image_id":
                continue
            if len(row) < 2:
                continue
            try:
                image_id = int(row[0].strip())
            except ValueError:
                continue
            image_path = Path(row[1].strip())
            if not image_path.is_absolute():
                image_path = Path(path).parent / image_path
            records.append((image_id, str(image_path)))
    if not records:
        raise ValueError(f"test file list has no records: {path}")
    return records


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _as_list(value: Any) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else list(value)


def _class_name(result, class_index: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(class_index, names.get(str(class_index), class_index)))
    if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


def prediction_records(result, image_id: int, task: str, class_ids: dict[int, int]) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None:
        return []
    xyxy = _as_list(boxes.xyxy.cpu())
    classes = _as_list(boxes.cls.cpu())
    scores = _as_list(boxes.conf.cpu())
    polygons = result.masks.xy if task == "segment" and result.masks is not None else []
    records: list[dict[str, Any]] = []
    for index, (box, class_id, score) in enumerate(zip(xyxy, classes, scores)):
        x1, y1, x2, y2 = (_as_float(item) for item in box[:4])
        class_index = int(_as_float(class_id))
        item: dict[str, Any] = {
            "prediction_id": f"{image_id}-{index + 1}",
            "image_id": image_id,
            "class_id": int(class_ids.get(class_index, class_index)),
            "class_name": _class_name(result, class_index),
            "score": _as_float(score),
            "geometry": {
                "type": "bbox",
                "format": "xywh",
                "coordinate_system": "image_pixels",
                "values": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            },
        }
        if index < len(polygons):
            points = [[_as_float(point[0]), _as_float(point[1])] for point in polygons[index]]
            if len(points) >= 3:
                item["geometry"] = {
                    "type": "polygon",
                    "format": "points",
                    "coordinate_system": "image_pixels",
                    "points": points,
                    "bounds": {
                        "x": x1,
                        "y": y1,
                        "width": max(0.0, x2 - x1),
                        "height": max(0.0, y2 - y1),
                    },
                }
        records.append(item)
    return records


def write_predictions(task_db: str, records: dict[int, list[dict[str, Any]]]) -> int:
    if not task_db:
        raise ValueError("task_db is empty")
    rows = [(image_id, json.dumps(values, ensure_ascii=False)) for image_id, values in records.items()]
    with sqlite3.connect(task_db) as connection:
        connection.executemany(
            "INSERT INTO prediction (image_id, data) VALUES (?, ?) "
            "ON CONFLICT(image_id) DO UPDATE SET data=excluded.data",
            rows,
        )
    return sum(len(values) for values in records.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool Ultralytics prediction entry")
    add_task_arguments(parser)
    args = parser.parse_args()
    client = create_task_client(args)
    try:
        train_values = train_params(args)
        test_values = test_params(args)
        inference = group(test_values, "inference")
        records = read_test_records(args.test_file_list)
        _, dataset_config = load_dataset_yaml(args.dataset_dir)
        class_ids = {
            int(key): int(value)
            for key, value in (dataset_config.get("class_ids", {}) or {}).items()
        }
        checkpoint = text(inference, "checkpoint_path")
        if checkpoint and not Path(checkpoint).is_absolute():
            checkpoint = str(Path(args.model_root) / checkpoint)
        if not checkpoint or not Path(checkpoint).is_file():
            checkpoint = str(Path(args.weight_dir) / "model.pt")
        if not Path(checkpoint).is_file():
            checkpoint = resolve_model_source(args, train_values)

        from ultralytics import YOLO

        task = model_task(args.model_architecture)
        model = YOLO(checkpoint, task=task, verbose=False)
        publish_status(client, args, TaskStatus.RUNNING, 0, "开始 Ultralytics 推理", task=task)
        results = model.predict(
            source=[path for _, path in records],
            task=task,
            imgsz=integer(inference, "image_size", 640),
            conf=floating(inference, "confidence_threshold", 0.25),
            iou=floating(inference, "nms_threshold", 0.45),
            save=False,
            verbose=False,
        )
        by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id, _ in records}
        for (image_id, _), result in zip(records, results):
            by_image[image_id] = prediction_records(result, image_id, task, class_ids)
        prediction_count = write_predictions(args.task_db, by_image)
        report_result(client, args, "预测结果", {"prediction_count": prediction_count})
        publish_status(client, args, TaskStatus.FINISHED, 100, "Ultralytics 推理完成", prediction_count=prediction_count)
        return 0
    except Exception:
        report_failure(client, args, "Ultralytics 推理")
        return 1
    finally:
        if client is not None:
            client.close()
