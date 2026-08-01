import argparse
from pathlib import Path

from dltool_common import (
    DltoolProgressCallback,
    TaskStatus,
    TaskStopRequested,
    add_task_arguments,
    build_datamodule,
    build_engine,
    build_model,
    create_task_client,
    group,
    load_config,
    report_failure,
    status,
    text,
)


def build_image_id_lookup(image_ids: dict[str, str]) -> dict[str, str]:
    """Index exported image paths across Windows/POSIX spelling variants."""
    lookup: dict[str, str] = {}

    def add(value: str, image_id: str) -> None:
        value = str(value).strip()
        if not value:
            return
        for variant in {value, value.replace("\\", "/"), value.replace("/", "\\")}:
            lookup.setdefault(variant, image_id)
            lookup.setdefault(variant.casefold(), image_id)
        try:
            resolved = Path(value).expanduser().resolve(strict=False)
            resolved_text = str(resolved)
            for variant in {resolved_text, resolved_text.replace("\\", "/"), resolved_text.replace("/", "\\")}:
                lookup.setdefault(variant, image_id)
                lookup.setdefault(variant.casefold(), image_id)
        except (OSError, RuntimeError, ValueError):
            pass

    for path, image_id in image_ids.items():
        add(path, str(image_id))
    return lookup


def lookup_image_id(image_lookup: dict[str, str], image_path: object) -> str | None:
    value = str(image_path).strip()
    if not value:
        return None
    candidates = [value, value.replace("\\", "/"), value.replace("/", "\\")]
    try:
        resolved = Path(value).expanduser().resolve(strict=False)
        resolved_text = str(resolved)
        candidates.extend(
            [resolved_text, resolved_text.replace("\\", "/"), resolved_text.replace("/", "\\")]
        )
    except (OSError, RuntimeError, ValueError):
        pass
    for candidate in candidates:
        image_id = image_lookup.get(candidate) or image_lookup.get(candidate.casefold())
        if image_id:
            return image_id
    return None


def save_prediction_score_maps(predictions, output_dir: str, image_ids: dict[str, str]) -> int:
    """Save raw model anomaly scores as float32 TIFF files."""
    import numpy as np
    import tifffile

    if predictions is None:
        return 0
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]

    image_lookup = build_image_id_lookup(image_ids)
    saved = 0
    for batch in predictions:
        for item in batch:
            anomaly_map = getattr(item, "anomaly_map", None)
            image_path = getattr(item, "image_path", None)
            if anomaly_map is None or not image_path:
                continue

            if hasattr(anomaly_map, "detach"):
                anomaly_map = anomaly_map.detach().cpu().numpy()
            score_map = np.asarray(anomaly_map, dtype=np.float32).squeeze()
            if score_map.ndim != 2:
                raise ValueError(
                    f"Expected a 2D anomaly score map for {image_path}, got shape {score_map.shape}"
                )

            image_id = lookup_image_id(image_lookup, image_path)
            if not image_id:
                raise ValueError(f"Missing database image_id for predicted image: {image_path}")

            # The C++ storage service already passes the task's final pred/
            # directory.  Do not append a second pred component here.
            output_path = Path(output_dir) / f"{image_id}.tiff"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_path, score_map)
            saved += 1
    return saved


def save_normalized_manifest(
    predictions,
    output_dir: str,
    image_ids: dict[str, str],
    model_uuid: str,
    test_task_uuid: str,
    method: str,
) -> int:
    """Persist image-level anomaly scores using the shared PRED protocol."""
    import yaml

    if predictions is None:
        predictions = []
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]
    image_lookup = build_image_id_lookup(image_ids)
    records = []
    serial = 0
    for batch in predictions:
        for item in batch:
            image_path = getattr(item, "image_path", None)
            if not image_path:
                continue
            if not isinstance(image_path, str):
                image_path = str(image_path)
            image_id = lookup_image_id(image_lookup, image_path)
            if not image_id:
                continue
            score = getattr(item, "pred_score", getattr(item, "anomaly_score", 0.0))
            if hasattr(score, "detach"):
                score = score.detach().cpu().item()
            serial += 1
            records.append(
                {
                    "prediction_id": f"pred-{serial}",
                    "image_id": int(image_id),
                    "class_id": 1,
                    "class_name": "anomaly",
                    "score": float(score),
                }
            )
    manifest = {
        "schema_version": 1,
        "model_uuid": model_uuid,
        "test_task_uuid": test_task_uuid,
        "method": method,
        "record_count": len(records),
        "records": records,
    }
    output_path = Path(output_dir) / "manifest.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, allow_unicode=True, sort_keys=False)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool anomalib prediction entry")
    add_task_arguments(parser)
    args = parser.parse_args()

    client = create_task_client(args)
    try:
        config = load_config(args.config)
        inference = group(config, "test_params", "inference")
        checkpoint_path = text(inference, "checkpoint_path")
        if not checkpoint_path:
            weight_dir = text(config, "weight_dir")
            if weight_dir:
                checkpoint_path = str(Path(weight_dir) / "model.ckpt")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is empty")

        result_dir = text(inference, "output_dir") or text(config, "result_dir", "results")

        progress = DltoolProgressCallback(client, args.dltool_task_id, "anomalib predict")
        status(client, args.dltool_task_id, TaskStatus.RUNNING, 0, -1, "开始 anomalib 预测")

        datamodule = build_datamodule(config, "test_params")
        model = build_model(config, "test_params", visualizer=False)
        engine = build_engine(config, "test_params", progress.callback)
        predictions = engine.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
            return_predictions=True,
        )
        save_prediction_score_maps(predictions, result_dir, datamodule.test_image_ids)
        prediction_count = save_normalized_manifest(
            predictions,
            result_dir,
            datamodule.test_image_ids,
            text(config, "model_uuid"),
            text(config, "test_task_uuid"),
            text(config, "method", text(config, "task_type", "test")),
        )

        final_payload = {"phase": "test", "started": True, "phase_progress": 100}
        # The manifest is the canonical prediction protocol.  A model may
        # provide image-level scores without an anomaly_map, so the number of
        # TIFF artifacts is not the number of predictions used by C++.
        final_payload["prediction_count"] = prediction_count
        final_payload["output_dir"] = str(Path(result_dir))
        status(client, args.dltool_task_id, TaskStatus.FINISHED, 100, 0, "预测完成", **final_payload)
        return 0
    except TaskStopRequested:
        return 2
    except Exception:
        report_failure(client, args, "预测")
        return 1
    finally:
        if client is not None:
            client.close()
