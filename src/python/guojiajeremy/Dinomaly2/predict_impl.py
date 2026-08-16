"""DLTool entry for Dinomaly2 prediction (test task).

Predicts anomaly score maps for the exported test split, writes one float32
TIFF per image into the prediction directory and persists image-level scores
into ``task.db.prediction`` so the C++ evaluator can consume them.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image

from dltool_common import (
    DltoolProgressReporter,
    TaskStopRequested,
    add_task_arguments,
    build_mask_constraint_model,
    build_predict_dataset,
    create_task_client,
    group,
    integer,
    load_database_config,
    report_failure,
    select_device,
    text,
)


def _save_prediction_records(task_db: str, records: list[tuple[int, dict]]) -> int:
    if not task_db:
        raise ValueError("task_db is empty")
    with sqlite3.connect(task_db) as connection:
        connection.executemany(
            "INSERT INTO prediction (image_id, data) VALUES (?, ?) "
            "ON CONFLICT(image_id) DO UPDATE SET data=excluded.data",
            [(int(image_id), json.dumps(payload)) for image_id, payload in records],
        )
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool Dinomaly2 prediction entry")
    add_task_arguments(parser)
    args = parser.parse_args()

    client = create_task_client(args)
    reporter = DltoolProgressReporter(client, args.dltool_task_id, "Dinomaly2 predict")
    try:
        from utils import cal_anomaly_maps, get_gaussian_kernel

        config = load_database_config(args, "test_params")
        inference = group(config, "test_params", "inference")
        checkpoint = text(inference, "checkpoint")
        if not checkpoint:
            weight_dir = text(config, "weight_dir")
            if weight_dir:
                checkpoint = str(Path(weight_dir) / "model.pth")
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError(f"checkpoint not found: {checkpoint}")

        network = group(config, "train_params", "network")
        training = group(config, "train_params", "training")
        image_size = integer(network, "image_size", 448)
        crop_size = integer(network, "crop_size", 392)
        batch_size = integer(inference, "batch_size", 8)
        num_workers = integer(inference, "num_workers", 4)
        device = select_device(inference, "device", "cuda:0")

        reporter.start(
            "开始 Dinomaly2 预测",
            phase="test",
            started=True,
            phase_progress=0,
        )

        model, _trainable = build_mask_constraint_model(network, training, device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        dataset = build_predict_dataset(config, image_size, crop_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        result_dir = text(inference, "output_dir") or text(config, "result_dir", "results")
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        records: list[tuple[int, dict]] = []
        total = len(dataset)
        done = 0
        with torch.no_grad():
            for image, image_ids, image_paths in dataloader:
                reporter.check_stop()
                image = image.to(device)
                encoder_features, decoder_features = model(image)
                anomaly_map, _ = cal_anomaly_maps(
                    encoder_features,
                    decoder_features,
                    image.shape[-1],
                )
                anomaly_map = gaussian_kernel(anomaly_map)

                for sample_index in range(anomaly_map.shape[0]):
                    score_map = anomaly_map[sample_index]
                    image_id = image_ids[sample_index]
                    image_path = image_paths[sample_index]

                    original = Image.open(image_path).convert("RGB")
                    original_width, original_height = original.size

                    score_map = F.interpolate(
                        score_map.unsqueeze(0),
                        size=(original_height, original_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0].float().cpu().numpy()

                    output_path = Path(result_dir) / f"{image_id}.tiff"
                    tifffile.imwrite(output_path, np.asarray(score_map, dtype=np.float32))

                    flat = np.sort(np.asarray(score_map, dtype=np.float64).ravel())[::-1]
                    top_count = max(1, int(flat.size * 0.01))
                    image_score = float(flat[:top_count].mean())
                    records.append((int(image_id), {"image_score": image_score}))
                done += anomaly_map.shape[0]
                reporter.report(
                    done,
                    total,
                    f"预测中 [{done}/{total}]",
                    phase="test",
                    started=True,
                    phase_progress=int(100 * done / total),
                )

        prediction_count = _save_prediction_records(args.task_db, records)
        reporter.status(
            f"预测完成，共 {prediction_count} 幅图像",
            phase="test",
            started=True,
            phase_progress=100,
        )
        reporter.finish(
            "预测完成",
            phase="test",
            started=True,
            phase_progress=100,
            prediction_count=prediction_count,
            output_dir=str(Path(result_dir)),
        )
        return 0
    except TaskStopRequested:
        return 2
    except Exception:
        report_failure(client, args, "预测")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())


