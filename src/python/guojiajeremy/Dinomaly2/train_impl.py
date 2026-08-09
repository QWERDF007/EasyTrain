"""DLTool entry for Dinomaly2 mask-constraint training."""

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

from dltool_common import (
    DltoolProgressReporter,
    TaskStopRequested,
    add_task_arguments,
    boolean,
    build_mask_constraint_model,
    build_train_dataset,
    create_task_client,
    evaluate_validation,
    floating,
    group,
    integer,
    load_database_config,
    mask_value_lists,
    report_failure,
    select_device,
    text,
)


def _sec2hms(s):
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool Dinomaly2 mask training entry")
    add_task_arguments(parser)
    args = parser.parse_args()

    client = create_task_client(args)
    reporter = DltoolProgressReporter(client, args.dltool_task_id, "Dinomaly2 mask train")
    try:
        import numpy as np
        import torch
        import torch.nn as nn

        from dinomaly_2D import setup_seed

        config = load_database_config(args, "train_params")
        training = group(config, "train_params", "training")
        network = group(config, "train_params", "network")
        mask_values = group(config, "train_params", "mask")
        good_values, anomaly_values, ignore_values = mask_value_lists(args, config)
        reporter.log(
            f"mask values: good={good_values}, anomaly={anomaly_values}, ignore={ignore_values}"
        )

        seed = integer(training, "seed", 1)
        setup_seed(seed)
        image_size = integer(network, "image_size", 448)
        crop_size = integer(network, "crop_size", 392)
        max_iters = integer(training, "max_iters", 40000)
        batch_size = integer(training, "batch_size", 8)
        num_workers = integer(training, "num_workers", 4)
        device = text(training, "device", "cuda:0")
        device = select_device(training, "device", device)
        reporter.log(
            f"device={device}, image_size={image_size}, crop_size={crop_size}, "
            f"max_iters={max_iters}, batch_size={batch_size}, num_workers={num_workers}"
        )

        train_data = build_train_dataset(
            config,
            image_size,
            crop_size,
            good_values=good_values,
            anomaly_values=anomaly_values,
            ignore_values=ignore_values,
        )
        if len(train_data) < batch_size:
            raise ValueError(
                f"训练图像数量 {len(train_data)} 小于批量大小 {batch_size}，请减小批量大小或增加训练数据。"
            )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )

        model, trainable = build_mask_constraint_model(network, training, device)
        bottleneck = model.bottleneck
        decoder = model.decoder

        from optimizers import StableAdamW
        from utils import WarmupCosineScheduler

        optimizer = StableAdamW(
            [
                {"params": bottleneck[0].parameters(), "lr": 2e-4},
                {"params": bottleneck[1].parameters()},
                {"params": decoder.parameters()},
            ],
            lr=2e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=False,
            eps=1e-10,
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            final_ratio=floating(training, "lr_decay_ratio", 1.0),
            total_epochs=max_iters,
            warmup_epochs=100,
        )

        weight_dir = text(config, "weight_dir")
        if weight_dir:
            Path(weight_dir).mkdir(parents=True, exist_ok=True)
        writer = None
        log_dir = text(config, "log_dir")
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            except Exception as exc:
                reporter.log(f"tensorboard 初始化失败，跳过: {exc}")

        from mask_constraint_losses import calculate_mask_constraint_losses

        reporter.start(
            "开始 Dinomaly2 mask 训练",
            phase="train",
            started=True,
            phase_progress=0,
        )
        total_epochs = int(math.ceil(max_iters / len(train_loader)))
        iteration = 0
        start_time = time.time()
        loss_window = []
        good_window = []
        anomaly_window = []
        for epoch in range(max(1, total_epochs)):
            model.train()
            for image, masks, has_mask, _paths in train_loader:
                if iteration >= max_iters:
                    break
                iteration += 1
                reporter.check_stop()

                image = image.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                has_mask = has_mask.to(device, non_blocking=True)
                encoder_features, decoder_features = model(image)

                p_final = floating(network, "ll_ratio", 0.9)
                p = min(p_final * iteration / 1000.0, p_final)
                loss_dinomaly, loss_good, loss_anomaly = calculate_mask_constraint_losses(
                    encoder_features,
                    decoder_features,
                    masks,
                    has_mask,
                    good_values=good_values,
                    anomaly_values=anomaly_values,
                    use_loose_loss=boolean(network, "ll", True),
                    p=p,
                    factor=floating(network, "ll_factor", 0.1),
                    ignore_values=ignore_values,
                )
                good_term = floating(mask_values, "lambda_good", 0.5) * loss_good
                anomaly_term = -floating(mask_values, "lambda_anomaly", 0.5) * loss_anomaly
                total_loss = loss_dinomaly + good_term + anomaly_term

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()

                loss_window.append(float(loss_dinomaly.detach().cpu()))
                good_window.append(float(loss_good.detach().cpu()))
                anomaly_window.append(float(loss_anomaly.detach().cpu()))

                if iteration % 100 == 0 or iteration == max_iters:
                    elapsed = time.time() - start_time
                    eta = elapsed * (max_iters - iteration) / max(iteration, 1)
                    mean_loss = float(np.mean(loss_window))
                    mean_good = float(np.mean(good_window))
                    mean_anomaly = float(np.mean(anomaly_window))
                    mean_total = (
                        mean_loss
                        + floating(mask_values, "lambda_good", 0.5) * mean_good
                        - floating(mask_values, "lambda_anomaly", 0.5) * mean_anomaly
                    )
                    message = (
                        f"iter [{iteration}/{max_iters}], loss={mean_total:.4f}, "
                        f"dinomaly={mean_loss:.4f}, good={mean_good:.4f}, "
                        f"anomaly={mean_anomaly:.4f}, elapsed={_sec2hms(elapsed)}, "
                        f"ETA={_sec2hms(eta)}"
                    )
                    reporter.report(
                        iteration,
                        max_iters,
                        message,
                        phase="train",
                        started=True,
                        phase_progress=int(100 * iteration / max_iters),
                        epoch=f"{epoch + 1} / {total_epochs}",
                        iter=f"{iteration} / {max_iters}",
                        lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                        loss=f"{mean_total:.4f}",
                        elapsed=_sec2hms(elapsed),
                        eta=_sec2hms(eta),
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", mean_loss, iteration)
                        writer.add_scalar("train/loss_good", mean_good, iteration)
                        writer.add_scalar("train/loss_anomaly", mean_anomaly, iteration)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
                    loss_window.clear()
                    good_window.clear()
                    anomaly_window.clear()

        if writer is not None:
            writer.close()

        model_path = os.path.join(weight_dir, "model.pth") if weight_dir else "model.pth"
        torch.save(model.state_dict(), model_path)
        reporter.log(f"save to {model_path}")

        validation_metrics = evaluate_validation(
            model,
            config,
            device,
            batch_size,
            image_size,
            crop_size,
            client,
            args.dltool_task_id,
            anomaly_values=anomaly_values,
        )
        final_payload = {"phase": "train", "started": True, "phase_progress": 100}
        if validation_metrics:
            final_payload["metrics"] = "\n".join(
                f"{key}: {value}" for key, value in validation_metrics.items()
            )
        reporter.finish("训练完成", **final_payload)
        return 0
    except TaskStopRequested:
        return 2
    except Exception:
        report_failure(client, args, "训练")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
