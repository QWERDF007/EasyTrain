import argparse

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
    integer,
    load_config,
    metrics_text,
    report_failure,
    report_result,
    status,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="DLTool anomalib training entry")
    add_task_arguments(parser)
    args = parser.parse_args()

    client = create_task_client(args)
    try:
        from lightning.pytorch import seed_everything

        config = load_config(args.config)
        seed = integer(group(config, "train_params", "trainer"), "seed", 42)
        seed_everything(seed, workers=True)

        progress = DltoolProgressCallback(client, args.dltool_task_id, "anomalib train")
        status(client, args.dltool_task_id, TaskStatus.RUNNING, 0, -1, "开始 anomalib 训练")

        datamodule = build_datamodule(config, "train_params")
        model = build_model(config, "train_params")
        engine = build_engine(config, "train_params", progress.callback)
        engine.fit(model=model, datamodule=datamodule)
        results = engine.test(model=model, datamodule=datamodule)

        best_model_path = engine.best_model_path or ""
        message = f"训练完成: {best_model_path}" if best_model_path else "训练完成"
        final_payload = {"phase": "test", "started": True, "phase_progress": 100}
        if results:
            report_result(client, args, "验证评估", results)
            final_payload["metrics"] = metrics_text(results)
        status(client, args.dltool_task_id, TaskStatus.FINISHED, 100, 0, message, **final_payload)
        return 0
    except TaskStopRequested:
        return 2
    except Exception:
        report_failure(client, args, "训练")
        return 1
    finally:
        if client is not None:
            client.close()
