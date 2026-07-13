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
    load_config,
    report_failure,
    report_result,
    status,
    text,
)


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
            raise ValueError("checkpoint_path is empty")

        progress = DltoolProgressCallback(client, args.dltool_task_id, "anomalib predict")
        status(client, args.dltool_task_id, TaskStatus.RUNNING, 0, -1, "开始 anomalib 测试")

        datamodule = build_datamodule(config, "test_params")
        model = build_model(config, "test_params")
        engine = build_engine(config, "test_params", progress.callback)
        results = engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

        if results:
            report_result(client, args, "测试评估", results)
        status(client, args.dltool_task_id, TaskStatus.FINISHED, 100, 0, "测试完成")
        return 0
    except TaskStopRequested:
        return 2
    except Exception:
        report_failure(client, args, "测试")
        return 1
    finally:
        if client is not None:
            client.close()
