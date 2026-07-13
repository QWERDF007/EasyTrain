"""Common exception boundary for task script entry points."""

from __future__ import annotations

import importlib
import sys
import traceback
from argparse import Namespace
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from dltool_task_protocol import TaskStatus  # noqa: E402
from dltool_task_reporting import create_task_client, report_log, report_status  # noqa: E402


ERROR_LOG_PREFIX = "[DLTOOL_ERROR] "


def _task_args() -> Namespace:
    values: dict[str, str | int] = {
        "dltool_task_host": "",
        "dltool_task_port": 0,
        "dltool_task_id": -1,
    }
    option_names = {
        "--dltool_task_host": "dltool_task_host",
        "--dltool_task_port": "dltool_task_port",
        "--dltool_task_id": "dltool_task_id",
    }
    for index, argument in enumerate(sys.argv):
        name = option_names.get(argument)
        if name is None or index + 1 >= len(sys.argv):
            continue
        value = sys.argv[index + 1]
        values[name] = int(value) if name != "dltool_task_host" else value
    return Namespace(**values)


def run_entrypoint(module_name: str, action: str) -> int:
    client = None
    try:
        module = importlib.import_module(module_name)
        return int(module.main())
    except SystemExit as error:
        code = error.code
        return int(code) if isinstance(code, int) else 1
    except BaseException:
        details = traceback.format_exc().strip()
        print(details, file=sys.stderr, flush=True)
        try:
            args = _task_args()
            client = create_task_client(args)
            if client is not None:
                report_log(client, args, ERROR_LOG_PREFIX + details)
                report_status(client, args, TaskStatus.FAILED, -1, -1, f"{action}启动失败，请查看详细日志。")
        finally:
            if client is not None:
                client.close()
        return 1
