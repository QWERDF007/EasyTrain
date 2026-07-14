"""Shared task result and failure reporting for external model scripts."""

from __future__ import annotations

import json
import sys
import traceback
from argparse import Namespace
from typing import Any

from dltool_task_protocol import TaskClient, TaskStatus


ERROR_LOG_PREFIX = "[DLTOOL_ERROR] "


class TaskStopRequested(Exception):
    """Raised when DLTool asks an external task to stop."""


def task_id(value: Namespace | int) -> int:
    return int(getattr(value, "dltool_task_id", value))


def create_task_client(args: Namespace) -> TaskClient | None:
    if not args.dltool_task_host or args.dltool_task_port <= 0 or args.dltool_task_id < 0:
        return None
    return TaskClient(args.dltool_task_host, args.dltool_task_port)


def report_status(client: TaskClient | None, task: Namespace | int, status: TaskStatus,
                  progress: int, eta_seconds: int, message: str = "", **payload: Any) -> None:
    if client is not None:
        client.status(task_id(task), status, progress, eta_seconds, message, **payload)


def report_progress(client: TaskClient | None, task: Namespace | int, progress: int,
                    eta_seconds: int, message: str = "", **payload: Any) -> None:
    if client is not None:
        client.progress(task_id(task), progress, eta_seconds, message, **payload)


def report_log(client: TaskClient | None, task: Namespace | int, message: str) -> None:
    if client is not None and message:
        client.log(task_id(task), message)


def report_result(client: TaskClient | None, task: Namespace | int, label: str, result: Any) -> None:
    """Publish structured task output through the common UI-log channel."""
    report_log(client, task, f"{label}: " + json.dumps(result, ensure_ascii=False, default=str))


def report_failure(client: TaskClient | None, task: Namespace | int, action: str) -> None:
    """Send details to the UI log and keep the InfoBar failure text concise."""
    details = traceback.format_exc().strip()
    if details and details != "NoneType: None":
        print(details, file=sys.stderr, flush=True)
        report_log(client, task, ERROR_LOG_PREFIX + details)
    report_status(client, task, TaskStatus.FAILED, -1, -1, f"{action}失败，请查看日志。")
