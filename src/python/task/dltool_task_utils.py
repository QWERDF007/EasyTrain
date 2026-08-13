"""Shared value helpers for DLTool python runtimes (ultralytics/anomalib/Dinomaly2/FS-SAM2).

各框架 dltool_common.py 的重复辅助函数集中于此，避免功能相同、命名不同的
重复实现。该模块位于 <runtime>/python/task/，随 PYTHONPATH 统一注入。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def format_hms(seconds: Any) -> str:
    """将秒数格式化为 HH:MM:SS；非法或负数返回 "-"。"""
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if value < 0:
        return "-"
    total = max(0, int(round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def metric_value(value: Any) -> Any:
    """将 tensor/dict/list 递归归一为可序列化标量。"""
    if isinstance(value, dict):
        return {str(key): metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_value(item) for item in value]
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def format_number(value: Any, digits: int = 6) -> str:
    """格式化数值/tensor 为文本，空值返回 "-"。"""
    normalized = metric_value(value)
    if normalized is None:
        return "-"
    if isinstance(normalized, float):
        return f"{normalized:.{digits}f}"
    return str(normalized)


def estimate_eta(elapsed_seconds: float, done: int, total: int) -> int:
    """按已完成量线性估算剩余秒数；无法估算返回 -1。"""
    total = max(1, int(total))
    done = max(0, min(total, int(done)))
    if done <= 0 or elapsed_seconds <= 0:
        return -1
    return int(round(elapsed_seconds * (total - done) / done))


def is_character_sequence(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(len(str(item)) == 1 for item in value)


def scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if is_character_sequence(value):
        return "".join(str(item) for item in value)
    return value


def text(values: dict[str, Any], name: str, default: str = "") -> str:
    value = scalar(values.get(name, default), default)
    return default if value is None else str(value).strip()


def optional_text(values: dict[str, Any], name: str) -> str | None:
    value = text(values, name)
    return value or None


def integer(values: dict[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(scalar(values.get(name, default), str(default)))
    except (TypeError, ValueError):
        return default


def floating(values: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(scalar(values.get(name, default), str(default)))
    except (TypeError, ValueError):
        return default


def boolean(values: dict[str, Any], name: str, default: bool = False) -> bool:
    value = scalar(values.get(name, default), default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def string_list(values: dict[str, Any], name: str, default: list[str] | None = None) -> list[str]:
    fallback = list(default or [])
    if not isinstance(values, dict) or name not in values or values.get(name) is None:
        return fallback

    value = scalar(values.get(name))
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        return result or fallback

    parts = [part.strip() for part in str(value).replace(";", ",").split(",")]
    return [part for part in parts if part] or fallback


def square_size(values: dict[str, Any], name: str, default: int) -> tuple[int, int]:
    size = integer(values, name, default)
    return size, size


def parse_int_list(value: Any, default: list[int] | None = None) -> list[int]:
    """解析逗号/空格分隔的整数列表（如 "1,3,244"）。"""
    result: list[int] = []
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
    else:
        for part in str(value).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.append(int(part))
            except (TypeError, ValueError):
                continue
    return result or list(default or [])


def batch_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            try:
                total += max(0, int(item))
            except (TypeError, ValueError, OverflowError):
                continue
        return total
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def dataloader_batch_count(value: Any) -> int:
    try:
        loader = value() if callable(value) else value
        return batch_count(len(loader))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0


def should_stop(client: Any, task_id: int) -> bool:
    return client is not None and client.should_stop(task_id)


def select_device(values: dict[str, Any], name: str = "device", default: str = "cuda:0") -> str:
    """将设备参数解析为 torch 设备字符串。"""
    import torch

    selected = text(values, name, default).strip().lower()
    if torch.cuda.is_available():
        if selected.startswith(("cuda:", "gpu:")):
            try:
                index = int(selected.split(":", 1)[1])
                if 0 <= index < torch.cuda.device_count():
                    return f"cuda:{index}"
            except (TypeError, ValueError):
                pass
            return "cuda:0"
        if selected in {"cuda", "gpu"}:
            return "cuda:0"
    return "cpu"


def load_params_table(database_path: str | Path, table: str) -> dict[str, Any]:
    """从模型/任务数据库读取参数表（group, name_en, value, type），还原为嵌套字典。"""
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    if table not in {"train_params", "test_params"}:
        raise ValueError(f"unsupported parameter table: {table}")
    result: dict[str, Any] = {}
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            f"SELECT `group`, name_en, value, type FROM {table} ORDER BY `group`, name_en"
        ).fetchall()
    for group, name, text, type_name in rows:
        value: Any = text
        if type_name in {"bool", "boolean"}:
            value = str(text).strip().lower() in {"true", "1", "yes", "on"}
        elif type_name in {"int", "integer"}:
            try:
                value = int(str(text).strip())
            except (TypeError, ValueError):
                value = text
        elif type_name in {"double", "float", "real"}:
            try:
                value = float(str(text).strip())
            except (TypeError, ValueError):
                value = text
        result.setdefault(str(group), {})[str(name)] = value
    return result
