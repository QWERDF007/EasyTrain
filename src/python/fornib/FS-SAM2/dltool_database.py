"""SQLite and file-list readers used by the DLTool FS-SAM2 entry points."""

import csv
import copy
import json
import sqlite3
from pathlib import Path, PureWindowsPath
from typing import Any


def _insert_value(target: dict[str, Any], parts: list[str], value: Any) -> None:
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    if parts:
        current[parts[-1]] = value


def load_train_params(path: str | Path) -> dict[str, Any]:
    database_path = Path(path)
    if not database_path.is_file():
        raise FileNotFoundError(f"model database not found: {database_path}")

    params: dict[str, Any] = {}
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT name_en, value FROM train_params ORDER BY name_en").fetchall()
    for name, encoded in rows:
        if not str(name).strip():
            continue
        _insert_value(params, str(name).split("."), json.loads(encoded))
    return params


def load_dataset_selections(path: str | Path) -> dict[str, dict[str, Any]]:
    selections = {split: {"dataset_ids": set(), "label_classes": {}} for split in ("train", "validation", "test")}
    with sqlite3.connect(Path(path)) as connection:
        rows = connection.execute("SELECT type, dataset_id, class_ids FROM datasets ORDER BY type, dataset_id").fetchall()
    for split, dataset_id, encoded_classes in rows:
        if split not in selections:
            continue
        class_ids = json.loads(encoded_classes)
        if not class_ids:
            selections[split]["dataset_ids"].add(int(dataset_id))
        else:
            selections[split]["label_classes"][int(dataset_id)] = {int(value) for value in class_ids}
    return selections


def read_file_list(path: str | Path) -> list[tuple[str, str]]:
    file_list_path = Path(path)
    if not file_list_path.is_file():
        raise FileNotFoundError(f"dataset file list not found: {file_list_path}")
    rows: list[tuple[str, str]] = []
    with file_list_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].strip().lower() == "image_id":
                continue
            if len(row) < 2:
                continue
            image_id = row[0].strip()
            image_path = row[1].strip()
            if image_id and image_path:
                rows.append((image_id, image_path))
    if not rows:
        raise ValueError(f"dataset file list has no usable images: {file_list_path}")
    return rows


def read_label_file(path: str | Path) -> list[dict[str, Any]]:
    label_path = Path(path)
    if not label_path.is_file():
        return []
    with label_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(f"dataset label file is not a list: {label_path}")
    return [item for item in value if isinstance(item, dict)]


def _image_id(value: Any) -> str:
    return str(value).strip()


def _resolve_path(value: Any, base_dir: Path) -> str:
    raw = str(value).strip()
    path = Path(raw)
    # The exporter may store an absolute Windows path.  Keep it absolute even
    # when a helper is inspected from a POSIX environment, where Path alone
    # does not recognize drive-letter paths.
    if path.is_absolute() or PureWindowsPath(raw).is_absolute():
        return raw
    return str(base_dir / path)


def load_split_records(
    dataset_dir: str | Path,
    split: str,
    file_list_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Combine one exported CSV file list with its JSON image records.

    The C++ exporter owns the file-list and annotation format.  The CSV is
    authoritative for the image order and paths; the JSON supplies labels and
    mask paths.  Keeping this merge here makes all FS-SAM2 entry points use the
    same database-era storage protocol.
    """
    root = Path(dataset_dir)
    list_path = (
        Path(file_list_path)
        if file_list_path
        else root / "test.txt"
        if split == "test"
        else root / f"{split}.txt"
    )
    label_path = root / f"{split}_labels.json" if file_list_path is None else None
    rows = read_file_list(list_path)
    labels = {
        _image_id(item.get("id")): item
        for item in read_label_file(label_path)
    } if label_path is not None else {}

    records: list[dict[str, Any]] = []
    for image_id, image_path in rows:
        record = copy.deepcopy(labels.get(_image_id(image_id), {}))
        record["id"] = image_id
        record["path"] = _resolve_path(image_path, list_path.parent)
        raw_labels = record.get("labels", [])
        record["labels"] = raw_labels if isinstance(raw_labels, list) else []
        for label in record["labels"]:
            if not isinstance(label, dict):
                continue
            if "mask_path" in label and str(label["mask_path"]).strip():
                label["mask_path"] = _resolve_path(label["mask_path"], root)
        records.append(record)

    if not records:
        raise ValueError(f"dataset split has no usable images: {split}")
    return records
