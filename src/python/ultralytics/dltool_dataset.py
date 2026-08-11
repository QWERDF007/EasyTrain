"""Custom dataset reader for DLTool's image_id-prefixed YOLO files."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS


class DlToolYOLODataset(YOLODataset):
    """Read DLTool CSV split files and labels stored beside the images.

    Each label row is ``image_id class_id cx cy width height`` for detection or
    ``image_id class_id x1 y1 x2 y2 ...`` for instance segmentation.
    Coordinates are normalized to the source image dimensions.
    """

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        image_files: list[str] = []
        image_ids: list[str] = []
        values = img_path if isinstance(img_path, list) else [img_path]
        for value in values:
            path = Path(value)
            if not path.is_file():
                new_files = super().get_img_files(value)
                image_files.extend(new_files)
                image_ids.extend(Path(item).stem for item in new_files)
                continue
            with path.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.reader(stream):
                    if not row or not any(item.strip() for item in row):
                        continue
                    if row[0].strip().lower() == "image_id":
                        continue
                    if len(row) >= 2 and row[0].strip().lstrip("-").isdigit():
                        image_path = Path(row[1].strip())
                        if not image_path.is_absolute():
                            image_path = path.parent / image_path
                        image_ids.append(row[0].strip())
                        image_files.append(str(image_path))
                    else:
                        image_path = Path(row[0].strip())
                        if not image_path.is_absolute():
                            image_path = path.parent / image_path
                        image_files.append(str(image_path))
                        image_ids.append(image_path.stem)
        valid_entries = [
            (str(Path(item)), image_id)
            for item, image_id in zip(image_files, image_ids)
            if Path(item).suffix.lower().lstrip(".") in IMG_FORMATS
        ]
        image_files = [item for item, _ in valid_entries]
        image_ids = [image_id for _, image_id in valid_entries]
        if not image_files:
            raise FileNotFoundError(f"No images found in {img_path}")
        self._dltool_image_ids = image_ids
        self._dltool_label_root = next(
            (Path(value).parent for value in values if Path(value).is_file()), Path(image_files[0]).parent
        )
        return image_files

    def get_label_files(self) -> list[str]:
        root = getattr(self, "_dltool_label_root", Path(self.im_files[0]).parent)
        self.label_files = [str(root / f"{image_id}.txt") for image_id in self._dltool_image_ids]
        return self.label_files

    def get_labels(self) -> list[dict]:
        labels: list[dict] = []
        self.get_label_files()
        for image_path, label_path, image_id in zip(self.im_files, self.label_files, self._dltool_image_ids):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image Not Found {image_path}")
            height, width = image.shape[:2]
            classes: list[list[float]] = []
            boxes: list[list[float]] = []
            segments: list[np.ndarray] = []
            path = Path(label_path)
            if path.is_file():
                for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
                    fields = raw_line.strip().split()
                    if len(fields) < 6 or fields[0] != image_id:
                        continue
                    class_id = float(fields[1])
                    coordinates = [float(value) for value in fields[2:]]
                    if self.use_segments and len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                        points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
                        segments.append(points)
                        x_min, y_min = points.min(axis=0)
                        x_max, y_max = points.max(axis=0)
                        boxes.append(
                            [
                                (x_min + x_max) / 2.0,
                                (y_min + y_max) / 2.0,
                                x_max - x_min,
                                y_max - y_min,
                            ]
                        )
                        classes.append([class_id])
                    elif len(coordinates) == 4:
                        boxes.append(coordinates)
                        classes.append([class_id])

            labels.append(
                {
                    "im_file": image_path,
                    "shape": (height, width),
                    "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
                    "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "segments": segments,
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if not labels:
            raise RuntimeError(f"No images found in {self.img_path}")
        return labels
