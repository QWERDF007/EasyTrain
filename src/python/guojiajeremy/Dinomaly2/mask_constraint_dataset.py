"""Dataset utilities for the Dinomaly2 mask-constraint training in DLTool.

Masks are exported by the C++ side as ``{image_id}.png`` where each annotated
region is rasterized with the value of its label class (the class id). The
project's data management defines each label class as good, anomaly or
unlabeled; the corresponding class values are passed to the python scripts as
``--good_values`` / ``--anomaly_values`` / ``--ignore_values``. A mask may
therefore contain many values (e.g. 1, 2, 3, 244, 255), not only 0/255.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def load_mask_array(mask_path: Path) -> np.ndarray:
    """Load a 2D integer mask from an image or ``.npy`` file."""
    if mask_path.suffix.lower() == ".npy":
        mask = np.asarray(np.load(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
    else:
        image = Image.open(mask_path).convert("L")
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D: {mask_path}; got shape {mask.shape}"
        )
    return np.rint(mask).astype(np.int64, copy=False)


def _unique_values(values: Sequence[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _validate_value_lists(good_values, anomaly_values, ignore_values) -> None:
    good = _unique_values(good_values)
    anomaly = _unique_values(anomaly_values)
    ignore = _unique_values(ignore_values)
    if not good and not anomaly:
        raise ValueError(
            "good_values and anomaly_values are both empty; mask training "
            "needs at least one good or anomaly class value."
        )
    all_values = good + anomaly + ignore
    if 0 in all_values:
        raise ValueError(
            "good/anomaly/ignore values must differ from background value 0."
        )
    if len(set(all_values)) != len(all_values):
        raise ValueError(
            "good_values, anomaly_values and ignore_values must be pairwise "
            f"different; got good={good}, anomaly={anomaly}, ignore={ignore}."
        )
    return good, anomaly, ignore


class DltoolMaskConstraintTrainDataset(Dataset):
    """File-list based mask training dataset for the DLTool exported layout.

    Images come from an exported ``image_id,image_path`` file list. A mask
    ``{image_id}.png`` inside ``masks_dir`` carries class values; pixels
    belonging to good classes are pulled toward the encoder, anomaly classes
    are pushed away, and ignore values are excluded from the losses. Images
    without a mask contribute only the default full-image Dinomaly2 loss.
    """

    def __init__(
        self,
        samples: list[dict],
        image_transform,
        image_size: int,
        crop_size: int,
        masks_dir: str | Path | None = None,
        good_values: Sequence[int] = (1,),
        anomaly_values: Sequence[int] = (255,),
        ignore_values: Sequence[int] = (254,),
        joint_transform=None,
    ) -> None:
        if not samples:
            raise ValueError("DltoolMaskConstraintTrainDataset got no samples")
        self.good_values = _unique_values(good_values)
        self.anomaly_values = _unique_values(anomaly_values)
        self.ignore_values = _unique_values(ignore_values)
        _validate_value_lists(
            self.good_values, self.anomaly_values, self.ignore_values
        )
        self.image_transform = image_transform
        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.mask_crop = transforms.CenterCrop(crop_size)
        self.masks_dir = Path(masks_dir).expanduser() if masks_dir else None
        self.joint_transform = joint_transform

        self.samples = []
        for sample in samples:
            image_path = str(sample["path"])
            mask_path = None
            if self.masks_dir is not None:
                candidate = self.masks_dir / f"{sample['id']}.png"
                if candidate.is_file():
                    mask_path = candidate
            self.samples.append(
                {"image_path": image_path, "mask_path": mask_path}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _validate_mask_values(self, mask: torch.Tensor, mask_path: Path) -> None:
        valid_values = {0, *self.good_values, *self.anomaly_values, *self.ignore_values}
        actual_values = set(np.unique(mask.numpy()).tolist())
        invalid_values = actual_values - valid_values
        if invalid_values:
            raise ValueError(
                f"Invalid values {sorted(invalid_values)} in mask {mask_path}. "
                f"Expected only {sorted(valid_values)}."
            )

    def __getitem__(self, index: int):
        item = self.samples[index]
        image = Image.open(item["image_path"]).convert("RGB")

        mask_path = item["mask_path"]
        mask_image = None
        if mask_path is not None:
            mask_array = load_mask_array(mask_path)
            mask_image = Image.fromarray(mask_array.astype(np.int32), mode="I")

        if self.joint_transform is not None:
            image_tensor, mask_image = self.joint_transform(image, mask_image)
        else:
            image_tensor = self.image_transform(image)

        if mask_image is None:
            mask = torch.zeros(
                (image_tensor.shape[-2], image_tensor.shape[-1]),
                dtype=torch.long,
            )
            has_mask = False
        else:
            if self.joint_transform is None:
                mask_image = self.mask_resize(mask_image)
                mask_image = self.mask_crop(mask_image)
            mask = torch.from_numpy(
                np.asarray(mask_image, dtype=np.int64).copy()
            ).long()
            self._validate_mask_values(mask, mask_path)
            has_mask = True

        return (
            image_tensor,
            mask,
            torch.tensor(has_mask, dtype=torch.bool),
            str(item["image_path"]),
        )


class DltoolEvalDataset(Dataset):
    """MVTec-style evaluation dataset built from an exported file list.

    Each sample returns ``(image, gt, label, image_path)``. A mask
    ``{image_id}.png`` containing any anomaly value marks the image as an
    anomaly; the ground truth is the binary anomaly mask (anomaly pixels = 1).
    Images without an anomaly-valued mask are normal with zero ground truth.
    """

    def __init__(
        self,
        samples: list[dict],
        image_transform,
        gt_transform,
        masks_dir: str | Path | None = None,
        anomaly_values: Sequence[int] = (255,),
    ) -> None:
        if not samples:
            raise ValueError("DltoolEvalDataset got no samples")
        self.anomaly_values = set(_unique_values(anomaly_values))
        self.image_transform = image_transform
        self.gt_transform = gt_transform
        self.masks_dir = Path(masks_dir).expanduser() if masks_dir else None
        self.samples = []
        for sample in samples:
            image_path = str(sample["path"])
            mask_path = None
            if self.masks_dir is not None:
                candidate = self.masks_dir / f"{sample['id']}.png"
                if candidate.is_file():
                    mask_path = candidate
            self.samples.append(
                {"image_path": image_path, "mask_path": mask_path}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _is_anomaly(self, mask_path: Path | None) -> bool:
        if mask_path is None:
            return False
        mask = load_mask_array(mask_path)
        return bool(np.isin(mask, list(self.anomaly_values)).any())

    def __getitem__(self, index: int):
        item = self.samples[index]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.image_transform(image)

        anomaly = self._is_anomaly(item["mask_path"])
        if not anomaly:
            gt = torch.zeros(
                (1, image.size(-2), image.size(-1)), dtype=torch.float32
            )
        else:
            mask = load_mask_array(item["mask_path"])
            binary = np.isin(mask, list(self.anomaly_values)).astype(np.uint8) * 255
            gt = self.gt_transform(Image.fromarray(binary, mode="L"))

        assert image.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return image, gt, int(anomaly), item["image_path"]
