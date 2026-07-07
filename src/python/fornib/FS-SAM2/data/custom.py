r""" Custom few-shot semantic segmentation dataset """
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import tv_tensors
import PIL.Image as Image
import numpy as np
import yaml


class DatasetCustom(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size, use_original_imgsize, seed=None):
        self.split = split
        self.shot = shot
        self.benchmark = 'custom'
        self.manifest_path = datapath
        self.transform = transform
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.epoch = -1

        self.entries_by_class = self._load_entries(datapath)
        self.all_classes = sorted(self.entries_by_class.keys())
        if not self.all_classes:
            raise RuntimeError(f'No labeled classes found in manifest: {datapath}')
        self.nclass = len(self.all_classes)
        self.classes = self.all_classes[:]
        self.class_ids = list(range(len(self.classes)))
        self.img_metadata = self.build_img_metadata()
        if not self.img_metadata:
            raise RuntimeError(f'No usable images found in manifest: {datapath}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_id, query_name, query_img, query_mask, support_ids, support_names, support_imgs, support_masks, class_sample = self.sample_episode(idx)

        query_img = query_img.resize((self.img_size, self.img_size))
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze()
        query_mask = tv_tensors.Mask(query_mask)
        query_img, query_mask = self.transform(query_img, query_mask)

        for shot in range(self.shot):
            support_imgs[shot] = support_imgs[shot].resize((self.img_size, self.img_size))
            support_masks[shot] = tv_tensors.Mask(F.interpolate(support_masks[shot].unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze())
            support_imgs[shot], support_masks[shot] = self.transform(support_imgs[shot], support_masks[shot])

        return {
            'query_id': query_id,
            'query_name': query_name,
            'query_img': query_img,
            'query_mask': query_mask,
            'support_ids': support_ids,
            'support_names': support_names,
            'support_imgs': torch.stack(support_imgs),
            'support_masks': torch.stack(support_masks),
            'class_id': torch.tensor(class_sample),
        }

    def sample_episode(self, idx):
        rng = np.random.default_rng((self.seed, idx, self.epoch + 1))
        query_entry = self.img_metadata[idx]
        class_name = query_entry['class_name']
        class_sample = self.classes.index(class_name)
        candidates = [entry for entry in self.entries_by_class[class_name] if entry['id'] != query_entry['id']]
        if len(candidates) < self.shot:
            candidates = self.entries_by_class[class_name]
        if len(candidates) < self.shot:
            raise RuntimeError(f'Class {class_name} needs at least {self.shot} support samples')

        chosen = rng.choice(candidates, self.shot, replace=False).tolist()
        query_img, query_mask = self._load_image_and_mask(query_entry)
        support_imgs, support_masks = [], []
        support_ids, support_names = [], []
        for entry in chosen:
            support_img, support_mask = self._load_image_and_mask(entry)
            support_imgs.append(support_img)
            support_masks.append(support_mask)
            support_ids.append(str(entry['id']))
            support_names.append(entry['name'])

        return (
            str(query_entry['id']),
            query_entry['name'],
            query_img,
            query_mask,
            support_ids,
            support_names,
            support_imgs,
            support_masks,
            class_sample,
        )

    def build_img_metadata(self):
        metadata = []
        for class_name in self.classes:
            entries = self.entries_by_class[class_name]
            if len(entries) <= self.shot:
                continue
            metadata.extend(entries)
        return metadata

    def _load_entries(self, manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = yaml.safe_load(handle) or {}
        images = manifest.get('images', [])
        entries_by_class = {}
        for image in images:
            image_path = str(image.get('path', '')).strip()
            if not image_path:
                continue
            masks_by_class = {}
            for label in image.get('labels', []) or []:
                class_id = int(label.get('label_class_id', -1))
                class_name = str(label.get('label_class_name') or class_id)
                mask_path = str(label.get('mask_path', '')).strip()
                if class_id < 0:
                    continue
                if not mask_path:
                    continue
                masks_by_class.setdefault((class_id, class_name), []).append(mask_path)
            for (class_id, class_name), mask_paths in masks_by_class.items():
                entry = {
                    'id': int(image.get('id', -1)),
                    'name': os.path.splitext(os.path.basename(image_path))[0],
                    'path': image_path,
                    'class_id': class_id,
                    'class_name': class_name,
                    'mask_paths': mask_paths,
                    'width': int(image.get('width', 0) or 0),
                    'height': int(image.get('height', 0) or 0),
                }
                entries_by_class.setdefault(class_name, []).append(entry)
        return entries_by_class

    def _load_image_and_mask(self, entry):
        image = Image.open(entry['path']).convert('RGB')
        mask = self._mask_from_entry(entry, image.size)
        return image, mask

    def _mask_from_entry(self, entry, image_size):
        merged = None
        for mask_path in entry.get('mask_paths', []) or []:
            mask = Image.open(mask_path).convert('L')
            if mask.size != image_size:
                mask = mask.resize(image_size, Image.NEAREST)
            values = (np.array(mask, dtype=np.uint8) >= 128).astype(np.uint8)
            merged = values if merged is None else np.maximum(merged, values)
        if merged is None:
            raise FileNotFoundError(f"No mask_path entries for image: {entry['path']}")
        return torch.tensor(merged, dtype=torch.uint8)
