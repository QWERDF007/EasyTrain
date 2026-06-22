r""" Custom few-shot semantic segmentation dataset """
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import tv_tensors
import PIL.Image as Image
import numpy as np


class DatasetCustom(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size, use_original_imgsize, seed=None):
        self.split = split
        self.shot = shot
        self.benchmark = 'custom'
        self.base_path = datapath
        self.transform = transform
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.epoch = -1

        self.all_classes = sorted([
            d for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ])
        if not self.all_classes:
            raise RuntimeError(f'No class directories found in {self.base_path}')
        self.nclass = len(self.all_classes)

        n_train = max(1, int(len(self.all_classes) * 0.8))
        if split == 'trn':
            self.classes = self.all_classes[:n_train]
        else:  # val or test
            self.classes = self.all_classes[n_train:]

        self.class_ids = list(range(len(self.classes)))
        self.img_metadata = self.build_img_metadata()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample = self.sample_episode(idx)

        query_img = query_img.resize((self.img_size, self.img_size))
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze()
        query_mask = tv_tensors.Mask(query_mask)
        query_img, query_mask = self.transform(query_img, query_mask)

        for shot in range(self.shot):
            support_imgs[shot] = support_imgs[shot].resize((self.img_size, self.img_size))
            support_masks[shot] = tv_tensors.Mask(F.interpolate(support_masks[shot].unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze())
            support_imgs[shot], support_masks[shot] = self.transform(support_imgs[shot], support_masks[shot])

        batch = {'query_name': query_name,
                 'query_img': query_img,
                 'query_mask': query_mask,

                 'support_names': support_names,
                 'support_imgs': torch.stack(support_imgs),
                 'support_masks': torch.stack(support_masks),

                 'class_id': torch.tensor(class_sample)}

        return batch

    def sample_episode(self, idx):
        rng = np.random.default_rng((self.seed, idx, self.epoch + 1))

        query_name, query_class = self.img_metadata[idx]
        class_sample = self.classes.index(query_class)

        query_img = Image.open(self._find_image(query_class, query_name)).convert('RGB')
        query_mask = self.read_mask(os.path.join(self.base_path, query_class, 'masks', query_name + '.png'))

        # sample k support images from the same class
        support_txt = os.path.join(self.base_path, query_class, 'support.txt')
        with open(support_txt, 'r') as f:
            support_names = [line.strip() for line in f if line.strip()]
        support_names = rng.choice(support_names, self.shot, replace=False).tolist()
        support_imgs = [Image.open(self._find_image(query_class, name)).convert('RGB') for name in support_names]
        support_masks = [self.read_mask(os.path.join(self.base_path, query_class, 'masks', name + '.png')) for name in support_names]

        return query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample

    def _find_image(self, class_dir, name):
        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            path = os.path.join(self.base_path, class_dir, 'images', name + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'Image not found: {os.path.join(self.base_path, class_dir, "images", name)}.{{jpg,png}}')

    def read_mask(self, mask_path):
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def build_img_metadata(self):
        img_metadata = []
        for cls in self.classes:
            query_txt = os.path.join(self.base_path, cls, 'query.txt')
            with open(query_txt, 'r') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        img_metadata.append((name, cls))
        return img_metadata
