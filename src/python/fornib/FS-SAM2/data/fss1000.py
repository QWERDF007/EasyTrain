r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import tv_tensors
import PIL.Image as Image
import numpy as np


class DatasetFSS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size, use_original_imgsize, seed=None):
        self.split = split  # 'trn', 'val', 'test' (different validation from test split)
        #self.fold = fold  # NO different folds for FSS-1000
        self.nfolds = 1
        self.nclass = 1000
        self.benchmark = 'fss'
        self.shot = shot
        self.base_path = os.path.join(datapath, 'FSS-1000')
        self.transform = transform
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.epoch = -1

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open(f'data/splits/lists/fss/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample = self.sample_episode(idx)

        query_img = query_img.resize((self.img_size, self.img_size))  # use Pillow for resizing (like SAM2)
        if not self.use_original_imgsize:  # resize query_mask
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze()
        query_mask = tv_tensors.Mask(query_mask)  # for torchvision transforms
        query_img, query_mask = self.transform(query_img, query_mask)  # apply augmentations
        query_name = '_'.join(query_name.split('/')[-2:]).split('.')[0]

        for shot in range(self.shot):  # resize and apply augmentations to support_masks
            support_imgs[shot] = support_imgs[shot].resize((self.img_size, self.img_size))
            support_masks[shot] = tv_tensors.Mask(F.interpolate(support_masks[shot].unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze())
            support_imgs[shot], support_masks[shot] = self.transform(support_imgs[shot], support_masks[shot])
        support_names = ['_'.join(name.split('/')[-2:]).split('.')[0] for name in support_names]

        batch = {'query_name': query_name,
                 'query_img': query_img,
                 'query_mask': query_mask,
                 
                 'support_names': support_names,
                 'support_imgs': torch.stack(support_imgs),  # shape = (shot, 3, H, W)
                 'support_masks': torch.stack(support_masks),  # shape = (shot, H, W)
                 
                 'class_id': torch.tensor(class_sample)}

        return batch

    def sample_episode(self, idx):
        rng = np.random.default_rng((self.seed, idx, self.epoch+1))  # fixed randomness for reproducibility

        query_name = self.img_metadata[idx]
        query_id = int(query_name.split('/')[-1].split('.')[0])
        query_img = Image.open(query_name).convert('RGB')
        query_mask = self.read_mask(os.path.join(os.path.dirname(query_name), str(query_id)) + '.png')

        # get the corresponding class
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        # sample `shot` support images from the same class
        valid_support_ids = [x for x in range(1, 11) if x != query_id]
        support_ids = rng.choice(valid_support_ids, self.shot, replace=False).tolist()  # kshot images
        support_names = [os.path.join(os.path.dirname(query_name), str(support_id)) for support_id in support_ids]
        support_imgs = [Image.open(support_name + '.jpg').convert('RGB') for support_name in support_names]
        support_masks = [self.read_mask(support_name + '.png') for support_name in support_names]

        return query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata
