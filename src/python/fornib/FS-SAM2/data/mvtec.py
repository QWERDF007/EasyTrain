r""" MVTec-Unseen few-shot defect segmentation dataset """
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import tv_tensors
import PIL.Image as Image
import numpy as np



UNSEEN_CLASSES = {
    'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
    'grid': ['thread', 'bent', 'broken', 'glue', 'metal_contamination'],
    'leather': ['cut', 'fold', 'glue', 'poke', 'color'],
    'tile': ['gray_stroke', 'oil', 'rough', 'crack', 'glue_strip'],
    'wood': ['liquid', 'scratch', 'color', 'combined', 'hole'],
}


class DatasetMvtec(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size, use_original_imgsize, seed=None):
        self.split = 'test' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 5  # using Unseen classes
        self.nclass = 25  # 5 classes x 5 defects each
        self.benchmark = 'mvtec'
        self.shot = shot
        datapath = '/mnt/c/Users/bforni/Desktop/SEA/mmsegmentation/data/'
        self.base_path = Path(datapath) / 'mvtec_anomaly_detection'
        self.transform = transform
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.epoch = -1

        self.class_ids = self.build_class_ids()
        self.img_metadata, self.img_metadata_classwise = self.build_img_metadata()  # using UNSEEN_CLASSES

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
        for shot in range(self.shot):  # resize and apply augmentations to support_masks
            support_imgs[shot] = support_imgs[shot].resize((self.img_size, self.img_size))
            support_masks[shot] = tv_tensors.Mask(F.interpolate(support_masks[shot].unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze())
            support_imgs[shot], support_masks[shot] = self.transform(support_imgs[shot], support_masks[shot])

        batch = {'query_name': query_name.parent.parent.parent.name + '_' + query_name.parent.name + '_' + query_name.stem,
                 'query_img': query_img,
                 'query_mask': query_mask,
                 
                 'support_names': [support.parent.parent.parent.name + '_' + support.parent.name + '_' + support.stem for support in support_names],
                 'support_imgs': torch.stack(support_imgs),  # shape = (shot, 3, H, W)
                 'support_masks': torch.stack(support_masks),  # shape = (shot, H, W)

                 'class_id': torch.tensor(class_sample)}

        return batch
    
    def sample_episode(self, idx):
        rng = np.random.default_rng((self.seed, idx, self.epoch+1))  # fixed randomness for reproducibility

        query_name = self.img_metadata[idx]
        query_img = Image.open(query_name).convert('RGB')
        query_mask = self.read_mask(query_name.parent.parent.parent / 'ground_truth' / query_name.parent.name / (query_name.stem + '_mask.png'))  # change parent to 'GT' and extension to '.png'
        class_sample_name = (query_name.parent.parent.parent.name, query_name.parent.name)
        class_sample = list(self.img_metadata_classwise.keys()).index(class_sample_name)  # get the corresponding class id

        # sample `shot` support images from the same class
        valid_support_names = [x for x in self.img_metadata_classwise[class_sample_name] if x != query_name]
        support_names = rng.choice(valid_support_names, self.shot, replace=False).tolist()  # kshot images
        support_imgs = [Image.open(support_name).convert('RGB') for support_name in support_names]
        support_masks = [self.read_mask(support_name.parent.parent.parent / 'ground_truth' / support_name.parent.name / (support_name.stem + '_mask.png')) for support_name in support_names]

        return query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask == 255] = 1
        return mask
    
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        return class_ids

    def build_img_metadata(self):
        '''get folder name (=class) + sub-folder name (=defect) inside base_path'''

        img_metadata = []
        img_metadata_classwise = {}
        unseen_classes = list(UNSEEN_CLASSES.keys())
        for img_class, img_defects in UNSEEN_CLASSES.items():
            for img_defect in img_defects:
                img_metadata_classwise[(img_class, img_defect)] = []  # initialize dict, order used to index classes
        
        for img_class in self.base_path.iterdir():
            if img_class.name not in UNSEEN_CLASSES:  # or not img_class.is_dir():
                continue
            if self.split == 'trn':
                if unseen_classes.index(img_class.name) == self.fold:
                    continue
            else:  # take only images of same fold class
                if unseen_classes.index(img_class.name) != self.fold:
                    continue

            for img_defect in (img_class / 'test').iterdir():
                if img_defect.name not in UNSEEN_CLASSES[img_class.name]:  # or img_defect.name == 'good':
                    continue
                for img in img_defect.iterdir():
                    img_metadata.append(img)
                    img_metadata_classwise[(img_class.name,img_defect.name)].append(img)
        
        print(f'Total MVTec-Unseen images are: {len(img_metadata)}')

        return img_metadata, img_metadata_classwise
