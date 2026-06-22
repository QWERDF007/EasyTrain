""" COCO-20i few-shot semantic segmentation dataset """
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import tv_tensors
import PIL.Image as Image
import numpy as np


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size, use_original_imgsize, seed=None):
        self.split = split
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.epoch = -1

        self.class_ids = self.build_class_ids()
        #self.write_fss_list(filter_intersection=True)  # RUN JUST ONCE to generate the splits
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self.split == 'trn':
            return len(self.img_metadata)
        elif self.split == 'val':
            return 1000
        else:  # for testing: use more imgs (see PFENet and HDMNet: 10k images)
            return 1000
            #return 141 # all fold-0 class=32 episodes

    def __getitem__(self, idx):
        query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample = self.sample_episode(idx)

        query_img = query_img.resize((self.img_size, self.img_size))  # use Pillow for resizing (like SAM2)
        if not self.use_original_imgsize:  # resize query_mask
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze()
        query_mask = tv_tensors.Mask(query_mask.float())  # for torchvision transforms
        query_img, query_mask = self.transform(query_img, query_mask)  # apply augmentations
        query_mask = self.binary_mask(query_mask, class_sample)  # remove all other classes

        for shot in range(self.shot):  # resize and apply augmentations to support_masks
            support_imgs[shot] = support_imgs[shot].resize((self.img_size, self.img_size))
            support_masks[shot] = tv_tensors.Mask(F.interpolate(support_masks[shot].unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze())
            support_imgs[shot], support_masks[shot] = self.transform(support_imgs[shot], support_masks[shot])
            support_masks[shot] = self.binary_mask(support_masks[shot], class_sample)

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

        # ignores idx during training and testing, perform uniform sampling over object classes to form an episode (due to the large size of the COCO dataset)
        class_sample = rng.choice(self.class_ids)  # sample 1 class
        query_name = rng.choice(self.img_metadata_classwise[class_sample])  # sample 1 query from that class

        # query_name = self.img_metadata[idx]
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)  # 0: BG, 1-81: COCO FG classes

        # # sample 1 class from the query image
        # label_classes = query_mask.unique().tolist()
        # if 0 in label_classes:  # remove BG class
        #     label_classes.remove(0)
        # label_classes = [c - 1 for c in label_classes if c - 1 in self.class_ids]  # convert to 0-indexed classes and select only classes in fold
        # assert len(label_classes) > 0
        # class_sample = rng.choice(label_classes)

        # sample `shot` support images from the same class
        valid_support_names = [x for x in self.img_metadata_classwise[class_sample] if x != query_name]
        #support_names = [valid_support_names[0]]
        support_names = rng.choice(valid_support_names, self.shot, replace=False).tolist()  # kshot images
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        return query_name, query_img, query_mask, support_names, support_imgs, support_masks, class_sample

    def read_img(self, name):
        """Return RGB image as PIL Image"""
        img_path = os.path.join(self.base_path, 'val2014' if 'val2014' in name else 'train2014', name + '.jpg')
        return Image.open(img_path).convert('RGB')
    
    def read_mask(self, name):
        """Return segmentation mask as tensor from PIL Image"""
        mask_path = os.path.join(self.base_path, 'annotations', 'val2014' if 'val2014' in name else 'train2014', name + '.png')
        return torch.tensor(np.array(Image.open(mask_path)))
    
    def binary_mask(self, mask, class_id):
        """Remove all other classes from the mask to get binary mask"""
        mask[mask != class_id + 1] = 0  # all other classes are set as bg
        mask[mask == class_id + 1] = 1  # set the sampled class as fg
        return mask

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]  # FWB split
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = f'data/splits/lists/coco/fss_list/{split}/data_list_{fold_id}.txt'
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')
            
            fold_n_metadata = list(filter(None, fold_n_metadata))
            #fold_n_metadata = [data.split(' ')[0].split('/')[-1].split('.')[0] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # for training, read image-metadata of "the other" folds
            img_metadata = read_metadata('train', self.fold)
        else:  # for validation/testing, read image-metadata of "current" fold
            img_metadata = read_metadata('val', self.fold)

        print('Total (%s) images are: %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        split = 'train' if self.split == 'trn' else 'val'
        fold_n_subclsdata = f'data/splits/lists/coco/fss_list/{split}/sub_class_file_list_{self.fold}.txt'
            
        with open(fold_n_subclsdata, 'r') as f:
            fold_n_subclsdata = f.read()
            
        sub_class_file_list = eval(fold_n_subclsdata)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            img_metadata_classwise[sub_cls-1] = sub_class_file_list[sub_cls]
            #img_metadata_classwise[sub_cls-1] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        return img_metadata_classwise
    
    def write_fss_list(self, filter_intersection=False):
        # based on code from: https://github.com/chunbolang/BAM/blob/main/util/dataset.py

        import cv2

        nclass_trn = self.nclass // self.nfolds

        for fold in range(4):
            class_ids_val = [1 + fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(1, 1 + self.nclass) if x not in class_ids_val]

            for split in ['train', 'val']:
                sub_list = class_ids_trn if split == 'train' else class_ids_val
                data_list = f'data/splits/lists/coco/{split}.txt'

                image_label_list = []  
                list_read = open(data_list).readlines()
                print("Processing data:", sub_list)
                sub_class_file_list = {}
                for sub_c in sub_list:
                    sub_class_file_list[sub_c] = []

                for l_idx in range(len(list_read)):
                    line = list_read[l_idx]
                    line = line.strip()
                    line_split = line.split(' ')
                    image_name = os.path.join(self.base_path, line_split[0])
                    label_name = os.path.join(self.base_path, line_split[1])
                    item = (image_name, label_name)
                    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
                    label_class = np.unique(label).tolist()

                    if 0 in label_class:
                        label_class.remove(0)
                    if 255 in label_class:
                        label_class.remove(255)

                    new_label_class = []

                    if filter_intersection and split != 'val':  # filter images containing objects of novel categories during meta-training
                        if set(label_class).issubset(set(sub_list)):
                            for c in label_class:
                                if c in sub_list:
                                    tmp_label = np.zeros_like(label)
                                    target_pix = np.where(label == c)
                                    tmp_label[target_pix[0],target_pix[1]] = 1
                                    if tmp_label.sum() >= 2 * 32 * 32:
                                        new_label_class.append(c)
                    else:
                        for c in label_class:
                            if c in sub_list:
                                tmp_label = np.zeros_like(label)
                                target_pix = np.where(label == c)
                                tmp_label[target_pix[0],target_pix[1]] = 1
                                if tmp_label.sum() >= 2 * 32 * 32:
                                    new_label_class.append(c)

                    label_class = new_label_class

                    if len(label_class) > 0:
                        image_label_list.append(item)
                        for c in label_class:
                            if c in sub_list:
                                sub_class_file_list[c].append(item[0].split('/')[-1].split('.')[0])
                                
                print(f"Checking image&label list done for split={split} fold={fold}")

                fold_n_metadata = f'data/splits/lists/coco/fss_list/{split}/data_list_{fold}.txt'
                fold_n_subclsdata = f'data/splits/lists/coco/fss_list/{split}/sub_class_file_list_{fold}.txt'

                with open(fold_n_metadata, 'w') as f:
                    for img, label in image_label_list:
                        f.write(img.split('/')[-1].split('.')[0] + '\n')
                with open(fold_n_subclsdata, 'w') as f:
                    f.write(str(sub_class_file_list))
