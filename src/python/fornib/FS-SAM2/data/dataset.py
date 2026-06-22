r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss1000 import DatasetFSS
from data.custom import DatasetCustom
# from data.lvis import DatasetLVIS
# from data.mvtec import DatasetMvtec
from common import utils


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'custom': DatasetCustom,
        }

        cls.img_mean = (0.485, 0.456, 0.406)
        cls.img_std = (0.229, 0.224, 0.225)
        cls.img_size = img_size
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = v2.Compose([v2.ToImage(),  # convert PIL image to tensor (more efficient if done first)
                                    #v2.Resize(size=(img_size, img_size)),  # done inside dataloader
                                    v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale to [0, 1]
                                    v2.Normalize(cls.img_mean, cls.img_std)])
        
        cls.transform_train = v2.Compose([v2.ToImage(),
                                          v2.RandomRotation(degrees=(-10, 10)),
                                          v2.GaussianBlur(kernel_size=(5, 5)),
                                          v2.RandomHorizontalFlip(),
                                          #v2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
                                          v2.ToDtype(torch.float32, scale=True),
                                          v2.Normalize(cls.img_mean, cls.img_std)])
        # other possible augmentations: affine (deg: 25, shear: 20), colorjitter (b: 0.1, c: 0.03, s: 0.03, h: null), grayscale (0.05)

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, seed=None):
        shuffle = split == 'trn'  # shuffle only during training for diverse episode combinations
        #nworker = nworker if split == 'trn' else 0
        transform = cls.transform_train if split == 'trn' else cls.transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot, img_size=cls.img_size, use_original_imgsize=cls.use_original_imgsize, seed=seed)  #, lipstick_type=lipstick_type
        if utils.is_dist_avail_and_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # sampler is taking care of shuffling
        else:
            sampler = None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, pin_memory=True, num_workers=nworker, sampler=sampler)

        return dataloader


