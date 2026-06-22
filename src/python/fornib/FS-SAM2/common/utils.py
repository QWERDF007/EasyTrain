r""" Helper functions """
import os
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import numpy as np
import torch.distributed as dist

def fix_randseed(seed):
    """ Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.benchmark = False  # slower
        #torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()



###



def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_metric(metric, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return metric
    with torch.no_grad():
        if not metric.is_contiguous():
            metric = metric.contiguous()
        # dist.barrier()
        dist.all_reduce(metric)
        if average:
            metric /= world_size
    return metric

