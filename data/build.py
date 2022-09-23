# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from symbol import factor
from torch.utils import data

from . import datasets as D

from .datasets.rwf2000 import RWF2000
from .datasets.rlvs import RLVS
from .transforms import build_transforms
from .collate_batch import collate_fn


def build_dataset(dataset_name, transforms, is_train=True):
    if dataset_name == 'rwf-2000':
        dataset = RWF2000(transforms=transforms, is_train=is_train)
    
    if dataset_name == 'rlvs':
        dataset = RLVS(transforms=transforms, is_train=is_train)

    return dataset



def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
