import os
from typing import List

import torch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from firelab.config import Config

from .coco import CocoDetection
from .transforms import construct_transforms


def construct_dataloader(config:Config, is_train:bool, is_distributed:bool, shuffle=True):
    # TODO: move paths into config instead of hardcoding them here
    if is_train:
        images_dir = f'{config.data.coco_dir}/train2014'
        annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_train.json'
        # annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_small.json'
    else:
        images_dir = f'{config.data.coco_dir}/val2014'
        annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_minival.json'
        # images_dir = f'{config.data.coco_dir}/train2014'
        # annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_tiny.json'

    dataset = CocoDetection(images_dir, annotations_path, transforms=construct_transforms(is_train))
    sampler = construct_sampler(dataset, is_distributed, shuffle)
    batch_sampler = BatchSampler(sampler, config.hp.batch_size, drop_last=False) # TODO: aspect ratio grouping
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_batch, num_workers=0)

    return dataloader


def construct_sampler(dataset, distributed, shuffle):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        return RandomSampler(dataset)
    else:
        return SequentialSampler(dataset)


def collate_batch(batch):
    images = [img for img, target in batch]
    targets = [target for img, target in batch]

    return images, targets
