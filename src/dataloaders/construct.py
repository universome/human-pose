import os
from typing import List

import torch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from firelab.config import Config

from .coco import CocoDetection

def construct_dataloader(config:Config, is_train:bool, is_distributed:bool, shuffle=True):
    # TODO: move paths into config instead of hardcoding them here
    if is_train:
        images_dir = f'{config.data.coco_dir}/train2014'
        annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_train.json'
        # images_dir = f'{config.data.coco_dir}/train2014'
        # annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_small.json'
    else:
        images_dir = f'{config.data.coco_dir}/val2014'
        annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_valminusminival.json'
        # images_dir = f'{config.data.coco_dir}/train2014',
        # annotations_path = f'{config.data.annotations_dir}/densepose_coco_2014_tiny.json'

    dataset = CocoDetection(images_dir, annotations_path, transform=transforms.ToTensor(),
                       target_transform=torchvision_target_format)
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


def xywh_to_xyxy(bbox) -> List:
    # Converts bbox from coco format (x, y, width, height)
    # to "x1, y1, x2, y2" format (pascal voc?)
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def torchvision_target_format(target):
    return {
        'image_id': target[0]['image_id'],
        'labels': torch.Tensor([obj['category_id'] for obj in target]).long(),
        'boxes': torch.Tensor([xywh_to_xyxy(obj['bbox']) for obj in target]),
        'masks': torch.Tensor([obj['mask'] for obj in target]),
        'coco_anns': target,
    }
