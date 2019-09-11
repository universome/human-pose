from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from firelab.config import Config

from .coco import CocoDetection
from .transforms import construct_transforms


def construct_dataloader(config:Config, is_train:bool, is_distributed:bool, shuffle=True):
    if is_train:
        images_dir = config.data.images_dir.train
        annotations_path = config.data.annotations.train
    else:
        images_dir = config.data.images_dir.val
        annotations_path = config.data.annotations.val

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
