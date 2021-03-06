from typing import List

import torch
import numpy as np
from firelab import BaseTrainer
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchvision import transforms

from src.models.detection.mask_rcnn import MaskRCNN, maskrcnn_resnet50_fpn
from src.dataloaders.coco import CocoDetection

class MaskRCNNTrainer(BaseTrainer):
    def __init__(self, config):
        super(MaskRCNNTrainer, self).__init__(config)

    def init_models(self):
        # backbone = mobilenet_v2(pretrained=True).features
        # self.model = MaskRCNN(backbone, **self.config.hp.model_config.to_dict())
        self.model = maskrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=91,
            pretrained_backbone=False,
            **self.config.hp.model_config.to_dict())
        self.model = self.model.to(self.device_name)

    def init_dataloaders(self):
        def collate_batch(batch):
            images = [img for img, target in batch]
            targets = [target for img, target in batch]

            return images, targets

        def to_xyxy_format(bbox) -> List:
            # Converts bbox from coco format (x, y, width, height)
            # to "x1, y1, x2, y2" format (pascal voc?)
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

        def torchvision_target_format(target):
            return {
                'image_id': target[0]['image_id'],
                'labels': torch.Tensor([obj['category_id'] for obj in target]).long().to(self.device_name),
                'boxes': torch.Tensor([to_xyxy_format(obj['bbox']) for obj in target]).to(self.device_name),
                'masks': torch.Tensor([obj['mask'] for obj in target]).to(self.device_name)
            }

        train_ds = CocoDetection(f'{self.config.coco_data_dir}/coco_train2014',
                                 f'{self.config.coco_data_dir}/annotations/instances_train2014.json',
                                 transform=transforms.ToTensor(),
                                 target_transform=torchvision_target_format)
        # val_ds = CocoDetection(f'{self.config.coco_data_dir}/coco_val2014',
        #                          f'{self.config.coco_data_dir}/annotations/instances_val2014.json',
        #                          transform=transforms.ToTensor(),
        #                          target_transform=torchvision_target_format)

        self.train_dataloader = DataLoader(train_ds, batch_size=self.config.hp.batch_size, collate_fn=collate_batch)
        # self.val_dataloader = DataLoader(val_ds, batch_size=self.config.hp.batch_size, collate_fn=collate_batch, num_workers=-1)

    def init_optimizers(self):
        # TODO: lr scheduling
        # self.optim = torch.optim.Adam(self.model.parameters())
        self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim.to_dict())
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, 1e-6, 1e-2)

    def train_on_batch(self, batch):
        images, targets = batch
        images = [img.to(self.device_name) for img in images]

        self.model.train()

        with torch.autograd.detect_anomaly():
            losses = self.model(images, targets)
            coefs = [self.config.hp.get(f'loss_coefs.{loss_name}', 1) for loss_name in losses]
            total_loss = sum(c * l for c, l in zip(coefs, losses.values()))

            self.optim.zero_grad()
            total_loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_norm_value)
            self.optim.step()
            self.lr_scheduler.step()

        self.writer.add_scalar(f'Grad_norm', grad_norm, self.num_iters_done)
        for loss_name in losses:
            self.writer.add_scalar(f'Train/{loss_name}', losses[loss_name].item(), self.num_iters_done)
        self.writer.add_scalar(f'Train/total_loss', total_loss.item(), self.num_iters_done)
