from typing import List

import torch
import numpy as np
from firelab import BaseTrainer
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models import mobilenet_v2
# from torchvision.datasets import CocoDetection

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
            pretrained_backbone=True,
            **self.config.hp.model_config.to_dict())
        self.model = self.model.to(self.config.firelab.device_name)
        self.model.load_state_dict(torch.load('/home/skorokhodov/human-pose/experiments/mask-rcnn-00077/checkpoints/model-6000.pt'))

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
                'labels': torch.Tensor([obj['category_id'] for obj in target]).long().to(self.config.firelab.device_name),
                'boxes': torch.Tensor([to_xyxy_format(obj['bbox']) for obj in target]).to(self.config.firelab.device_name),
                'masks': torch.Tensor([obj['mask'] for obj in target]).to(self.config.firelab.device_name)
            }

        train_ds = CocoDetection(f'{self.config.coco_data_dir}/coco_train2014',
                                 f'{self.config.coco_data_dir}/annotations/instances_train2014.json',
                                 transform=transforms.ToTensor(),
                                 target_transform=torchvision_target_format)
        val_ds = CocoDetection(f'{self.config.coco_data_dir}/coco_val2014',
                                 f'{self.config.coco_data_dir}/annotations/instances_val2014.json',
                                 transform=transforms.ToTensor(),
                                 target_transform=torchvision_target_format)

        self.train_dataloader = DataLoader(train_ds, batch_size=self.config.hp.batch_size, collate_fn=collate_batch)
        self.val_dataloader = DataLoader(val_ds, batch_size=self.config.hp.batch_size, collate_fn=collate_batch)

    def init_optimizers(self):
        # TODO: lr scheduling
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim.to_dict())
        # self.optim.load_state_dict(torch.load('/home/skorokhodov/human-pose/experiments/mask-rcnn-00077/checkpoints/optim-6000.pt'))

    def train_on_batch(self, batch):
        images, targets = batch
        images = [img.to(self.config.firelab.device_name) for img in images]

        self.model.train()
        self.optim.zero_grad()

        with torch.autograd.detect_anomaly():
            losses = self.model(images, targets)

            for loss_name in losses:
                loss_coef = self.config.hp.get(f'loss_coefs.{loss_name}', 1)
                (loss_coef * losses[loss_name]).backward(retain_graph=True)
                grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_norm_value)

                self.optim.step()
                self.writer.add_scalar(f'Train/{loss_name}', losses[loss_name].item(), self.num_iters_done)
                self.writer.add_scalar(f'Grad_norm/from_{loss_name}', grad_norm, self.num_iters_done)