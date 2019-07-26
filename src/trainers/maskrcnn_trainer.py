import torch
import numpy as np
from firelab import BaseTrainer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models import mobilenet_v2
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO


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

    def init_dataloaders(self):
        coco = COCO(f'{self.config.coco_data_dir}/annotations/instances_val2014.json')

        def collate_batch(batch):
            images = [img for img, target in batch]
            targets = [target for img, target in batch]

            return images, targets

        def torchvision_target_format(target):
            return {
                'image_id': target[0]['image_id'],
                'labels': torch.Tensor([bbox['category_id'] for bbox in target]).long(),
                'boxes': torch.Tensor([bbox['bbox'] for bbox in target]),
                'masks': torch.Tensor([coco.annToMask(bbox) for bbox in target])
            }

        # train_ds = CocoDetection(f'{self.config.coco_data_dir}/coco_train2014',
        #                          f'{self.config.coco_data_dir}/annotations/instances_train2014.json',
        #                          transform=transforms.ToTensor())
        val_ds = CocoDetection(f'{self.config.coco_data_dir}/coco_val2014',
                                 f'{self.config.coco_data_dir}/annotations/instances_val2014.json',
                                 transform=transforms.ToTensor(),
                                 target_transform=torchvision_target_format)

        self.train_dataloader = DataLoader(val_ds, batch_size=4, collate_fn=collate_batch)
        self.val_dataloader = DataLoader(val_ds, batch_size=4, collate_fn=collate_batch)

    def train_on_batch(self, batch):
        images, targets = batch
        images = [img.to(self.config.firelab.device_name) for img in images]

        self.model.train()
        losses = self.model(images, targets)

        # self.writer.add_scalar('Train/clf_loss', clf_loss.item(), self.num_iters_done)
        # self.writer.add_scalar('Train/bbox_loss', bbox_loss.item(), self.num_iters_done)
        # self.writer.add_scalar('Train/mask_loss', mask_loss.item(), self.num_iters_done)

