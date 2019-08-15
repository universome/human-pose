import io
import os
import json
from typing import List
import base64

import torch
import numpy as np
from firelab import BaseTrainer
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchvision import transforms
from PIL import Image

from src.models.detection.backbone_utils import resnet_fpn_backbone
from src.dataloaders.coco import CocoDetection
from src.models import DensePoseRCNN
from src.utils.densepose_cocoeval import denseposeCOCOeval
from src.structures.bbox import Bbox


class DensePoseRCNNTrainer(BaseTrainer):
    def __init__(self, config):
        super(DensePoseRCNNTrainer, self).__init__(config)

        if self.device_name == 'cpu':
            self.logger.warn('I am running on CPU and gonna be very slow...')

    def init_models(self):
        # backbone = mobilenet_v2(pretrained=True)
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        # TODO: what is the correct number of classes we should use? Binary (human/bg)?
        self.model = DensePoseRCNN(backbone, num_maskrcnn_classes=2, box_detections_per_img=20)
        self.model = self.model.to(self.device_name)

    def init_dataloaders(self):
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
                'labels': torch.Tensor([obj['category_id'] for obj in target]).long().to(self.device_name),
                'boxes': torch.Tensor([xywh_to_xyxy(obj['bbox']) for obj in target]).to(self.device_name),
                'masks': torch.Tensor([obj['mask'] for obj in target]).to(self.device_name),
                'coco_anns': target,
            }

        # train_ds = CocoDetection(f'{self.config.coco_data_dir}/train2014',
        #                          f'{self.config.coco_data_dir}/annotations/instances_train2014.json',
        #                          transform=transforms.ToTensor(),
        #                          target_transform=torchvision_target_format)
        # val_ds = CocoDetection(f'{self.config.coco_data_dir}/val2014',
        #                          f'{self.config.coco_data_dir}/annotations/instances_val2014.json',
        #                          transform=transforms.ToTensor(),
        #                          target_transform=torchvision_target_format)
        train_ds = CocoDetection(f'{self.config.data.coco_dir}/train2014',
                                 f'{self.config.data.annotations_dir}/densepose_coco_2014_tiny.json',
                                 transform=transforms.ToTensor(),
                                 target_transform=torchvision_target_format)
        val_ds = train_ds

        self.train_dataloader = DataLoader(train_ds, batch_size=self.config.hp.batch_size, collate_fn=collate_batch)
        self.val_dataloader = DataLoader(val_ds, batch_size=1, collate_fn=collate_batch)

    def init_optimizers(self):
        # TODO: lr scheduling
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim.to_dict())
        # self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, 1e-6, 1e-2)

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
            # self.lr_scheduler.step()

        self.writer.add_scalar(f'Grad_norm', grad_norm, self.num_iters_done)
        for loss_name in losses:
            self.writer.add_scalar(f'Train/{loss_name}', losses[loss_name].item(), self.num_iters_done)
        self.writer.add_scalar(f'Train/total_loss', total_loss.item(), self.num_iters_done)

    def validate(self):
        self.model.eval()

        dp_predictions = []

        for images, targets in self.val_dataloader:
            with torch.no_grad():
                preds = self.model(torch.stack(images).to(self.device_name))

            for img_idx in range(len(images)):
                for bbox_idx in range(len(preds[img_idx]['boxes'])):
                    if preds[img_idx]['boxes'][bbox_idx].numel() == 0: continue

                    uv_data = torch.stack([
                        preds[img_idx]['dp_classes'][bbox_idx].float(),
                        preds[img_idx]['dp_u_coords'][bbox_idx],
                        preds[img_idx]['dp_v_coords'][bbox_idx]],
                    dim=0).data.cpu().numpy()
                    uv_data = uv_data.transpose(0, 2, 1) # [W x H] to [H x W], because densepose_cocoeval requires this
                    uv_data[1:] = uv_data[1:] * 255 # Converting to 255 range
                    uv_png = _encodePngData(uv_data.astype(np.uint8))
                    bbox = Bbox.from_torch_tensor(preds[img_idx]['boxes'][bbox_idx])

                    dp_predictions.append({
                        "image_id": targets[img_idx]['image_id'],
                        "category_id": 1,
                        "uv_shape": [3, round(bbox.height), round(bbox.width)],
                        "score": preds[img_idx]['scores'][bbox_idx].item(),
                        "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                        "uv_data": uv_png
                    })

        if len(dp_predictions) == 0:
            self.logger.warn(f'Skipping validation on epoch {self.num_epochs_done}, because no predictions are made')
            return

        val_results_dir = os.path.join(self.paths.custom_data_path, 'val_results')
        if not os.path.isdir(val_results_dir):
            os.mkdir(val_results_dir)

        results_file_path = f'{val_results_dir}/epoch-{self.num_epochs_done}.json'

        with open(results_file_path, 'w') as f:
            json.dump(dp_predictions, f)

        # Now we can validate our predictions
        coco = self.val_dataloader.dataset.coco
        coco_res = coco.loadRes(results_file_path)
        coco_eval = denseposeCOCOeval(self.config.data.eval_data, coco, coco_res)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.writer.add_scalar('val/mAP_at_IoU_0_50__0_95', coco_eval.stats[0], self.num_epochs_done)


def _encodePngData(arr):
    """
    Encode array data as a PNG image using the highest compression rate
    @param arr [in] Data stored in an array of size (3, M, N) of type uint8
    @return Base64-encoded string containing PNG-compressed data
    """
    assert len(arr.shape) == 3, "Expected a 3D array as an input," \
            " got a {0}D array".format(len(arr.shape))
    assert arr.shape[0] == 3, "Expected first array dimension of size 3," \
            " got {0}".format(arr.shape[0])
    assert arr.dtype == np.uint8, "Expected an array of type np.uint8, " \
            " got {0}".format(arr.dtype)
    data = np.moveaxis(arr, 0, -1)
    im = Image.fromarray(data)
    fStream = io.BytesIO()
    im.save(fStream, format='png', optimize=True)
    s = fStream.getvalue()
    # return s.encode('base64')
    return base64.b64encode(s).decode('utf-8')
