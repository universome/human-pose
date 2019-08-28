import os
import json

import torch
import torch.nn as nn
import numpy as np
from firelab import BaseTrainer
from torch.nn.utils.clip_grad import clip_grad_norm_

from src.models.detection.backbone_utils import resnet_fpn_backbone
from src.models import DensePoseRCNN
from src.utils.densepose_cocoeval import denseposeCOCOeval
from src.utils.dp_targets import _encodePngData
from src.utils.comm import reduce_loss_dict, synchronize, all_gather, is_main_process
from src.structures.bbox import Bbox
from src.dataloaders.construct import construct_dataloader
from src.dataloaders.utils import batch_to
from src.utils.training_utils import load_model_state


class DensePoseRCNNTrainer(BaseTrainer):
    def __init__(self, config):
        super(DensePoseRCNNTrainer, self).__init__(config)

        if self.device_name == 'cpu':
            self.logger.warn('I am running on CPU and gonna be very slow...')

        self.is_distributed = self.config.get('is_distributed', False)

    def init_models(self):
        # backbone = mobilenet_v2(pretrained=True)
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        self.model = DensePoseRCNN(backbone, num_maskrcnn_classes=2, box_detections_per_img=20)

        if self.config.has('load_checkpoint.model'):
            model_state = load_model_state(self.config.load_checkpoint.model, self.device_name)
            self.model.load_state_dict(model_state)

        self.model = self.model.to(self.device_name)

        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=self.gpus, output_device=self.gpus[0])

    def init_dataloaders(self):
        self.train_dataloader = construct_dataloader(self.config, True, self.is_distributed, True)
        self.val_dataloader = construct_dataloader(self.config, False, self.is_distributed, False)

    def init_optimizers(self):
        # TODO: lr scheduling
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim.to_dict())
        # self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, 1e-6, 1e-2)

        if self.config.has('load_checkpoint.optim'):
            self.optim.load_state_dict(torch.load(self.config.load_checkpoint.optim, map_location=self.device_name))

    def train_on_batch(self, batch):
        images, targets = batch_to(batch, self.device_name)

        self.model.train()

        losses = self.model(images, targets)
        coefs = [self.config.hp.get(f'loss_coefs.{loss_name}', 1) for loss_name in losses]
        total_loss = sum(c * l for c, l in zip(coefs, losses.values()))

        if (self.num_iters_done + 1) % self.config.hp.get('grad_accum', 1) == 0:
            # Run with synchronization (done automatically during backward)
            total_loss.backward()

            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_norm_value)

            self.optim.step()
            self.optim.zero_grad()

            self.write_scalar(f'Grad_norm', grad_norm, self.num_iters_done)
        else:
            # Run without synchronization
            if self.is_distributed:
                with self.model.no_sync():
                    total_loss.backward()

                # TODO: does not this sync? Is it possible to log sensible values without syncing then?
                losses = reduce_loss_dict(losses)
                total_loss = sum(c * l for c, l in zip(coefs, losses.values()))
            else:
                total_loss.backward()

            if is_main_process():
                for loss_name in losses:
                    self.write_scalar(f'Train/{loss_name}', losses[loss_name].item(), self.num_iters_done)

                self.write_scalar(f'Train/total_loss', total_loss.item(), self.num_iters_done)

    def write_scalar(self, *args):
        if not is_main_process(): return

        self.writer.add_scalar(*args)

    def validate(self):
        self.model.eval()

        dp_predictions = self.run_inference(self.val_dataloader)

        if self.is_distributed:
            synchronize()
            # TODO: in maskrcnn-benchmark they filter our predictions for the same image from different GPUs?
            #dp_predictions = accumulate_predictions_from_multiple_gpus(dp_predictions)
            dp_predictions = all_gather(dp_predictions)

            if not is_main_process():
                return

            dp_predictions = [p for gpu_preds in dp_predictions for p in gpu_preds]

        if len(dp_predictions) == 0:
            self.logger.warn(f'Skipping validation on epoch {self.num_epochs_done}, because no predictions are made')
            return

        val_results_dir = os.path.join(self.paths.custom_data_path, 'val_results')
        if not os.path.isdir(val_results_dir):
            os.mkdir(val_results_dir)

        results_file_path = f'{val_results_dir}/iter-{self.num_iters_done}.json'

        with open(results_file_path, 'w') as f:
            json.dump(dp_predictions, f)

        for iou_type in self.config.get('val_iou_types', ['uv', 'segm', 'bbox']):
            # Now we can validate our predictions
            # TODO: Compare with validation in json_dataset_evaluator.py
            # TODO: It feels like we should also use test_sigma=0.255
            print(f'Running {iou_type} validation...')
            coco = self.val_dataloader.dataset.coco
            coco_res = coco.loadRes(results_file_path)
            coco_eval = denseposeCOCOeval(self.config.data.eval_data, coco, coco_res, iouType=iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            self.writer.add_scalar(f'val/mAP_at_IoU_0_50__0_95_{iou_type}', coco_eval.stats[0], self.num_epochs_done)

    def run_inference(self, dataloader):
        results = []

        with torch.no_grad():
            for images, targets in dataloader:
                preds = self.model([img.to(self.device_name) for img in images])

                for img_idx in range(len(images)):
                    for bbox_idx in range(len(preds[img_idx]['boxes'])):
                        if preds[img_idx]['boxes'][bbox_idx].numel() == 0: continue

                        uv_data = torch.stack([
                            preds[img_idx]['dp_classes'][bbox_idx].float(),
                            preds[img_idx]['dp_u_coords'][bbox_idx],
                            preds[img_idx]['dp_v_coords'][bbox_idx]],
                            dim=0).data.cpu().numpy()
                        uv_data[1:] = uv_data[1:] * 255  # Converting to 255 range
                        uv_png = _encodePngData(uv_data.astype(np.uint8))
                        bbox = Bbox.from_torch_tensor(preds[img_idx]['boxes'][bbox_idx])

                        results.append({
                            "image_id": targets[img_idx]['image_id'],
                            "category_id": 1,
                            "uv_shape": uv_data.shape,
                            "score": preds[img_idx]['scores'][bbox_idx].item(),
                            # TODO: We have to use uv_data shape parameters, because densepose_cocoeval
                            #  fails otherwise in different places
                            "bbox": [bbox.x, bbox.y, uv_data.shape[2], uv_data.shape[1]],
                            "uv_data": uv_png
                        })
        return results
