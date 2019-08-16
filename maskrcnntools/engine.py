import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
from torch.nn.utils.clip_grad import clip_grad_norm_

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
from logger import Logger

grad_clip_norm_value = 100

class MaskRCNNTrainer:
    def __init__(self, 
                 model, 
                 optimizer, 
                 data_loader, 
                 device, 
                 init_epoch=0, 
                 print_freq=10,
                 logger=None,
                 emergency=True):
        self.model = model 
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.init_epoch = init_epoch
        self.epoch = init_epoch
        self.print_freq = print_freq
        self.logger = logger
        self.emergency = emergency
        
    def train_one_epoch(self, lr_schedule='cyclic'):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(self.epoch)

        lr_scheduler = None
        if (self.epoch == 0):
            if lr_schedule == 'warmup':
                warmup_factor = 1. / 1000
                warmup_iters = min(1000, len(self.data_loader) - 1)

                lr_scheduler = utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
            elif lr_schedule == 'cyclic':
                lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 1e-6, 1e-2)

        for iteration, (images, targets) in enumerate(metric_logger.log_every(self.data_loader, self.print_freq, header)):
            with torch.autograd.detect_anomaly():
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if self.emergency is True:
                    if not math.isfinite(loss_value):
                        print()
                        print("Loss is {}, stopping training".format(loss_value))
                        print(loss_dict_reduced)
                        sys.exit(1)

                self.optimizer.zero_grad()
                losses.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), grad_clip_norm_value)
                self.optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                if self.logger is not None:
                    if iteration % 50 == 0:
                        # 1. Log scalar values (scalar summary)
                        info = {'loss': losses_reduced, **loss_dict_reduced}

                        for tag, value in info.items():
                            self.logger.scalar_summary(tag, value, iteration+1)

                        # 2. Log values and gradients of the parameters (histogram summary)
                        for tag, value in self.model.named_parameters():
                            tag = tag.replace('.', '/')
                            self.logger.histo_summary(tag, value.data.cpu().numpy(), iteration+1)
                       
        self.epoch += 1         


    def _get_iou_types(self):
        model_without_ddp = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types


    @torch.no_grad()
    def evaluate(self):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(self.data_loader.dataset)
        iou_types = _get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for image, targets in metric_logger.log_every(self.data_loader, 100, header):
            image = list(img.to(self.device) for img in image)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator
