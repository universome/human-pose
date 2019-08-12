from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads

from .densepose_roi_heads import DensePoseRoIHeads

__all__ = [
    "DensePoseRCNN"
]


class DensePoseRCNN(MaskRCNN):
    def __init__(self, backbone, num_maskrcnn_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 dp_head_output_size: int=50):

        super(DensePoseRCNN, self).__init__(
            backbone, num_maskrcnn_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            mask_roi_pool, mask_head, mask_predictor)

        # Dirty tricks to make DensePose RoI heads work without
        # copypasting the whole torchvision.models.detection and changing staff there
        # Another option would be to copypaste all properties of self.roi_heads manually
        # (because they are created in FasterRCNN module and we cannot create DensePoseRoIHeads correctly ourselves)
        # but it does not seem much better
        #
        # TODO: We'll have to copypaste the whole torchvision.models.detection anyway.
        #       So you can do fix it now.
        self.roi_heads.forward = lambda *args, **kwargs: \
            DensePoseRoIHeads.forward(self.roi_heads, *args, **kwargs)
        self.roi_heads.run_densepose_head_during_training = lambda *args, **kwargs: \
            DensePoseRoIHeads.run_densepose_head_during_training(self.roi_heads, *args, **kwargs)
        self.roi_heads.run_densepose_head_during_eval = lambda *args, **kwargs: \
            DensePoseRoIHeads.run_densepose_head_during_eval(self.roi_heads, *args, **kwargs)

        self.roi_heads.densepose_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=14,
            sampling_ratio=2)
        self.roi_heads.densepose_head = MaskRCNNHeads(backbone.out_channels, (256, 256, 256, 256), 1)

        # TODO: maybe we should put sigmoid on top (UV coords are always in [0,1] range)
        self.roi_heads.densepose_uv_predictor = nn.Sequential(
            misc_nn_ops.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            misc_nn_ops.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            misc_nn_ops.Conv2d(256, 48, 1, 1, 0),
        )
        self.roi_heads.densepose_class_predictor = nn.Sequential(
            misc_nn_ops.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            misc_nn_ops.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            misc_nn_ops.Conv2d(256, 25, 1, 1, 0),
        )
