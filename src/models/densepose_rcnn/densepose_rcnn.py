from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from src.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads
from src.models.detection.transform import resize_boxes, paste_masks_in_image, resize_keypoints

from .densepose_roi_heads import DensePoseRoIHeads
from src.structures.bbox import Bbox

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
        self.transform.postprocess = lambda *args, **kwargs: \
            densepose_postprocess(self.transform, *args, **kwargs)

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


def densepose_postprocess(transform_module, result, image_shapes, original_image_sizes):
    if transform_module.training:
        return result
    for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
        boxes = pred["boxes"]
        boxes = resize_boxes(boxes, im_s, o_im_s)
        result[i]["boxes"] = boxes
        if "masks" in pred:
            masks = pred["masks"]
            masks = paste_masks_in_image(masks, boxes, o_im_s)
            result[i]["masks"] = masks
        if "keypoints" in pred:
            keypoints = pred["keypoints"]
            keypoints = resize_keypoints(keypoints, im_s, o_im_s)
            result[i]["keypoints"] = keypoints
        if "dp_cls_logits" in pred:
            pred["dp_classes"], pred["dp_u_coords"], pred["dp_v_coords"] = \
                postprocess_dp_predictions(pred["dp_cls_logits"], pred["dp_uv_coords"], boxes)

    return result


def postprocess_dp_predictions(dp_cls_logits, dp_uv_coords, boxes):
    if len(boxes) == 0:
        dp_classes = torch.empty(0, 10, 10)
        dp_u_coords = torch.empty(0, 10, 10)
        dp_v_coords = torch.empty(0, 10, 10)

        return dp_classes, dp_u_coords, dp_v_coords

    bboxes = [Bbox.from_torch_tensor(p).discretize() for p in boxes]
    target_sizes = [(bb.width, bb.height) for bb in bboxes]
    dp_cls_logits = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                     for x, s in zip(dp_cls_logits, target_sizes)]
    dp_uv_coords = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                    for x, s in zip(dp_uv_coords, target_sizes)]

    results = [get_most_confident_dp_predictions(l, uv) for l, uv in zip(dp_cls_logits, dp_uv_coords)]
    dp_classes, dp_u_coords, dp_v_coords = zip(*results)

    return dp_classes, dp_u_coords, dp_v_coords


def get_most_confident_dp_predictions(dp_cls_logits, dp_uv_coords):
    # TODO: We can try and parallelize the computation by first padding to largest value
    #       then stacking and computing for all. But this does not work because of OOM.
    #       So, we should sort by size and split into batches.

    # Convert to [C x W x H] to [W x H x C] for convenience
    c, w, h = dp_cls_logits.shape
    dp_cls_logits = dp_cls_logits.permute(1, 2, 0)
    dp_uv_coords = dp_uv_coords.permute(1, 2, 0)

    # Leaving only the most confident body part
    dp_classes = dp_cls_logits.argmax(dim=2, keepdim=True)
    idx = torch.arange(c).view(1, 1, -1).repeat(w, h, 1)
    max_cls_mask = idx.to(dp_classes.device) == dp_classes
    dp_classes = dp_classes.squeeze(2)

    # Extract UV-coords
    dp_u_coords, dp_v_coords = dp_uv_coords[:, :, :24], dp_uv_coords[:, :, 24:]

    # Add dummy background predictions so we can use masked_select later (make the same dimensionality as dp_classes)
    bg_dummy_coords = torch.zeros(w, h, 1).to(dp_u_coords.device)
    dp_u_coords = torch.cat([bg_dummy_coords, dp_u_coords], dim=2)
    dp_v_coords = torch.cat([bg_dummy_coords, dp_v_coords], dim=2)

    # Select coordinates for the most confident class
    dp_u_coords = dp_u_coords.masked_select(max_cls_mask).view(w, h)
    dp_v_coords = dp_v_coords.masked_select(max_cls_mask).view(w, h)

    return dp_classes, dp_u_coords, dp_v_coords
