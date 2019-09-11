from torch import nn
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from src.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from src.models.detection.faster_rcnn import FasterRCNN
from src.models.detection.mask_rcnn import MaskRCNN

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
                 bbox_reg_weights=None):

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
            bbox_reg_weights)

        self.roi_heads.densepose_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=14,
            sampling_ratio=2)
        self.roi_heads.densepose_head = MaskRCNNHeads(backbone.out_channels, (256, 256, 256, 256), 1)

        # TODO: maybe we should put sigmoid on top (UV coords are always in [0,1] range)?
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
        self.roi_heads.densepose_mask_predictor = nn.Sequential(
            misc_nn_ops.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            misc_nn_ops.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            misc_nn_ops.Conv2d(256, 15, 1, 1, 0),
        )
