from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    maskrcnn_loss,
    maskrcnn_inference)
import numpy as np

from src.utils.dp_targets import  create_target_for_dp, compute_dp_cls_loss, compute_dp_uv_loss
from src.structures.bbox import Bbox

__all__ = [
    "DensePoseRoIHeads"
]


class DensePoseRoIHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super(DensePoseRoIHeads, self).__init__(*args, **kwargs)

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        # TODO: add keypoints head


        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}

        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]

            if self.training:
                # during training, only focus on positive boxes
                mask_proposals, pos_matched_idxs = get_positive_proposals(proposals, labels, matched_idxs)

            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # Compute DensePose branch
        if self.training:
            dp_losses = self.run_densepose_head_during_training(
                features, image_shapes, targets,
                proposals, labels, matched_idxs
            )
            losses.update(dp_losses)
        else:
            dp_proposals = [p["boxes"] for p in result]
            dp_classes, dp_u_coords, dp_v_coords = self.run_densepose_head_during_eval(features, image_shapes, dp_proposals)

            for cls_idx, v_coords, u_coords, r in zip(dp_classes, dp_u_coords, dp_v_coords, result):
                r["dp_u_coords"] = u_coords
                r["dp_v_coords"] = v_coords
                r["dp_classes"] = cls_idx

        return result, losses

    def run_densepose_head_during_training(self, features, image_shapes, targets,
                                                 proposals, labels, matched_idxs):
        # during training, only focus on boxes of foreground class
        dp_proposals, pos_matched_idxs = get_positive_proposals(proposals, labels, matched_idxs)

        assert len(dp_proposals) == len(pos_matched_idxs)
        assert all([len(ps) == len(idxs) for ps, idxs in zip(dp_proposals, pos_matched_idxs)])

        dp_features = self.densepose_roi_pool(features, dp_proposals, image_shapes)

        # Flattening out our data
        coco_anns_flat = [targets[img_i]['coco_anns'][pos_matched_idxs[img_i][p_i]] for img_i, obj_idx in enumerate(pos_matched_idxs) for p_i in range(len(obj_idx))]
        gt_bboxes_flat = [targets[img_i]['boxes'][pos_matched_idxs[img_i][p_i]] for img_i, obj_idx in enumerate(pos_matched_idxs) for p_i in range(len(obj_idx))]
        dp_proposals = [p for ps in dp_proposals for p in ps]
        pos_matched_idxs = [(img_i, bb_i) for img_i, bbox_idxs in enumerate(pos_matched_idxs) for bb_i in bbox_idxs]

        # Filtering out those proposals, which correspond to objects that do not have densepose annotation
        has_dp = lambda i: 'dp_masks' in coco_anns_flat[i]
        gt_bboxes_flat = [bb for i, bb in enumerate(gt_bboxes_flat) if has_dp(i)]
        dp_features = torch.stack([f for i, f in enumerate(dp_features) if has_dp(i)])
        dp_proposals = [p for i, p in enumerate(dp_proposals) if has_dp(i)]
        pos_matched_idxs = [idx for i, idx in enumerate(pos_matched_idxs) if has_dp(i)]

        dp_features = self.densepose_head(dp_features)
        dp_cls_logits = self.densepose_class_predictor(dp_features)
        dp_uv_preds = self.densepose_uv_predictor(dp_features)

        # Converting to Bbox format
        dp_proposals = [Bbox.from_torch_tensor(bb) for bb in dp_proposals]
        gt_bboxes_flat = [Bbox.from_torch_tensor(bb) for bb in gt_bboxes_flat]

        dp_head_output_size = dp_cls_logits.size(-1)
        dp_targets = [create_target_for_dp(targets[img_idx]['coco_anns'][bbox_idx], gt_bbox, proposal, dp_head_output_size) \
                        for (img_idx, bbox_idx), gt_bbox, proposal in zip(pos_matched_idxs, gt_bboxes_flat, dp_proposals)]

        if len(dp_targets) == 0:
            return {
                'dp_cls_bg_loss': torch.zeros(1, device=dp_cls_logits.device),
                'dp_cls_fg_loss': torch.zeros(1, device=dp_cls_logits.device),
                'dp_uv_loss': torch.zeros(1, device=dp_cls_logits.device)
            }

        dp_cls_losses = torch.Tensor([compute_dp_cls_loss(cls_logits, dp_trg) for cls_logits, dp_trg in zip(dp_cls_logits, dp_targets)])
        dp_uv_losses = torch.Tensor([compute_dp_uv_loss(uv_preds, dp_trg) for uv_preds, dp_trg in zip(dp_uv_preds, dp_targets)])

        dp_cls_bg_loss, dp_cls_fg_loss = dp_cls_losses[:, 0].mean(), dp_cls_losses[:, 1].mean()
        dp_u_loss, dp_v_loss = dp_uv_losses[:, 0].mean(), dp_uv_losses[:, 1].mean()

        return {
            'dp_cls_bg_loss': dp_cls_bg_loss,
            'dp_cls_fg_loss': dp_cls_fg_loss,
            'dp_u_loss': dp_u_loss,
            'dp_v_loss': dp_v_loss
        }

    def run_densepose_head_during_eval(self, features, image_shapes, dp_proposals):
        # TODO: not sure if features will always have a zero layer property...
        if len(dp_proposals) == 1 and dp_proposals[0].numel() == 0:
            dp_classes = [torch.empty(0, 1, 1)]
            dp_u_coords = [torch.empty(0, 1, 1)]
            dp_v_coords = [torch.empty(0, 1, 1)]

            return dp_classes, dp_u_coords, dp_v_coords

        dp_features = self.densepose_roi_pool(features, dp_proposals, image_shapes)
        dp_features = self.densepose_head(dp_features)
        dp_cls_logits = self.densepose_class_predictor(dp_features)
        dp_uv_coords = self.densepose_uv_predictor(dp_features)

        dp_proposals_flat = [p for ps in dp_proposals for p in ps]
        bboxes = [Bbox.from_torch_tensor(p).discretize() for p in dp_proposals_flat]
        target_sizes = [(bb.width, bb.height) for bb in bboxes]
        dp_cls_logits = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                            for x, s in zip(dp_cls_logits, target_sizes)]
        dp_uv_coords = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                            for x, s in zip(dp_uv_coords, target_sizes)]

        results = [extract_dp_predictions(l.unsqueeze(0), uv.unsqueeze(0), [[p]], [s]) \
                       for l, uv, p, s in zip(dp_cls_logits, dp_uv_coords, dp_proposals_flat, target_sizes)]
        dp_classes, dp_u_coords, dp_v_coords = zip(*results)

        num_boxes_per_image = [len(ps) for ps in dp_proposals]
        dp_classes = np.split([x[0].squeeze(0) for x in dp_classes], num_boxes_per_image)
        dp_u_coords = np.split([x[0].squeeze(0) for x in dp_u_coords], num_boxes_per_image)
        dp_v_coords = np.split([x[0].squeeze(0) for x in dp_v_coords], num_boxes_per_image)

        return dp_classes, dp_u_coords, dp_v_coords


def get_positive_proposals(proposals, labels, matched_idxs):
    # during training, only focus on positive boxes
    num_images = len(proposals)
    positive_proposals = []
    pos_matched_idxs = []

    for img_id in range(num_images):
        pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
        positive_proposals.append(proposals[img_id][pos])
        pos_matched_idxs.append(matched_idxs[img_id][pos])

    return positive_proposals, pos_matched_idxs


def extract_dp_predictions(dp_cls_logits, dp_uv_coords, dp_proposals, target_sizes):
    # TODO: We can try and parallelize the computation by first padding to largest value
    #       then stacking and computing for all. But this does not work because of OOM.
    #       So, we should sort by size and split into batches.
    # dp_cls_logits = pad_to_largest(dp_cls_logits)
    # dp_uv_coords = pad_to_largest(dp_uv_coords)

    # Convert to N x W x H x C for convenience
    dp_cls_logits = dp_cls_logits.permute(0, 2, 3, 1)
    dp_uv_coords = dp_uv_coords.permute(0, 2, 3, 1)

    # Leaving only the most confident body part
    dp_classes = dp_cls_logits.argmax(dim=3, keepdim=True)
    idx = torch.arange(dp_cls_logits.size(3)).view(1, 1, 1, -1).repeat(*dp_cls_logits.shape[:3], 1)
    max_cls_mask = idx.to(dp_classes.device) == dp_classes
    dp_classes = dp_classes.squeeze(3)

    # Extract UV-coords
    dp_u_coords, dp_v_coords = dp_uv_coords[:, :, :, :24], dp_uv_coords[:, :, :, 24:]

    # Add dummy background predictions so we can use masked_select later (make the same dimensionality as dp_classes)
    bg_dummy_coords = torch.zeros(*dp_u_coords.shape[:3], 1).to(dp_u_coords.device)
    dp_u_coords = torch.cat([bg_dummy_coords, dp_u_coords], dim=3)
    dp_v_coords = torch.cat([bg_dummy_coords, dp_v_coords], dim=3)

    # Select coordinates for the most confident class
    # TODO: if there will be two max classes with the same value, this will fail,
    #       because we'll have more values than possible for view
    #       and this can really happen in low precision I suppose
    dp_u_coords = dp_u_coords.masked_select(max_cls_mask).view(dp_u_coords.shape[:3])
    dp_v_coords = dp_v_coords.masked_select(max_cls_mask).view(dp_v_coords.shape[:3])

    num_boxes_per_image = [len(ps) for ps in dp_proposals]
    dp_classes = dp_classes.split(num_boxes_per_image, dim=0)
    dp_u_coords = dp_u_coords.split(num_boxes_per_image, dim=0)
    dp_v_coords = dp_v_coords.split(num_boxes_per_image, dim=0)

    # target_sizes = torch.tensor(target_sizes).split(num_boxes_per_image, dim=0)
    # dp_classes = [[x[:s[0], :s[1]] for x, s in zip(xs, ss)] for xs, ss in zip(dp_classes, target_sizes)]
    # dp_u_coords = [[x[:s[0], :s[1]] for x, s in zip(xs, ss)] for xs, ss in zip(dp_u_coords, target_sizes)]
    # dp_v_coords = [[x[:s[0], :s[1]] for x, s in zip(xs, ss)] for xs, ss in zip(dp_v_coords, target_sizes)]

    return dp_classes, dp_u_coords, dp_v_coords


def pad_to_largest(xs: List[torch.Tensor], pad_val:float=-10000) -> torch.Tensor:
    """
        Pads to largest width and height and returns stacked tensor
        Assumes that each tensor in the list is in C x W x H format
    """
    max_w = max([x.size(1) for x in xs])
    max_h = max([x.size(2) for x in xs])
    pad_sizes = [(0, max_w - x.size(1), 0, max_h - x.size(2)) for x in xs]
    padded = [F.pad(x, pad=p, mode='constant', value=pad_val) for p, x in zip(pad_sizes, xs)]

    return torch.stack(padded)

