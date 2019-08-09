import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    maskrcnn_loss,
    maskrcnn_inference,
    keypointrcnn_loss,
    keypointrcnn_inference,
    project_masks_on_boxes)

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
            raise NotImplementedError

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

    def run_densepose_head_during_eval(self):
        raise NotImplementedError


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


def compute_ious(proposals, matched_idxs, targets):
    pass


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou