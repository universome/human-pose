from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from pycocotools import mask as mask_utils

from src.structures.bbox import Bbox


def create_target_for_dp(coco_ann: Dict,
                         gt_bbox: Bbox,
                         proposal: Bbox,
                         dp_head_output_size: int):
    """
    This function takes a raw coco annotation and prepares target, which is then
    passed into the model to compute the loss. It is not a trivial function, because
    we have to account for the following things:
        - we have annotations for ground-truth bbox, but model predictions are made for bbox proposals
        - we can have complex data augmentations (flip, zoom, etc.)

    Arguments:
        - coco_ann — raw coco annotation with the following required fields:
                dp_x, dp_y, dp_I, dp_U, dp_V, dp_masks — these are all densepose annotatoins
        - gt_bbox — ground truth bounding box. We can't extract it from coco_ann,
                because MaskRCNN resized it (via GeneralizedRCNNTransform)
        - proposal — box proposals for an image (produced by detection head)
        - dp_head_output_size — what is the output size of the model (usually it is 56x56)

    Output:
        - points_idx:List[List[int]] — indices showing what points are labeled in the bbox (~100-150 per bbox)
        - body_part_classes — tensor of size len(points_idx), which specifies body part class for each pixel
        - uv_coords — tensor of size len(points_idx), which specifies body
    """

    # TODO: use dp_masks field
    # TODO: add horizontal flip
    # TODO: add tests

    # Since densepose annotations are computed for gt bbox (rescaled to 256x256)
    # we should take the intersection between gt bbox and proposal bbox
    # and leave only the pixels which lie in this intersection

    # Compute coordinates in the coordinate system where (0,0) is the whole image left upper corner
    img_dp_x = gt_bbox.x + np.array(coco_ann['dp_x']) / 256 * gt_bbox.width
    img_dp_y = gt_bbox.y + np.array(coco_ann['dp_y']) / 256 * gt_bbox.height

    # Compute coordinates in the coordinate system where (0,0) is proposal bbox left upper corner
    proposal_dp_x = img_dp_x - proposal.x
    proposal_dp_y = img_dp_y - proposal.y

    # Removing those UV points that lie outside the proposal
    out_of_proposal_mask = (proposal_dp_x < 0) + \
                           (proposal_dp_x > proposal.width) + \
                           (proposal_dp_y < 0) + \
                           (proposal_dp_y > proposal.height)

    proposal_dp_x = proposal_dp_x[~out_of_proposal_mask]
    proposal_dp_y = proposal_dp_y[~out_of_proposal_mask]
    dp_U = np.array(coco_ann['dp_U'])[~out_of_proposal_mask]
    dp_V = np.array(coco_ann['dp_V'])[~out_of_proposal_mask]
    dp_I = np.array(coco_ann['dp_I'])[~out_of_proposal_mask]

    # We should to rescale the pixels in such a way
    # that it fits to model output size
    proposal_dp_x = proposal_dp_x / proposal.width * (dp_head_output_size - 1)
    proposal_dp_y = proposal_dp_y / proposal.height * (dp_head_output_size - 1)

    # Since proposal_dp_x and proposal_dp_y are fractional
    # We should convert them to integers to be able to select points on mask
    # TODO: U,V coords should we should do it carefully, because they are a little bit off?
    #       i.e. use bilinear interpolation, for example?

    proposal_dp_x = np.round(proposal_dp_x).astype(int)
    proposal_dp_y = np.round(proposal_dp_y).astype(int)

    assert proposal_dp_x.min() >= 0
    assert proposal_dp_x.max() < dp_head_output_size
    assert proposal_dp_y.min() >= 0
    assert proposal_dp_y.max() < dp_head_output_size

    return {
        'dp_U': dp_U,
        'dp_V': dp_V,
        'dp_I': dp_I,
        'dp_x': proposal_dp_x,
        'dp_y': proposal_dp_y,
        'dp_masks': extract_dp_masks_from_coco_ann(coco_ann, dp_head_output_size),
    }


def compute_dp_uv_loss(uv_preds: torch.FloatTensor, dp_targets: Dict):
    """
    Computes loss for densepose

    :param dp_targets: dict of preprocessed targets for densepose
    :param predictions:
        tensor of UV-coordinates predictions of size [M x M x 48],
        where 48 is used because for each densepose class we use a separate UV predictor


    :return: Tuple(torch.FloatTensor, torch.FloatTensor)
    """
    assert all(dp_targets['dp_I'] >= 1) # dp_I of 0 is background

    u_preds = uv_preds[dp_targets['dp_I'] - 1, dp_targets['dp_y'], dp_targets['dp_x']]
    v_preds = uv_preds[dp_targets['dp_I'] + 24 - 1, dp_targets['dp_y'], dp_targets['dp_x']]

    # TODO: compute and return loss for each label separately for visualizaiton purposes
    u_loss = (u_preds - torch.tensor(dp_targets['dp_U']).float().to(u_preds.device)).pow(2).mean()
    v_loss = (v_preds - torch.tensor(dp_targets['dp_V']).float().to(v_preds.device)).pow(2).mean()

    return u_loss, v_loss


def compute_dp_cls_loss(cls_logits: torch.LongTensor, dp_targets: Dict):
    """
    :param dp_targets: dict of preprocessed targets for densepose
    :param cls_logits: tensor of class logits of size [M x M x 25]
    :return: classification loss
    """
    # TODO:
    #   It feels like we should weight losses for different body parts by number of points.
    #   Or we can just average by body part first and than average out all losses

    bg_loss = compute_dp_bg_cls_loss(cls_logits, dp_targets)
    fg_loss = F.cross_entropy(
        cls_logits.permute(1, 2, 0)[dp_targets['dp_x'], dp_targets['dp_y']],
        torch.LongTensor(dp_targets['dp_I']).to(cls_logits.device))

    return bg_loss, fg_loss


def compute_dp_bg_cls_loss(cls_logits: torch.LongTensor, dp_targets: Dict):
    bg_mask = torch.ByteTensor(create_bg_mask_from_dp_masks(dp_targets['dp_masks'])).to(cls_logits.device)
    bg_logits = cls_logits.permute(1, 2, 0)[bg_mask]

    assert bg_logits.size() == (bg_mask.sum(), 25), f"Wrong bg_logits shape: {bg_logits.size()}"

    return F.cross_entropy(bg_logits, torch.zeros(bg_mask.sum()).long().to(bg_logits.device))


def extract_dp_masks_from_coco_ann(coco_ann, dp_head_output_size: int) -> List[np.array]:
    """
    Returns binary masks for each body part
    """
    masks = [mask_utils.decode(m) if isinstance(m, dict) else np.zeros((256, 256)) for m in coco_ann['dp_masks']]
    masks = [downsample_mask(m, dp_head_output_size) for m in masks]

    return np.array(masks)


def downsample_mask(mask: np.array, output_size: int, threshold: float=0.5) -> np.array:
    assert mask.ndim == 2

    # TODO: check that we return meaningful outputs (i.e. there are no artifacts)
    return resize(mask.astype(np.float), (output_size, output_size)) > threshold


def create_bg_mask_from_dp_masks(dp_masks: np.array):
    assert dp_masks.shape[0] == 14, f"Expected binary mask for each of 14 body parts, got {dp_masks.shape[0]}"
    assert dp_masks.ndim == 3, f"Expected each mask to be binary"

    return (dp_masks.astype(bool).sum(axis=0) - 1).astype(bool)
