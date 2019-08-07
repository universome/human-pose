from typing import List

import numpy as np

from src.structures.bbox import Bbox


def create_target_for_dp(coco_ann: List,
                         proposal: Bbox,
                         model_output_size: int):
    """
    This function takes a raw coco annotation and prepares target, which is then
    passed into the model to compute the loss. It is not a trivial function, because
    we have to account for the following things:
        - we have annotations for ground-truth bbox, but model predictions are made for bbox proposals
        - we can have complex data augmentations (flip, zoom, etc.)

    Arguments:
        - coco_ann — raw coco annotation with the following required fields:
            dp_x, dp_y, dp_I, dp_U, dp_V, dp_masks — these are all densepose annotatoins
        - proposal — box proposals for an image (produced by detection head)
        - model_output_size — what is the output size of the model (usually it is 56x56)

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
    gt_bbox = Bbox.from_coco_ann(coco_ann)

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
    proposal_dp_x = proposal_dp_x / proposal.width * model_output_size
    proposal_dp_y = proposal_dp_y / proposal.height * model_output_size

    return {
        'dp_U': dp_U,
        'dp_V': dp_V,
        'dp_I': dp_I,
        'dp_x': proposal_dp_x,
        'dp_y': proposal_dp_y
    }

