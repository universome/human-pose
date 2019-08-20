import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import misc as misc_nn_ops

from src.structures.bbox import Bbox
from .image_list import ImageList
from .roi_heads import paste_masks_in_image


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "masks" in target:
            mask = target["masks"]
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target["masks"] = mask

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints

        return image, target

    def batch_images(self, images, size_divisible=32):
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
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
                pred["dp_classes"], pred["dp_u_coords"], pred["dp_v_coords"], pred['dp_masks'] = \
                    postprocess_dp_predictions(pred["dp_cls_logits"], pred["dp_uv_coords"], pred['dp_mask_logits'], boxes)

        return result


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def postprocess_dp_predictions(dp_cls_logits, dp_uv_coords, dp_mask_logits, boxes):
    if len(boxes) == 0:
        dp_classes = torch.empty(0, 10, 10)
        dp_u_coords = torch.empty(0, 10, 10)
        dp_v_coords = torch.empty(0, 10, 10)
        dp_masks = torch.empty(0, 10, 10)

        return dp_classes, dp_u_coords, dp_v_coords, dp_masks

    bboxes = [Bbox.from_torch_tensor(p).discretize() for p in boxes]
    target_sizes = [(bb.height, bb.width) for bb in bboxes]
    dp_cls_logits = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                     for x, s in zip(dp_cls_logits, target_sizes)]
    dp_uv_coords = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                    for x, s in zip(dp_uv_coords, target_sizes)]
    dp_mask_logits = [F.interpolate(x.unsqueeze(0), size=s, mode='bilinear').squeeze(0) \
                      for x, s in zip(dp_mask_logits, target_sizes)]

    results = [get_confident_fg_predictions(l, uv, m) for l, uv, m in zip(dp_cls_logits, dp_uv_coords, dp_mask_logits)]
    dp_classes, dp_u_coords, dp_v_coords, dp_masks = zip(*results)

    return dp_classes, dp_u_coords, dp_v_coords, dp_masks


def get_confident_fg_predictions(dp_cls_logits, dp_uv_coords, dp_mask_logits):
    # TODO: We can try and parallelize the computation by first padding to largest value
    #       then stacking and computing for all. But this does not work because of OOM.
    #       So, we should sort by size and split into batches.

    # Convert to [C x W x H] to [W x H x C] for convenience
    c, w, h = dp_cls_logits.shape

    # Leaving only the most confident body part
    dp_classes = dp_cls_logits.argmax(dim=0, keepdim=True)
    idx = torch.arange(c).view(-1, 1, 1).repeat(1, w, h)
    max_cls_mask = idx.to(dp_classes.device) == dp_classes
    dp_classes = dp_classes.squeeze(0)

    # Extract UV-coords
    dp_u_coords, dp_v_coords = dp_uv_coords[:24, :, :], dp_uv_coords[24:, :, :]

    # Add dummy background predictions so we can use masked_select later (make the same dimensionality as dp_classes)
    bg_dummy_coords = torch.zeros(1, w, h).to(dp_u_coords.device)
    dp_u_coords = torch.cat([bg_dummy_coords, dp_u_coords], dim=0)
    dp_v_coords = torch.cat([bg_dummy_coords, dp_v_coords], dim=0)

    # Select coordinates for the most confident class
    dp_u_coords = dp_u_coords.masked_select(max_cls_mask).view(w, h)
    dp_v_coords = dp_v_coords.masked_select(max_cls_mask).view(w, h)

    # Copmuting dp_mask is easy:
    dp_mask = dp_mask_logits.argmax(dim=0)

    # We should set predictions of background to 0
    dp_classes[dp_mask == 0] = 0
    dp_u_coords[dp_mask == 0] = 0.
    dp_v_coords[dp_mask == 0] = 0.

    return dp_classes, dp_u_coords, dp_v_coords, dp_mask
