from typing import List, Dict, Callable

import numpy as np
import torch
import torchvision.transforms.functional as F
import albumentations as A

from src.utils.dp_targets import combine_dp_masks


def construct_transforms(train:bool=True) -> Callable:
    if train:
        augmentations = [
            ConvertBatchTarget(),
            AlbuTransformWrapper(
                A.Compose([
                    HorizontalFlipWithDP(p=0.5)
                ], bbox_params={
                    'format': 'pascal_voc',
                    'min_area': 0.,
                    'min_visibility': 0.,
                    'label_fields': ['labels']
                })
            ),
            ToTensors(),
        ]
    else:
        augmentations = [
            ConvertBatchTarget(),
            ToTensors(),
        ]

    return Compose(augmentations)


class HorizontalFlipWithDP(A.HorizontalFlip):
    @property
    def targets(self):
        return {'image': self.apply,
                'mask': self.apply_to_mask,
                'masks': self.apply_to_masks,
                'bboxes': self.apply_to_bboxes,
                'keypoints': self.apply_to_keypoints,
                'dp_anns': horizontal_flip_dp_anns}



class ConvertBatchTarget:
    def __call__(self, image, target):
        return image, torchvision_target_format(target)


class ToTensors(object):
    def __call__(self, image, target):
        return F.to_tensor(image), {
            'image_id': target['image_id'],
            'labels': torch.Tensor(target['labels']).long(),
            'boxes': torch.Tensor(target['boxes']),
            'masks': torch.Tensor(target['masks']),
            'dp_anns': target['dp_anns']
        }


# class ToNumpyArray(object):
#     def __call__(self, image, target):
#         return np.array(image), target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class AlbuTransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        albu_input = {
            'image': np.array(image),
            'labels': target['labels'],
            'bboxes': target['boxes'],
            'masks': target['masks'],
            'dp_anns': target['dp_anns']
        }

        albu_result = self.transform(**albu_input)

        return albu_result['image'], {
            'image_id': target['image_id'],
            'labels': albu_result['labels'],
            'boxes': albu_result['bboxes'],
            'masks': albu_result['masks'],
            'dp_anns': albu_result['dp_anns'],
        }


def horizontal_flip_dp_anns(dp_anns:List[Dict], **kwargs) -> List[Dict]:
    return [({
        'dp_I': obj['dp_I'],
        'dp_U': obj['dp_U'],
        'dp_V': obj['dp_V'],
        'dp_x': [(256 - x) for x in obj['dp_x']],
        'dp_y': obj['dp_y'],
        'dp_mask': obj['dp_mask'][:, ::-1],
    } if not obj is None else None) for obj in dp_anns]


def torchvision_target_format(target):
    return {
        'image_id': target[0]['image_id'],
        'labels': [obj['category_id'] for obj in target],
        'boxes': [xywh_to_xyxy(obj['bbox']) for obj in target],
        'masks': [obj['mask'] for obj in target],
        # TODO: maybe we should have a special class for dp_ann?
        'dp_anns': [({
            'dp_I': obj['dp_I'],
            'dp_U': obj['dp_U'],
            'dp_V': obj['dp_V'],
            'dp_x': obj['dp_x'],
            'dp_y': obj['dp_y'],
            'dp_mask': combine_dp_masks(obj['dp_masks']),
        } if 'dp_masks' in obj else None) for obj in target]
    }


def xywh_to_xyxy(bbox) -> List:
    # Converts bbox from coco format (x, y, width, height)
    # to "x1, y1, x2, y2" format (pascal voc?)
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
