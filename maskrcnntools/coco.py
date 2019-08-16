import os
from typing import List
from copy import deepcopy

import torch
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CocoDetection(VisionDataset):
    """
        It's similar to CocoDetection dataset, but filters out images without objects
    """
    def __init__(self, root, annFile,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 should_ignore_images_without_objects=True,
                 should_add_masks=True):

        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)

        if should_ignore_images_without_objects:
            self.ids = self.get_valid_img_ids(self.coco)
        else:
            self.ids = self.coco.imgs.keys()

        self.ids = list(sorted(self.ids))
        self.should_add_masks = should_add_masks

    def get_valid_img_ids(self, coco) -> List[int]:
        anns = [coco.loadAnns(coco.getAnnIds(imgIds=img_id)) for img_id in coco.imgs.keys()]
        valid_idx = [targets[0]['image_id'] for targets in anns if len(targets) > 0]

        return valid_idx


    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        # To prevent memory leaks when extending annotations
        target = deepcopy(target)

        if self.should_add_masks:
            for obj in target:
                obj['mask'] = self.coco.annToMask(obj)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    
class CocoDataset(object):
    def __init__(self, torch_dataset, transforms=None):
        self.torch_dataset = torch_dataset
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img, target = self.torch_dataset[idx]
        
        masks = annsToSeg(target, self.torch_dataset.coco)[3]
        masks = masks.astype(bool)
        obj_ids = [x['category_id'] for x in target]
        
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = [x['category_id'] for x in target]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = [x['iscrowd'] for x in target]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.torch_dataset)