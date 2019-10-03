import sys; sys.path.append('..')
from typing import List

from src.structures.bbox import Bbox
from skimage.io import imread
import cv2


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def load_img(img_id, mode='train'):
    img_path = f"/home/skorokhodov/human-pose/data/coco/{mode}2014/COCO_{mode}2014_{img_id:012d}.jpg"
    
    return imread(img_path)


def get_point_coords(ann, point_i):
    point = [ann['dp_x'][point_i], ann['dp_y'][point_i]]
    point[0] = ann['bbox'][0] + (point[0] / 256) * ann['bbox'][2]
    point[1] = ann['bbox'][1] + (point[1] / 256) * ann['bbox'][3]
    point = [int(p) for p in point]
    
    return point


def add_bbox_to_img(img, bbox:Bbox, color=GREEN, lw=1):
    result = img.copy()
    cv2.rectangle(result, bbox.discretize().corners()[:2], bbox.discretize().corners()[2:], color, lw)

    return result


def get_point_coords(dp_x:List[float], dp_y:List[float], bbox:Bbox, point_i):
    assert len(dp_x) == len(dp_y)
    
    point = [dp_x[point_i], dp_y[point_i]]
    point[0] = bbox.x + (point[0] / 256) * bbox.width
    point[1] = bbox.y + (point[1] / 256) * bbox.height
    point = [int(p) for p in point]
    
    return point


def add_points_to_image(img, dp_I:List[int], dp_x:List[float], dp_y:List[float], bbox:Bbox):
    img = img.copy()
    
    for i in range(len(dp_I)):
        point = get_point_coords(dp_x, dp_y, bbox, i)
        label = dp_I[i]
        
        if label == 24:
            color = (255, 0, 0)
        elif label == 1:
            color = (0, 0, 255)
        elif label == 17:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        
        cv2.rectangle(img, tuple(point), tuple(point), color, 5)
        
    return img
