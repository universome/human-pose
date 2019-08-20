from typing import Tuple


# TODO: be compatible between two modes (xyxy and xywh)
#       and switch to xyxy when possible


class Bbox:
    def __init__(self, *args, format='xywh'):
        if format == 'xywh':
            x, y, w, h = args
            self.x = x
            self.y = y
            self.width = w
            self.height = h
        elif format == 'xyxy':
            x1, y1, x2, y2 = args
            self.x = x1
            self.y = y1
            self.width = x2 - x1
            self.height = y2 - y1
        else:
            raise NotImplementedError

    @classmethod
    def from_coco_ann(cls, coco_ann):
        return Bbox(*coco_ann['bbox'])

    @classmethod
    def from_torch_tensor(cls, bbox):
        x_min, y_min, x_max, y_max = bbox.cpu().data.numpy().tolist()

        assert x_max >= x_min
        assert y_max >= y_min

        return Bbox(x_min, y_min, x_max - x_min, y_max - y_min)

    @property
    def x1(self):
        return self.x

    @property
    def y1(self):
        return self.y

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height

    def discretize(self) -> "Bbox":
        # TODO: it feels like using `round` instead of `int` is better, because it is not that rough
        #  but since densepose_cocoeval uses `int` under the hood, we can't use round (without hacks)
        return Bbox(*map(int, [self.x, self.y, self.width, self.height]))

    def corners(self) -> Tuple:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    # def adjust(self, dc_x, dc_y, log_dw, log_dh) -> "Bbox":
    #     c_x = self.c_x + self.w * dc_x
    #     c_y = self.c_y + self.h * dc_y
    #     w = self.w * log_dw.exp()
    #     h = self.h * log_dh.exp()
    #
    #     return Bbox(c_x, c_y, w, h)