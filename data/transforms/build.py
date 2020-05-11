import random

import torch
import torchvision
from torchvision.transforms import functional as F

from .keypoints3d import *


#class Compose(object):
#    def __init__(self, transforms):
#        self.transforms = transforms
#
#    def __call__(self, image, target):
#        for t in self.transforms:
#            image, target = t(image, target)
#        return image, target
#
#    def __repr__(self):
#        format_string = self.__class__.__name__ + "("
#        for t in self.transforms:
#            format_string += "\n"
#            format_string += "    {0}".format(t)
#        format_string += "\n)"
#        return format_string
#
#class ToTensor(object):
#    def __call__(self, image, target):
#        return F.to_tensor(image), target


class LiftingTrans(object):
    def __init__(self, is_train=True):
        self.is_train = is_train
    def __call__(self, inputs, targets):
        "inputs are 2d keypoints 42x2 and 3d keypoints 42x3"
        palm_coord()

def totensor(arr):
    if torch.is_tensor(arr):
        return arr
    if isinstance(arr, str):
        return arr
    if not isinstance(arr, (np.ndarray)):
        arr = np.array([arr])
    if arr.dtype is np.bool or arr.dtype is np.bool_:
        arr = arr.astype(np.int)
        return torch.tensor(arr, dtype=torch.long)
    return torch.tensor(arr, dtype=torch.float)

def build_transforms(cfg, is_train=True):
    if cfg.DATSETS.TASK == 'lifting':
        transform = Compose(
            [
                LiftingTrans(is_train),
                ToTensor(),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
            ]
        )

    return transform

