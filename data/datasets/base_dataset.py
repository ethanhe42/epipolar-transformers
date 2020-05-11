import pickle
import os
from scipy.stats import truncnorm
import struct
import numpy as np
from tqdm import tqdm
import copy
import random

import torch
import torchvision
from torchvision import transforms
from PIL import ImageFile
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from core import cfg
from data.transforms.build import *
from data.transforms.keypoints2d import *
from data.transforms.keypoints3d import *
from data.transforms.image import RGB, is_grey_scale, quantized_color_preprocess, cv2_loader, dropout2d

from utils.file_utils import *
import matplotlib.pyplot as plt

class BaseDataset(Dataset):
    def post__init__(self):
        """
        """
        self.outputsize = cfg.KEYPOINT.HEATMAP_SIZE

        self.unit = 1

        self.visible_anno = [
                'hand-side',
                'points-2d'        ,
                'points-3d'        ,
                'can-points-3d'    ,
                'normed-points-3d' ,
                'rotation'         ,
                'scale'            ,
                'visibility'       ,
                'R',
                'RT',
                'K',
                'KRT',
                'other_img',
                'other_KRT',
                'other_heatmap',
                ]
        if cfg.DATASETS.TASK in ['keypoint', 'keypoint_lifting_rot', 'multiview_keypoint']:
            if cfg.DATASETS.CROP_AFTER_RESIZE:
                from data.transforms.image import Crop
                self.transform = transforms.Compose([
                    transforms.Resize(cfg.DATASETS.IMAGE_SIZE),
                    Crop(0, 0, cfg.DATASETS.CROP_SIZE[0], cfg.DATASETS.CROP_SIZE[1]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(cfg.DATASETS.IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
            self.heatmapcreator = Heatmapcreator(
                    self.outputsize,
                    cfg.KEYPOINT.SIGMA, 
                    cfg.BACKBONE.DOWNSAMPLE)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((cfg.LIFTING.CROP_SIZE, cfg.LIFTING.CROP_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            self.heatmapcreator = Heatmapcreator(
                    cfg.KEYPOINT.HEATMAP_SIZE, 
                    cfg.KEYPOINT.SIGMA, 
                    cfg.BACKBONE.DOWNSAMPLE)
        
    def post__getitem__(self, ret):
        """
        """
        ret['unit'] = self.unit
        return ret

