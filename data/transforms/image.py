from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
import cv2
import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms

from core import cfg

def is_grey_scale(img_path):
    a = Image.open(img_path)
    
    if len(a.size) == 2:
        ret = True
    ret = False
    a.close()
    return ret

#class Resize(object):
#    def __init__(self, min_size, max_size):
#        if not isinstance(min_size, (list, tuple)):
#            min_size = (min_size,)
#        self.min_size = min_size
#        self.max_size = max_size
#
#    # modified from torchvision to add support for max size
#    def get_size(self, image_size):
#        w, h = image_size
#        size = random.choice(self.min_size)
#        max_size = self.max_size
#        if max_size is not None:
#            min_original_size = float(min((w, h)))
#            max_original_size = float(max((w, h)))
#            if max_original_size / min_original_size * size > max_size:
#                size = int(round(max_size * min_original_size / max_original_size))
#
#        if (w <= h and w == size) or (h <= w and h == size):
#            return (h, w)
#
#        if w < h:
#            ow = size
#            oh = int(size * h / w)
#        else:
#            oh = size
#            ow = int(size * w / h)
#
#        return (oh, ow)
#
#    def __call__(self, image, target):
#        size = self.get_size(image.size)
#        image = F.resize(image, size)
#        target = target.resize(image.size)
#        return image, target
#
#
#class RandomHorizontalFlip(object):
#    def __init__(self, prob=0.5):
#        self.prob = prob
#
#    def __call__(self, image, target):
#        if random.random() < self.prob:
#            image = F.hflip(image)
#            target = target.transpose(0)
#        return image, target
#
#class Normalize(object):
#    def __init__(self, mean, std, to_bgr255=True):
#        self.mean = mean
#        self.std = std
#        self.to_bgr255 = to_bgr255
#
#    def __call__(self, image, target):
#        if self.to_bgr255:
#            image = image[[2, 1, 0]] * 255
#        image = F.normalize(image, mean=self.mean, std=self.std)
#        return image, target

class Crop(object):
   def __init__(self, top, left, height, width):
       self.top = top
       self.left = left
       self.height = height
       self.width = width

   def __call__(self, image):
       image = F.crop(image, self.top, self.left, self.height, self.width)
       return image

class RGB(object):
    """Convert image to RGB.
    Returns:
        PIL RGB Image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
        Returns:
            PIL Image: Randomly grayscaled image.
        """
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

def default_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def cv2_loader(image, size):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = np.float32(image) / 255.0
    # opencv use w h instead
    image = cv2.resize(image, size[::-1])
    return image

def BGR2Lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def quantized_color_preprocess(image, centroids, size):
    image = cv2_loader(image, size)
    h, w, c = image.shape
    image = BGR2Lab(image)
    ab = image[:,:,1:]
    a = np.argmin(np.linalg.norm(centroids[None, :, :] - ab.reshape([-1,2])[:, None, :], axis=2), axis=1)
    # 256 256  quantized color (4bit)
    quantized_ab = a.reshape([h, w])
    return quantized_ab
    # return transforms.ToTensor()(quantized_ab)

def one_hot(labels, C):
    labels = labels.unsqueeze(1)
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target

def dropout2d(arr, drop_ch_num=None, drop_ch_ind=None, p=0.3): 
    if np.random.random() < p:
        return arr, 0, 0
    if drop_ch_num == 0:
        return arr, None, None
    if drop_ch_num is None:
        assert drop_ch_ind is None
        drop_ch_num = int(np.random.choice(np.arange(1, 2 + 1), 1))
        drop_ch_ind = np.random.choice(np.arange(3), drop_ch_num, replace=False)
    # for a in arr:
    assert arr.shape[0] == 3
    for dropout_ch in drop_ch_ind:
        arr[dropout_ch] = 0
    arr *= (3 / (3 - drop_ch_num))

    return arr, drop_ch_num, drop_ch_ind


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com), modified by Yihui He
# ------------------------------------------------------------------------------

import numpy as np
import cv2

import torch


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def affine_transform_pts(pts, t):
    xyz = np.add(
        np.array([[1, 0], [0, 1], [0, 0]]).dot(pts.T), np.array([[0], [0],
                                                                 [1]]))
    return np.dot(t, xyz).T


def affine_transform_pts_cuda(pts, t):
    npts = pts.shape[0]
    pts_homo = torch.cat([pts, torch.ones(npts, 1, device=pts.device)], dim=1)
    out = torch.mm(t, torch.t(pts_homo))
    return torch.t(out[:2, :])


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img,
        trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR)

    return dst_img

def de_transform(img):
    img[..., 0, :, :] = img[..., 0, :, :] * 0.229 + 0.485
    img[..., 1, :, :] = img[..., 1, :, :] * 0.224 + 0.456
    img[..., 2, :, :] = img[..., 2, :, :] * 0.225 + 0.406
    return img