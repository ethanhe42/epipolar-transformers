# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math
import numpy as np

from data.transforms.image import transform_preds
from vision.multiview import pix2coord

# find the peack of heatmaps
def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
    # heatmap shape: 42 x 128 x 84
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()
    
    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    # def normalize(x, L):
    #     return -1. + 2. * (x.data + 0.5) / L
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    Iradius = int(radius+0.5)
    affine_parameter = torch.zeros((num_pts, 2, 3), dtype=heatmap.dtype, device=heatmap.device)
    affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
    affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
    affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
    affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    grid_size = torch.Size( [num_pts, 1, Iradius*2+1, Iradius*2+1] )
    grid = F.affine_grid(affine_parameter, grid_size)
    #sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid, mode='bilinear', padding_mode='reflection').squeeze(1)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros').squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, 0)

    X = torch.arange(-radius, radius+0.0001, radius*1.0/Iradius, dtype=heatmap.dtype, device=heatmap.device).view(1, 1, Iradius*2+1)
    Y = torch.arange(-radius, radius+0.0001, radius*1.0/Iradius, dtype=heatmap.dtype, device=heatmap.device).view(1, Iradius*2+1, 1)
    
    sum_region = torch.sum(sub_feature.view(num_pts,-1), 1) + np.finfo(float).eps
    x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h

    x = pix2coord(x, downsample)
    y = pix2coord(y, downsample)
    return torch.stack([x, y],1), score

############################# human pose 3.6m specific #############################

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
