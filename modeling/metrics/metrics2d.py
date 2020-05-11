# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, time, warnings, numpy as np
import numbers
from sklearn.metrics import auc

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbones.basic_batch import get_max_preds
from core import cfg

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        if not cfg.KEYPOINT.LOSS_PER_JOINT:
            loss /= num_joints
        return loss # * batch_size

class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))        
        dimension = heatmaps_pred.shape[-1]
        diff = (heatmaps_gt - heatmaps_pred) ** 2 * target_weight[..., None]
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(target_weight)))
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        reductions = ['none', 'mean', 'sum', 'batch']
        assert reduction in reductions, 'Invalid {:}, not in {:}'.format(reduction, reductions)
        self.reduction = reduction
        if self.reduction == 'batch':
            self.reduction = 'sum'
            self.divide_by_batch = True
        else:
            self.divide_by_batch = False

    def forward(self, inputs, targets, masks=None):
        batch_size = inputs.size(0)
        if masks is not None:
            inputs  = torch.masked_select(inputs, masks)
            targets = torch.masked_select(targets, masks)
        mse_loss = F.mse_loss(inputs, targets, reduction=self.reduction)
        if self.divide_by_batch:
            mse_loss = mse_loss / batch_size
        return mse_loss

def compute_stage_loss(criterion, targets, outputs, masks=None):
    assert isinstance(outputs, list), 'The outputs type is wrong : {:}'.format(type(outputs))
    total_loss, each_stage_loss = 0, []
    for output in outputs:
        stage_loss = criterion(output, targets, masks)
        total_loss = total_loss + stage_loss
        each_stage_loss.append(stage_loss)
    return total_loss, each_stage_loss

def compute_PCKh(PCK_head):
    PCK_head = [np.array(x) for x in PCK_head]
    distances = np.concatenate((PCK_head[0], PCK_head[5]))
    print_log('PCKh - ankle    : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate((PCK_head[1], PCK_head[4]))
    print_log('PCKh - knee     : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate((PCK_head[2], PCK_head[3]))
    print_log('PCKh - hip      : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate((PCK_head[6], PCK_head[11]))
    print_log('PCKh - wrist    : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate((PCK_head[7], PCK_head[10]))
    print_log('PCKh - elbow    : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate((PCK_head[8], PCK_head[9]))
    print_log('PCKh - shoulder : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate((PCK_head[12], PCK_head[13]))
    print_log('PCKh - head     : {:6.3f} %'.format( (distances<0.5).mean() * 100 ))
    distances = np.concatenate(PCK_head)
    print_log('PCKh : {:6.3} %'.format( (distances<0.5).mean() * 100 ))

def AUCat(max_threshold, error_per_image):
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 100.0 / error_per_image.size
    return auc(threshold, accuracys) / max_threshold

def evaluate_normalized_mean_error(predictions, groundtruth, visibility, thresholds):
    """compute total average normlized mean error
    Args:
        predictions: N x 2 x Joints
        groundtruth: N x 2 x Joints
    """
    assert len(predictions) == len(groundtruth), 'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format( len(predictions), len(groundtruth) )
    assert len(predictions) > 0, 'The length of predictions must be greater than 0 vs {}'.format( len(predictions) )
    assert predictions.shape[1] == 2, predictions.shape
    assert groundtruth.shape[1] == 2, groundtruth.shape

    num_images = len(predictions)
    for i in range(num_images):
        c, g = predictions[i], groundtruth[i]
        assert isinstance(c, np.ndarray) and isinstance(g, np.ndarray), 'The type of predictions is not right : [{:}] :: {} vs {} '.format(i, type(c), type(g))

    num_points = predictions[0].shape[1]
    error_per_image = np.zeros((num_images,1))
    # if num_points == 16: 
    #     PCK_head = [ [] for i in range(16) ]
    # else: 
    #     PCK_head = None

    joint_err = [ [] for i in range(num_points)]
    joints_err = [] # overall err distance

    for i in range(num_images):
        detected_points = predictions[i]
        ground_truth_points = groundtruth[i]
        interocular_distance = 1
        dis_sum, pts_sum = 0, 0
        for j in range(num_points):
            if visibility[i,j]:
                distance = np.linalg.norm(detected_points[:1, j] - ground_truth_points[:1, j])
                dis_sum, pts_sum = dis_sum + distance, pts_sum + 1
                # if PCK_head is not None: # calculate PCKh
                    # PCK_head[j].append(distance / headsize)
                joint_err[j].append(distance)
                joints_err.append(distance)
        error_per_image[i] = dis_sum / (pts_sum*interocular_distance)
        # ith image's err distance average over joints

    normalise_mean_error = error_per_image.mean()
    # # calculate the auc for 0.07
    # area_under_curve07 = AUCat(0.07, error_per_image)
    # # calculate the auc for 0.08
    # area_under_curve08 = AUCat(0.08, error_per_image)
    
    # accuracy_under_007 = np.sum(error_per_image<0.07) * 100. / error_per_image.size
    # accuracy_under_008 = np.sum(error_per_image<0.08) * 100. / error_per_image.size

    # if PCK_head is not None: 
    #     compute_PCKh(PCK_head)

    # TODO PCKh?

    PCKs = {}
    PCK_joint = {}
    for th in thresholds:
        PCKs['PCK@'+str(th)] = sum(d < th for d in joints_err) * 100.0 / len(joints_err)
        # correct over total joints detected
        # for j in range(num_points):
        #     if len(joint_err[j]) != 0:
        #         PCK_joint['PCK@'+str(th)+'_joint_'+str(j)] = sum(d < th for d in joint_err[j]) * 100.0 / len(joint_err[j]) 


    acc_curve = np.zeros((100))
    acc_total = np.zeros((100))
    # calculate auc
    max_threshold = thresholds[-1]
    threshold = np.linspace(0, max_threshold, num=100)
    acc_curve = np.zeros(threshold.shape)
    for i in range(threshold.size):
        acc_total[i] = np.sum(joints_err < threshold[i]) * 1.0
        acc_curve[i] = acc_total[i] / len(joints_err)
        
    pck_auc = auc(threshold, acc_curve) / max_threshold

    return normalise_mean_error, PCKs, acc_total, len(joints_err) #, pck_auc#, acc_curve #, PCK_joint #, for_pck_curve


def calculate_err(predictions, groundtruth, visibility, thresholds, max_threshold):
    """compute error for each threshold in each image
    Args:
        predictions: N x 2 x Joints
        groundtruth: N x 2 x Joints
    """
    assert len(predictions) == len(groundtruth), 'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format( len(predictions), len(groundtruth) )
    assert len(predictions) > 0, 'The length of predictions must be greater than 0 vs {}'.format( len(predictions) )
    assert predictions.shape[1] == 2, predictions.shape
    assert groundtruth.shape[1] == 2, groundtruth.shape

    num_images = len(predictions)
    err_joints = np.zeros((num_images, int(max_threshold)))
    total_joints = np.zeros((num_images, 1))
    num_points = predictions[0].shape[1]

    threshold = np.linspace(0, max_threshold, num=int(max_threshold))
    joints_err_batch = []
    PCKs = {}
    for i in range(num_images):
        detected_points = predictions[i]
        ground_truth_points = groundtruth[i]
        joints_err = []
        for j in range(num_points):
            if visibility[i,j]:
                distance = np.linalg.norm(detected_points[:1, j] - ground_truth_points[:1, j])
                joints_err.append(distance)
                joints_err_batch.append(distance)
        
        for j in range(threshold.size):
            err_joints[i][j] = np.sum(joints_err < threshold[j]) * 1.0
            total_joints[i] = len(joints_err)

    for th in thresholds:
        PCKs['PCK@'+str(th)] = sum(d < th for d in joints_err_batch) * 100.0 / len(joints_err_batch)
        
    return PCKs, err_joints, total_joints


def calc_pck(predictions, groundtruth, visibility, thresholds):
    """compute error for each threshold in each image
    Args:
        predictions: N x 2 x Joints
        groundtruth: N x 2 x Joints
    """
    assert len(predictions) == len(groundtruth), 'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format( len(predictions), len(groundtruth) )
    assert len(predictions) > 0, 'The length of predictions must be greater than 0 vs {}'.format( len(predictions) )
    assert predictions.shape[1] == 2, predictions.shape
    assert groundtruth.shape[1] == 2, groundtruth.shape

    num_images = len(predictions)
    num_points = predictions[0].shape[1]

    joints_err_batch = []
    PCKs = {}
    for i in range(num_images):
        detected_points = predictions[i]
        ground_truth_points = groundtruth[i]
        for j in range(num_points):
            if visibility[i,j]:
                distance = np.linalg.norm(detected_points[:1, j] - ground_truth_points[:1, j])
                joints_err_batch.append(distance)

    for th in thresholds:
        PCKs['PCK@'+str(th)] = sum(d < th for d in joints_err_batch) * 100.0 / len(joints_err_batch)
        
    return PCKs

############################# human pose 3.6m specific #############################

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def JDR(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred