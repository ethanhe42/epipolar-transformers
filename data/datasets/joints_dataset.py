# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com), modified by Yihui He
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.transforms.image import get_affine_transform
from data.transforms.image import affine_transform
from data.transforms.keypoints2d import Heatmapcreator
from vision.multiview import coord2pix
from core import cfg
from .base_dataset import BaseDataset


class JointsDataset(BaseDataset):

    def __init__(self, root, subset, is_train, transform=None):
        self.heatmapcreator = Heatmapcreator(
                cfg.KEYPOINT.HEATMAP_SIZE,
                cfg.KEYPOINT.SIGMA, 
                cfg.BACKBONE.DOWNSAMPLE)
        self.is_train = is_train
        self.subset = subset

        self.root = root
        self.data_format = cfg.DATASETS.DATA_FORMAT
        self.scale_factor = cfg.DATASETS.SCALE_FACTOR
        self.rotation_factor = cfg.DATASETS.ROT_FACTOR
        self.image_size = cfg.DATASETS.IMAGE_SIZE #NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.KEYPOINT.HEATMAP_SIZE #NETWORK.HEATMAP_SIZE
        self.sigma = cfg.KEYPOINT.SIGMA #NETWORK.SIGMA
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.db = []

        self.num_joints = cfg.KEYPOINT.NUM_PTS
        self.union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'thorax', #new
            9: 'neck',
            10: 'upper neck', #new
            11: 'nose',
            12: 'head',
            13: 'head top', #new
            14: 'lsho',
            15: 'lelb',
            16: 'lwri',
            17: 'rsho',
            18: 'relb',
            19: 'rwri'
        }
        #mask
        # np.array([0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 11, 12, 14, 15, 16, 17, 18, 19])
        #self.actual_joints = {
        #    0: 'root',
        #    1: 'rhip',
        #    2: 'rkne',
        #    3: 'rank',
        #    4: 'lhip',
        #    5: 'lkne',
        #    6: 'lank',
        #    7: 'belly',
        #    8: 'neck',
        #    9: 'nose',
        #    10: 'head',
        #    11: 'lsho',
        #    12: 'lelb',
        #    13: 'lwri',
        #    14: 'rsho',
        #    15: 'relb',
        #    16: 'rwri'
        #}
        self.actual_joints = {}
        self.u2a_mapping = {}

        if cfg.DATALOADER.BENCHMARK:
            from utils.timer import Timer
            self.timer = Timer()
            self.timer0 = Timer()

        if cfg.VIS.H36M:
            self.checked = []
    
    # def compute_distorted_meshgrid(self, image, fx, fy, cx, cy, k, p):
    #     h, w = image.shape[:2]
    #     print('h ', h, 'w', w, 'cx', cx, 'cy', cy, 'fx', fx, 'fy', fy, 'p', p, 'k',k)
    #     grid_x = (np.arange(w, dtype=np.float32) - cx) / fx
    #     grid_y = (np.arange(h, dtype=np.float32) - cy) / fy
    #     meshgrid = np.stack(np.meshgrid(grid_x, grid_y), axis=2).reshape(-1, 2)
    #     r2 = meshgrid[:, 0] ** 2 + meshgrid[:, 1] ** 2
    #     radial = meshgrid * (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3).reshape(-1, 1)
    #     tangential_1 = p.reshape(1, 2) * np.broadcast_to(meshgrid[:, 0:1] * meshgrid[:, 1:2], (len(meshgrid), 2))
    #     tangential_2 = p[::-1].reshape(1, 2) * (meshgrid**2 + np.broadcast_to(r2.reshape(-1, 1), (len(meshgrid), 2)))

    #     meshgrid = radial + tangential_1 + tangential_2

    #     # move back to screen coordinates
    #     meshgrid *= np.array([fx, fy]).reshape(1, 2)
    #     meshgrid += np.array([cx, cy]).reshape(1, 2)

    #     # cache (save) distortion maps
    #     meshgrid_int16 = cv2.convertMaps(meshgrid.reshape((h, w, 2)), None, cv2.CV_16SC2)
    #     image_undistorted = cv2.remap(image, *meshgrid_int16, cv2.INTER_CUBIC)
    #     #meshgrid_int16 = meshgrid.reshape(h, w, 2)
    #     #image_undistorted = cv2.remap(image, meshgrid_int16, cv2.INTER_CUBIC)
    #     return image_undistorted

    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def do_mapping(self):
        mapping = self.u2a_mapping
        for item in self.db:
            joints = item['joints_2d']
            joints_vis = item['joints_vis']

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 2))
            joints_union_vis = np.zeros(shape=(njoints, 3))

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_2d'] = joints_union
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, preds, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def isdamaged(self, db_rec):
        #damaged seq
        #'Greeting-2', 'SittingDown-2', 'Waiting-1'
        if db_rec['subject'] == 9:
            if db_rec['action'] != 5 or db_rec['subaction'] != 2:
                if db_rec['action'] != 10 or db_rec['subaction'] != 2:
                    if db_rec['action'] != 13 or db_rec['subaction'] != 1:
                        return False
        else:
            return False
        return True

    def __getitem__(self, idx):
        if cfg.DATALOADER.BENCHMARK: self.timer0.tic()
        db_rec = copy.deepcopy(self.db[idx])
        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            if cfg.VIS.H36M:
                #seq = (db_rec['subject'], db_rec['action'], db_rec['subaction'])
                #if not seq in self.checked:
                #    print(seq)
                #    print(self.isdamaged(db_rec))
                #    self.checked.append(seq)
                #else:
                #    return np.ones(2)
                    
                print(db_rec['image'])
            # print(db_rec['image'])

            if self.data_format == 'undistoredzip':
                image_dir = 'undistoredimages.zip@'
            elif self.data_format == 'zip':
                image_dir = 'images.zip@'
            else:
                image_dir = ''
            image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                                db_rec['image'])
            if 'zip' in self.data_format:
                from utils import zipreader
                data_numpy = zipreader.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            else:
                data_numpy = cv2.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # crop image from 1002 x 1000 to 1000 x 1000
            data_numpy = data_numpy[:1000]
            assert data_numpy.shape == (1000, 1000, 3), data_numpy.shape

        joints = db_rec['joints_2d'].copy()
        joints_3d = db_rec['joints_3d'].copy()
        joints_3d_camera = db_rec['joints_3d_camera'].copy()

        joints_3d_camera_normed = joints_3d_camera - joints_3d_camera[0]
        keypoint_scale = np.linalg.norm(joints_3d_camera_normed[8] - joints_3d_camera_normed[0])
        joints_3d_camera_normed /= keypoint_scale

        if cfg.DATALOADER.BENCHMARK: 
            assert joints.shape[0] == cfg.KEYPOINT.NUM_PTS, joints.shape[0]
            #assert db_rec['joints_3d'].shape[0] == cfg.KEYPOINT.NUM_PTS,db_rec['joints_3d'].shape[0] 
        center = np.array(db_rec['center']).copy()
        joints_vis = db_rec['joints_vis'].copy()
        scale = np.array(db_rec['scale']).copy()
        #undistort
        camera = db_rec['camera']
        R = camera['R'].copy()
        rotation = 0
        K = np.array([
            [float(camera['fx']), 0, float(camera['cx'])], 
            [0, float(camera['fy']), float(camera['cy'])], 
            [0, 0, 1.], 
            ])
        T = camera['T'].copy()
        world3d = (R.T @ joints_3d_camera.T  + T).T
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()
        # Rt[:, :3] = R.T
        # Rt[:, 3] = T.squeeze()
        
        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            if cfg.VIS.H36M:
                if not np.isclose(world3d, joints_3d).all():
                    print('world3d difference')
                    print(world3d)
                    print('joints_3d')
                    print(joints_3d)
                from IPython import embed
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                fig = plt.figure(1)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)            
                ax3 = fig.add_subplot(233)  
                ax4 = fig.add_subplot(234)  
                ax5 = fig.add_subplot(235)  
                ax6 = fig.add_subplot(236)  
                ax1.imshow(data_numpy[..., ::-1])
                ax1.set_title('raw')

        #0.058 s
        distCoeffs = np.array([float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        
        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            if self.data_format != 'undistoredzip':
                data_numpy = cv2.undistort(data_numpy, K, distCoeffs)

        #0.30 s
        if cfg.DATALOADER.BENCHMARK: print('timer0', self.timer0.toc())
        if cfg.DATALOADER.BENCHMARK: self.timer.tic()

        if cfg.VIS.H36M:
            ax1.scatter(joints[:, 0], joints[:, 1], color='green')
            imagePoints, _ = cv2.projectPoints(joints_3d[:, None, :], (0,0,0), (0,0,0), K, distCoeffs)
            imagePoints = imagePoints.squeeze()
            ax1.scatter(imagePoints[:, 0], imagePoints[:, 1], color='yellow')
            from vision.multiview import project_point_radial
            camera = db_rec['camera']
            f = (K[0, 0] + K[1, 1])/2.
            c = K[:2, 2].reshape((2, 1))
            iccv19Points = project_point_radial(joints_3d_camera, f, c, camera['k'], camera['p'])
            ax1.scatter(iccv19Points[:, 0], iccv19Points[:, 1], color='blue')
            # trans1 = get_affine_transform(center, scale, rotation, self.image_size)
            # box1 = affine_transform(np.array([[0, 0], [999, 999]]), trans1)
            # print(box1)
            # rect1 = patches.Rectangle(box1[0],box1[1][0] - box1[0][0],box1[1][1] - box1[0][1],linewidth=1,edgecolor='r',facecolor='none')
            # ax1.add_patch(rect1)
            # print(joints, joints.shape, center.shape)
        joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
        center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()
        #data_numpy  = self.compute_distorted_meshgrid(data_numpy , 
        #        float(camera['fx']), 
        #        float(camera['fy']), 
        #        float(camera['cx']), 
        #        float(camera['cy']), 
        #        np.array([float(i) for i in camera['k']]),
        #        np.array([float(i) for i in camera['p']]))
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

 

        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            if cfg.VIS.H36M:
                # print(joints.shape, center.shape)
                # print(trans)
                ax2.imshow(data_numpy[..., ::-1])
                projected2d = K.dot(joints_3d_camera.T)
                projected2d[:2] = projected2d[:2] / projected2d[-1]            
                ax1.scatter(projected2d[0], projected2d[1], color='red')
                ax2.scatter(joints[:, 0], joints[:, 1], color='green')              
                ax2.scatter(projected2d[0], projected2d[1], color='red')
                # box1 = affine_transform(np.array([[0, 0], [999, 999]]), trans)
                # rect1 = patches.Rectangle(box1[0],box1[1][0] - box1[0][0],box1[1][1] - box1[0][1],linewidth=1,edgecolor='r',facecolor='none')
                # ax2.add_patch(rect1)
                ax2.set_title('undistort')

        #input = data_numpy
        trans = get_affine_transform(center, scale, rotation, self.image_size)
        cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)
        KRT = cropK.dot(Rt)

        

        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            input = cv2.warpAffine(
                data_numpy,
                trans, (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

        # 0.31 s


        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                if (np.min(joints[i, :2]) < 0 or
                        joints[i, 0] >= self.image_size[0] or
                        joints[i, 1] >= self.image_size[1]):
                    joints_vis[i, :] = 0

        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            if cfg.VIS.H36M:
                ax3.imshow(input[..., ::-1])
                # ax3.scatter(joints[:, 0], joints[:, 1])
                # projected2d = KRT.dot(np.concatenate((db_rec['joints_3d'], np.ones( (len(db_rec['joints_3d']), 1))), 1).T)
                ax3.scatter(joints[:, 0], joints[:, 1])
                ax3.set_title('cropped')
                ax4.imshow(input[..., ::-1])
                # ax4.scatter(joints[:, 0], joints[:, 1])
                # projected2d = KRT.dot(np.concatenate((db_rec['joints_3d'], np.ones( (len(db_rec['joints_3d']), 1))), 1).T)
                projected2d = cropK.dot(joints_3d_camera.T)
                projected2d[:2] = projected2d[:2] / projected2d[-1]
                #ax4.scatter(joints[:, 0], joints[:, 1], color='green')            
                #ax4.scatter(projected2d[0], projected2d[1], color='red')
                ax4.scatter(joints[-2:, 0], joints[-2:, 1], color='green')            
                ax4.scatter(projected2d[0, -2:], projected2d[1, -2:], color='red')
                ax4.set_title('cropped, project 3d to 2d')

            if self.transform:
                input = self.transform(input)
        
        target = self.heatmapcreator.get(joints)
        target = target.reshape((-1, target.shape[1], target.shape[2]))
        target_weight = joints_vis[:, 0, None]
        ## inaccurate heatmap
        #target, target_weight = self.generate_target(joints, joints_vis)
        # target = torch.from_numpy(target).float()
        # target_weight = torch.from_numpy(target_weight)

        if cfg.VIS.H36M:
            #ax5.imshow(target.max(0)[0])
            #ax5.scatter(coord2pix(joints[:, 0], 4), coord2pix(joints[:, 1], 4), color='green')
            from modeling.backbones.basic_batch import find_tensor_peak_batch
            # pred_joints, _ = find_tensor_peak_batch(target, self.sigma, cfg.BACKBONE.DOWNSAMPLE)
            # ax5.scatter(coord2pix(pred_joints[:, 0], 4), coord2pix(pred_joints[:, 1], 4), color='blue')            
            # ax6.scatter(coord2pix(pred_joints[:, 0], 4), coord2pix(pred_joints[:, 1], 4), color='blue')            

            heatmap_by_creator = self.heatmapcreator.get(joints)
            heatmap_by_creator = heatmap_by_creator.reshape((-1, heatmap_by_creator.shape[1], heatmap_by_creator.shape[2]))
            ax6.imshow(heatmap_by_creator.max(0))
            ax6.scatter(coord2pix(joints[:, 0], 4), coord2pix(joints[:, 1], 4), color='green')            
            # pred_joints, _ = find_tensor_peak_batch(torch.from_numpy(heatmap_by_creator).float(), self.sigma, cfg.BACKBONE.DOWNSAMPLE)
            # print('creator found', pred_joints)
            # ax5.scatter(coord2pix(pred_joints[:, 0], 4), coord2pix(pred_joints[:, 1], 4), color='red')            
            # ax6.scatter(coord2pix(pred_joints[:, 0], 4), coord2pix(pred_joints[:, 1], 4), color='red')            
            plt.show()
        ret = {
            'heatmap': target,
            'visibility':target_weight,
            'KRT': KRT,
            'points-2d': joints,
            'points-3d': world3d.astype(np.double) if 'lifting' not in cfg.DATASETS.TASK else world3d, 
            'camera-points-3d': joints_3d_camera, 
            'normed-points-3d': joints_3d_camera_normed, 
            'scale': keypoint_scale,
            'action' : torch.tensor([db_rec['action']]),
            'img-path': db_rec['image'],
        }
        if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'lifting_rot']:
            ret['img'] = input
        ret['K'] = cropK
        ret['RT'] = Rt
        if cfg.VIS.MULTIVIEWH36M:
            ret['T'] = T
            ret['R'] = R
            ret['original_image'] = data_numpy
        if cfg.KEYPOINT.TRIANGULATION == 'rpsm' and not self.is_train:
            ret['origK'] = K
            ret['crop_center'] = center
            ret['crop_scale'] = scale

        if cfg.DATALOADER.BENCHMARK: print('timer1', self.timer.toc())
        return ret
        #         meta = {
        #     'scale': scale,
        #     'center': center,
        #     'rotation': rotation,
        #     'joints_2d': db_rec['joints_2d'],
        #     'joints_2d_transformed': joints,
        #     'joints_vis': joints_vis,
        #     'source': db_rec['source']
        # }
        #return input, target, target_weight, meta

    def generate_target(self, joints_3d, joints_vis):
        target, weight = self.generate_heatmap(joints_3d, joints_vis)
        return target, weight

    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = np.zeros(2)
            feat_stride[0] = self.image_size[0] / self.heatmap_size[0]
            feat_stride[1] = self.image_size[1] / self.heatmap_size[1]
            mu_x = int(coord2pix(joints[joint_id][0], feat_stride[0]) + 0.5)
            mu_y = int(coord2pix(joints[joint_id][1], feat_stride[1]) + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
