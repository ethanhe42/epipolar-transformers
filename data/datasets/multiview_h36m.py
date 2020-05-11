# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com), modified by Yihui He
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import random
import pickle
import collections
import numpy as np

import torch

from .joints_dataset import JointsDataset
from data.transforms.build import totensor
from vision.multiview import neighbor_cameras
from core import cfg

class MultiViewH36M(JointsDataset):
    actual_joints = {
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    }

    def __init__(self, root, image_set, is_train, transform=None):
        super().__init__(root, image_set, is_train, transform)

        anno_file = osp.join(self.root, 'h36m', 'annot',
                            'h36m_{}.pkl'.format(image_set))
        self.db = self.load_db(anno_file)

        if cfg.DATASETS.H36M.FILTER_DAMAGE:
            print('before filter', len(self.db))
            self.db = [db_rec for db_rec in self.db if not self.isdamaged(db_rec)]
            print('after filter', len(self.db))
            
        if cfg.DATASETS.H36M.MAPPING:
            assert cfg.KEYPOINT.NUM_PTS == 20
            self.u2a_mapping = super().get_mapping()
            super().do_mapping()
        else:
            assert cfg.KEYPOINT.NUM_PTS == 17

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)

        if cfg.VIS.MULTIVIEWH36M:
            from utils.metric_logger import MetricLogger
            self.meters = MetricLogger()

    @staticmethod
    def index_to_action_names():
        return {
            2: 'Direction',
            3: 'Discuss',
            4: 'Eating',
            5: 'Greet',
            6: 'Phone',
            7: 'Pose',
            8: 'Purchase',
            9: 'Sitting',
            10: 'SittingDown',
            11: 'Smoke',
            12: 'Photo',
            13: 'Wait',
            14: 'WalkDog',
            15: 'Walk',
            16: 'WalkTo'
        }

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        if self.is_train:
            if cfg.DATASETS.H36M.TRAIN_SAMPLE:
                filtered_grouping = filtered_grouping[::cfg.DATASETS.H36M.TRAIN_SAMPLE]
        else:
            if cfg.DATASETS.H36M.TEST_SAMPLE:
                filtered_grouping = filtered_grouping[::cfg.DATASETS.H36M.TEST_SAMPLE]

        return filtered_grouping

    def __getitem__(self, idx):
        if cfg.VIS.H36M:
            return super().__getitem__(idx)
        items = self.grouping[idx].copy()
        data = {}
        d = {}
        for cam, item in enumerate(items):
            datum = super().__getitem__(item)
            data[cam] = datum
            d[cam] = datum['KRT']
        rank = neighbor_cameras(d)

        if self.is_train:
            #TODO our training is shorter than the original code below
            if cfg.EPIPOLAR.TOPK == 3:
                # 0~3
                ref_cam, other_cam = np.random.choice(len(items), 2, replace=False)
            elif cfg.EPIPOLAR.TOPK == 2:
                ref_cam = np.random.randint(len(items))
                other_cam = np.random.choice(rank[ref_cam][0][:2])
            elif cfg.EPIPOLAR.TOPK == 1:
                ref_cam = np.random.randint(len(items))
                # ref_cam = random.choice(len(items))
                other_cam = rank[ref_cam][0][0]
            else:
                raise NotImplementedError

            

            ret = data[ref_cam]
            other_item = data[other_cam]
            if cfg.EPIPOLAR.PRIOR:
                ret['camera'] = ref_cam
                ret['other_camera'] = other_cam
            for i in ['img', 'KRT', 'heatmap', 'img-path']:
                ret['other_'+i] = other_item[i]
            print('multi h36m this view', ret['img-path'])            
            print('multi h36m other image', ret['other_img-path'])

            if cfg.VIS.MULTIVIEWH36M:
                from vision.multiview import findFundamentalMat, camera_center
                import matplotlib.pyplot as plt
                from data.transforms.image import de_transform
                P1 = ret['KRT']
                P2 = ret['other_KRT']
                C, _ = camera_center(P1)
                C_ = np.ones(4, dtype=C.dtype)
                C_[:3] = C
                e2 = P2 @ C_
                e2 /= e2[2]
                # world3d = ret['R'].T @ ret['points-3d'].T  + ret['T']
                othercam3d = other_item['R'] @ (ret['points-3d'].T  - other_item['T'])
                other2d = other_item['MultiViewH36M'] @ othercam3d
                other2d /=  other2d[-1]
                N = len(other_item['points-2d'])
                # print(other2d)
                # import matplotlib.pyplot as plt
                # plt.imshow(other_item['img'].cpu().numpy().transpose((1,2,0)))
                # plt.scatter(other2d[0], other2d[1])
                # plt.show()
                F = findFundamentalMat(P1, P2, engine='numpy')
                print(F)
                # points_2d = 
                print(ret['points-2d'])
                # ls = F @ np.concatenate((ret['points-2d']*4, np.ones((N, 1))), 1).T
                test_points = np.concatenate((np.ones((N, 1)) * 128, np.linspace(10, 250, N)[:, None], np.ones((N, 1))), 1)
                C2, _ = camera_center(P2)
                C_ = np.ones(4, dtype=C.dtype)
                C_[:3] = C2
                e1 = P1 @ C_
                e1 /= e1[2]
                l1s = np.cross(test_points, e1)
                ls = F @ test_points.T
                # res = np.concatenate((other_item['points-2d'], np.ones((N, 1))), 1)  @ F @ np.concatenate((ret['points-2d'], np.ones((N, 1))), 1).T
                # print(res)                
                fig = plt.figure(1)
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(de_transform(ret['img']).cpu().numpy().transpose((1,2,0))[..., ::-1])
                ax2.imshow(de_transform(other_item['img']).cpu().numpy().transpose((1,2,0))[..., ::-1])
                def scatterline(l):
                    x = np.arange(0, 256)
                    y = (-l[2] - l[0] * x) / l[1]
                    mask = (y < 256) & (y > 0)
                    return x[mask],  y[mask]

                for a, b in zip(ret['points-2d'][:, 0], ret['points-2d'][:, 1]):
                    ax1.scatter(a, b, color='red')
                for a, b in zip(other_item['points-2d'][:, 0], other_item['points-2d'][:, 1]):
                    ax2.scatter(a, b, color='green')
                for a, b in zip(other2d[0], other2d[1]):
                    ax2.scatter(a, b, color='red')
                for idx, (l, l1) in enumerate(zip(ls.T, l1s)):
                    # ax1.imshow(ret['original_image'][..., ::-1])
                    x1, y1 = scatterline(l1)
                    ax1.scatter(x1, y1, s=1)
                    # ax1.scatter(ret['points-2d'][idx, 0]*4, ret['points-2d'][idx, 1]*4, color='red')
                    # ax2.imshow(other_item['original_image'])
                    # ax2.scatter(other_item['points-2d'][idx, 0]*4, other_item['points-2d'][idx, 1]*4, color='yellow')
                    x, y = scatterline(l)
                    ax2.scatter(x, y, s=1)
                    # ax2.scatter(e2[0], e2[1], color='green')
                plt.show()

            return {k: totensor(v) for k, v in ret.items()}
        else:
            ret = {'camera': []}
            for k in datum.keys():
                ret[k] = []
            for k in ['img', 'KRT', 'heatmap', 'camera', 'img-path']:
                ret['other_'+k] = []
            for ref_cam, datum in data.items():
                ret['camera'].append(ref_cam)
                other_cam = rank[ref_cam][0][0]
                ret['other_camera'].append(other_cam)
                for k, v in datum.items():
                    ret[k].append(v)
                for k in ['img', 'KRT', 'heatmap', 'img-path']:
                    ret['other_'+k].append(data[other_cam][k])
            if cfg.KEYPOINT.NUM_CAM:
                for k in ret:
                    ret[k] = ret[k][:cfg.KEYPOINT.NUM_CAM]
            for k in ret:
                if not k in ['img-path', 'other_img-path']:
                    ret[k] = np.stack(ret[k])
            if cfg.DATASETS.H36M.REAL3D:
                real3d = self.computereal3d(ret['points-2d'], ret['K'], ret['RT'])
                ret['points-3d'][:] = real3d
            if cfg.VIS.MULTIVIEWH36M:
                return {k: totensor(v) if isinstance(v, torch.Tensor) else v for k, v in ret.items()}
                self.computereal3d(ret['points-2d'], ret['K'], ret['RT'], ret['KRT'], ret['points-3d'][0])
                return ret
            return {k: totensor(v) if isinstance(v, torch.Tensor) else v for k, v in ret.items()}

    def __len__(self):
        if cfg.VIS.H36M:
            return super().__len__()
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
        gt = np.array(gt)
        pred = pred[:, su, :2]

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)

    def computereal3d(self, pts, Ks, RTs, KRTs=None, gt3ds=None):
        from vision.triangulation import triangulate_pymvg
        if cfg.DATASETS.H36M.MAPPING:
            actualjoints = np.array([0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 11, 12, 14, 15, 16, 17, 18, 19])
            pts = pts[:, actualjoints]
        confs = np.ones((pts.shape[0], pts.shape[1]))
        real3ds = triangulate_pymvg(pts, Ks, RTs, confs)
        if not cfg.VIS.MULTIVIEWH36M:
            return real3ds

        gt3ds = gt3ds.cpu().numpy()
        KRTs = KRTs.cpu().numpy()
        pts = pts.cpu().numpy()
        # print('3d delta')
        # print(gt3ds-real3ds)
        # print('3d error')
        # print(np.linalg.norm(gt3ds-real3ds, axis=1))
        
        def reprojerr(real3ds):
            real2ds = []
            real3ds = np.concatenate((real3ds, np.ones((len(real3ds), 1))), 1)
            for KRT in KRTs:
                real2d = KRT @ real3ds.T
                real2d /= real2d[-1]
                real2ds.append(real2d[:2].T)
            # views x Njoints x 2
            real2ds = np.stack(real2ds)
            delta = real2ds - pts
            return np.linalg.norm(delta, axis=2).sum()
        
        err0 = reprojerr(gt3ds)
        err1 = reprojerr(real3ds)
        self.meters.update(gt3d=err0, real3d=err1)
        print(self.meters)
        
