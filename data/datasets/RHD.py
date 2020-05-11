import pickle
import os
import scipy.misc
from scipy.stats import truncnorm
import struct
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

from core import cfg
from data.transforms.build import *
from data.transforms.keypoints2d import *
from data.transforms.keypoints3d import *

class RHDDataset(Dataset):
    def __init__(self, root, ann_file, is_train=False):
        '''
        task: lifting, keypoint
        '''
        self.is_train = is_train
        with open(ann_file, 'rb') as fi:
            self.anno_all = pickle.load(fi)

        num_samples = len(self.anno_all.items())
        self.ids = [i for i in self.anno_all]
        self._imgpath = os.path.join(root, 'color', '%.5d.png')
        self._maskpath = os.path.join(root, 'mask', '%.5d.png')
        self.image_path = [self._imgpath % i for i in self.ids]

        self.coord_uv_noise = True
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_offset_noise = True
        self.crop_center_noise_sigma = 20.0
        self.crop_offset_noise_sigma = 10.0 
        self.hand_crop = True
        self.crop_size = cfg.LIFTING.CROP_SIZE
        self.image_size = cfg.LIFTING.IMAGE_SIZE
        self.unit = 1000
        self.heatmapcreator = Heatmapcreator(cfg.KEYPOINT.HEATMAP_SIZE, cfg.KEYPOINT.SIGMA)

    def __getitem__(self, idx):
        """
        return: [heatmap, handside, cropped_img],
                [3d joints, rotation, scale, visiblily, normed 3d joints]
        """
        sample_id = self.ids[idx]
        #Image.open
        img = scipy.misc.imread(self._imgpath % sample_id)
        img = img / 255. - .5

        hand_parts_mask = scipy.misc.imread(self._maskpath % sample_id)
        hand_parts_mask = hand_parts_mask.astype(int)
        hand_mask = hand_parts_mask > 1
        bg_mask = ~hand_mask
        hand_mask = np.stack((bg_mask, hand_mask), 2)

        anno = self.anno_all[sample_id]
        # get info from annotation dictionary
        # u, v coordinates of 42 hand keypoints, pixel
        keypoint_uv = anno['uv_vis'][:, :2].astype(float)  
        keypoint_vis = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
        # x, y, z coordinates of the keypoints, in meters, 42x3
        keypoint_xyz = anno['xyz']   
        cam_mat = anno['K']  # matrix containing intrinsic parameters

        # calculate palm coord
        if not cfg.DATASETS.WRIST_COORD:
            #calculate palm coord
            keypoint_xyz = palm_coord(keypoint_xyz)
            keypoint_uv = palm_coord(keypoint_uv)
            # calculate palm visibility
            palm_vis_l = np.logical_or(keypoint_vis[0], keypoint_vis[12])
            palm_vis_r = np.logical_or(keypoint_vis[21], keypoint_vis[33])
            keypoint_vis = np.hstack([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]])

        if self.is_train:
            #TODO add noise
            noise = np.random.normal(0, self.coord_uv_noise_sigma, (42, 2))
            keypoint_uv += noise
            # noise = tf.truncated_normal([42, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            # keypoint_uv += noise
            #TODO: hue augmentation
            # image = tf.image.random_hue(image, self.hue_aug_max)
            # pass
        
        """ DEPENDENT DATA ITEMS: SUBSET of 21 keypoints"""
        # figure out dominant hand by analysis of the segmentation mask
        num_px_left_hand = ((hand_parts_mask > 1) & (hand_parts_mask < 18)).sum()
        num_px_right_hand = (hand_parts_mask > 17).sum()

        # PRODUCE the 21 subset using the segmentation masks
        # We only deal with the more prominent hand for each frame and discard the second set of keypoints
        kp_coord_xyz_left = keypoint_xyz[:21, :]
        kp_coord_xyz_right = keypoint_xyz[-21:, :]

        if num_px_left_hand > num_px_right_hand:
            cond_left = True
            kp_coord_xyz21 = kp_coord_xyz_left
            hand_side = 0
        else:
            cond_left = False
            kp_coord_xyz21 = kp_coord_xyz_right
            hand_side = 1

        # make coords relative to root joint
        kp_coord_xyz_root = kp_coord_xyz21[0, :] # this is the palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        keypoint_scale = np.sqrt(np.sum((kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])**2))

        keypoint_xyz21_normed = kp_coord_xyz21_rel / keypoint_scale  # normalized by length of 12->11

        # calculate local coordinates
        #kp_coord_xyz21_local = bone_rel_trafo(keypoint_xyz21_normed)
        #kp_coord_xyz21_local = tf.squeeze(kp_coord_xyz21_local)

        # calculate viewpoint and coords in canonical coordinates
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(keypoint_xyz21_normed)
        if not cond_left and cfg.LIFTING.FLIP_ON:
            kp_coord_xyz21_rel_can = flip_hand(kp_coord_xyz21_rel_can)

        # Set of 21 for visibility
        keypoint_vis_left = keypoint_vis[:21]
        keypoint_vis_right = keypoint_vis[-21:]
        # Set of 21 for UV coordinates
        keypoint_uv_left = keypoint_uv[:21, :]
        keypoint_uv_right = keypoint_uv[-21:, :]
        if cond_left:
            keypoint_vis21 = keypoint_vis_left
            keypoint_uv21 = keypoint_uv_left
        else:
            keypoint_vis21 = keypoint_vis_right
            keypoint_uv21 = keypoint_uv_right


        """ DEPENDENT DATA ITEMS: HAND CROP """
        if self.hand_crop:
            crop_center = keypoint_uv21[12, ::-1]

            # catch problem, when no valid kp available (happens almost never)
            if not np.isfinite(crop_center).all():
                crop_center = np.array([0.0, 0.0])

            #TODO
            if self.is_train:
                noise = np.random.normal(0, self.crop_center_noise_sigma, 2)
                crop_center += noise
                # noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_center_noise_sigma)
                # crop_center += noise
            crop_scale_noise = 1.
            #if self.crop_scale_noise:
            #        crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=1.0, maxval=1.2))

            # select visible coords only
            kp_coord_hw = keypoint_uv21 * keypoint_vis21[..., None] 
            # determine size of crop (measure spatial extend of hw coords first)
            min_coord = np.maximum(np.min(kp_coord_hw, 0), 0.0)
            max_coord = np.minimum(np.max(kp_coord_hw, 0), self.image_size)

            # find out larger distance wrt the center of crop
            crop_size_best = 2*np.maximum(max_coord - crop_center, crop_center - min_coord)
            crop_size_best = np.max(crop_size_best)
            crop_size_best = np.minimum(np.maximum(crop_size_best, 50.0), 500.0)

            # catch problem, when no valid kp available
            if not np.isfinite(crop_size_best).all():
                crop_size_best = 200.

            # calculate necessary scaling
            scale = float(self.crop_size) / crop_size_best
            scale = np.minimum(np.maximum(scale, 1.0), 10.0)
            scale *= crop_scale_noise

            # TODO: some keypoint_uv have negative values, remove them instead cropping
            clip_crop_center = clip_to_image(crop_center, (self.image_size, self.image_size))
            raw_bbox = (clip_crop_center[0], clip_crop_center[1], crop_size_best, crop_size_best)
            bbox = ccwh_to_xyxy(raw_bbox)
            bbox = clip_to_image(bbox, (self.image_size, self.image_size))
            cropped_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            cropped_img = scipy.misc.imresize(cropped_img, (self.crop_size, self.crop_size))
            #try:
            #    cropped_img = scipy.misc.imresize(cropped_img, (self.crop_size, self.crop_size))
            #except:
            #    print(raw_bbox)
            #    print(bbox)
            #    exit()

            cropped_img = np.transpose(cropped_img, (2, 0, 1))

            #TODO
            if self.crop_offset_noise:
                noise = np.random.normal(0, self.crop_offset_noise_sigma, 2)
                crop_center += noise
                # noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma)
                # crop_center += noise

            #TODO
            # Crop image
            # img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, self.crop_size, scale)
            # data_dict['image_crop'] = tf.squeeze(img_crop)
            # img_crop = crop_image_from_xy(image[np.newaxis, :, :, :], crop_center, self.crop_size, scale)
            # data_dict['image_crop'] = tf.squeeze(img_crop)
            

            # Modify uv21 coordinates
            keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center[0]) * scale + self.crop_size // 2
            keypoint_uv21 = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)

            #TODO
            ## Modify camera intrinsics
            #scale = tf.reshape(scale, [1, ])
            #scale_matrix = tf.dynamic_stitch([[0], [1], [2],
            #                                  [3], [4], [5],
            #                                  [6], [7], [8]], [scale, [0.0], [0.0],
            #                                                   [0.0], scale, [0.0],
            #                                                   [0.0], [0.0], [1.0]])
            #scale_matrix = tf.reshape(scale_matrix, [3, 3])
            #crop_center_float = tf.cast(crop_center, tf.float32)
            #trans1 = crop_center_float[0] * scale - self.crop_size // 2
            #trans2 = crop_center_float[1] * scale - self.crop_size // 2
            #trans1 = tf.reshape(trans1, [1, ])
            #trans2 = tf.reshape(trans2, [1, ])
            #trans_matrix = tf.dynamic_stitch([[0], [1], [2],
            #                                  [3], [4], [5],
            #                                  [6], [7], [8]], [[1.0], [0.0], -trans2,
            #                                                   [0.0], [1.0], -trans1,
            #                                                   [0.0], [0.0], [1.0]])
            #trans_matrix = tf.reshape(trans_matrix, [3, 3])
            #cam_mat = tf.matmul(trans_matrix, tf.matmul(scale_matrix, cam_mat))

        """ DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints"""
        # create scoremaps from the subset of 2D annoataion
        keypoint_hw21 = np.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], 1)

        scoremap = self.heatmapcreator.get(keypoint_hw21, keypoint_vis21)

        ret = {
                'heatmap': scoremap,
                'hand-side' : hand_side,
                'img': cropped_img,
                'can-points-3d'    : kp_coord_xyz21_rel_can,
                'rotation'         : rot_mat,
                'scale'            : keypoint_scale,
                'visibility'       : keypoint_vis21,
                'normed-points-3d' : keypoint_xyz21_normed,
                'unit': self.unit,
              }
        return {k: totensor(v) for k, v in ret.items()}

    def crop_image_from_xy(self, image, crop_location, crop_size, scale=1.0):
        s = image.get_shape().as_list()
        assert len(s) == 4, "Image needs to be of shape [batch, width, height, channel]"
        scale = tf.reshape(scale, [-1])
        crop_location = tf.cast(crop_location, tf.float32)
        crop_location = tf.reshape(crop_location, [s[0], 2])
        crop_size = tf.cast(crop_size, tf.float32)

        crop_size_scaled = crop_size / scale
        y1 = crop_location[:, 0] - crop_size_scaled//2
        y2 = y1 + crop_size_scaled
        x1 = crop_location[:, 1] - crop_size_scaled//2
        x2 = x1 + crop_size_scaled
        y1 /= s[1]
        y2 /= s[1]
        x1 /= s[2]
        x2 /= s[2]
        boxes = tf.stack([y1, x1, y2, x2], -1)

        crop_size = tf.cast(tf.stack([crop_size, crop_size]), tf.int32)
        box_ind = tf.range(s[0])
        image_c = tf.image.crop_and_resize(tf.cast(image, tf.float32), boxes, box_ind, crop_size, name='crop')
        return image_c

    def _preprocess(self):
        #TODO preprocess the data to speed up
        pass
        
        
    def __len__(self):
        return len(self.ids)
