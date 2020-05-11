import logging
import pickle
from collections import OrderedDict
from IPython import embed
import numpy as np

import torch

from core import cfg
from utils.registry import Registry

basic_pairs = [
        ["/", "."],
        ['weights', 'weight'],
        ["biases", "bias"],
        ['0_1', '0'],
        ['0_2', '2'],
        ['1_1', '4'],
        ['1_2', '6'],
        ['2_1', '8'],
        ['2_2', '10'],
        ]
lifting_pairs = [
        ['PosePrior.conv_pose_', 'conv1.'],
        ['PosePrior.fc_', 'poseprior.'],
        ['ViewpointNet.conv_vp_', 'conv2.'],
        ['ViewpointNet.fc_', 'viewpoint.'],
        ['rel0', '0'],
        ['rel1', '3'],
        ['xyz', '6'],
        ['vp0', '0'],
        ['vp1', '3'],
        ]


def _rename_weights(layer_keys, replace_pairs):
    for i, j in replace_pairs:
        layer_keys = [k.replace(i, j) for k in layer_keys]
    return layer_keys

def prefix_modulename(d, modulename='liftingnet.module.'):
    for i in list(d.keys()):
        d[modulename + i] = d.pop(i)
    return d

def totorch(d):
    for i in d.keys():
        w = d[i]
        if len(w.shape) == 4:
            #HWCN -> NCHW
            w = np.transpose(w, [3,2,0,1])
        elif len(w.shape) == 2:
            w = w.T
        d[i] = torch.from_numpy(w)

def _load_tf_pickled_weights(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

EXTERNAL_LOADER = Registry()

@EXTERNAL_LOADER.register("lifting_rot")
def load_lifting_net(path='/home/mscv/hand3d/weights/lifting-proposed.pickle'):
    path = cfg.WEIGHTS
    weights = _load_tf_pickled_weights(path)
    keys = _rename_weights(weights.keys(), basic_pairs + lifting_pairs)
    key_map = {k: v for k, v in zip(weights.keys(), keys)}

    for i, j in zip(weights, keys):
        print(i, j)
    print(len(set(keys)))

    new_weights = OrderedDict()
    for k, w in weights.items():
        new_weights[key_map[k]] = w
    new_weights['viewpoint.6.weight'] = np.hstack([
        new_weights['viewpoint.vp_ux.weight'],
        new_weights['viewpoint.vp_uy.weight'],
        new_weights['viewpoint.vp_uz.weight'],
        ])
    new_weights['viewpoint.6.bias'] = np.hstack([
        new_weights['viewpoint.vp_ux.bias'],
        new_weights['viewpoint.vp_uy.bias'],
        new_weights['viewpoint.vp_uz.bias'],
        ])
    new_weights.pop('viewpoint.vp_ux.weight')
    new_weights.pop('viewpoint.vp_uy.weight')
    new_weights.pop('viewpoint.vp_uz.weight')
    new_weights.pop('viewpoint.vp_ux.bias')
    new_weights.pop('viewpoint.vp_uy.bias')
    new_weights.pop('viewpoint.vp_uz.bias')
    prefix_modulename(new_weights)
    for k, w in new_weights.items():
        print(k, w.shape)
    totorch(new_weights)
    return dict(model=new_weights)


def load_ext_weights():
    return EXTERNAL_LOADER[cfg.DATASETS.TASK]()

if __name__ == '__main__':
    load_lifting_net()
