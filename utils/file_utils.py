import os, sys, glob, numbers
from os import path as osp
import numpy as np

from vision.multiview import neighbor_cameras

def load_txt_file(file_path):
    '''
    load data or string from text file.
    '''
    if not osp.exists(file_path):
        file_path = file_path.replace('/home/mscv', '~')
        file_path = osp.expanduser(file_path)
    cfile = open(file_path, 'r')
    content = cfile.readlines()
    cfile.close()
    content = [x.strip() for x in content]
    num_lines = len(content)
    return content, num_lines


def readKRT_file(file):
    with open(file, "r") as fp:
        lines = [l.strip() for l in fp]
    KRT = dict()
    Rt = dict()
    intrinsic = dict()
    C = dict()
    wh = dict()
    u = 0
    while u < len(lines):
        cam, w, h = lines[u].split()
        w = int(w)
        h = int(h)
        K = np.array( [lines[u+1].split(), lines[u+2].split(), lines[u+3].split()] ).astype(np.float32)
        Rt_ = np.array( [lines[u+5].split(), lines[u+6].split(), lines[u+7].split()] ).astype(np.float32)
        KRT[cam] = np.dot(K, Rt_)
        Rt[cam] = Rt_
        C[cam] = - np.dot(np.transpose(Rt_[:, :3]), Rt_[:, 3])
        wh[cam] = (w, h)
        intrinsic[cam] = K
        u += 9
    return {'KRT': KRT, 
            'Rt' :Rt, 
            'K'  : intrinsic,
            'C'  :C, 
            'wh' :wh, 
            'neighbor': neighbor_cameras(KRT),
            }
