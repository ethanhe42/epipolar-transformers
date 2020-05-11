if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

import os
import torch
from utils.model_serialization import strip_prefix_if_present
from utils import zipreader
import argparse
from tqdm import tqdm
import pickle
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Keypoints Training")
parser.add_argument(
    "--src",
    default="~/datasets",
    help="source model",
    type=str,
)
parser.add_argument(
    "--dst",
    default="~/local/datasets/h36m/undistortedimages",
    help="dst model",
    type=str,
)
parser.add_argument(
    "--anno",
    default="~/datasets/h36m/annot/h36m_validation.pkl",
    type=str,
)

args = parser.parse_args()
src = os.path.expanduser(args.src)
dst = os.path.expanduser(args.dst)
with open(os.path.expanduser(args.anno), 'rb') as f:
    data = pickle.load(f) 

for db_rec in tqdm(data):
    path = db_rec['image']
    image_dir = 'images.zip@'
    image_file = os.path.join(src, db_rec['source'], image_dir, 'images', db_rec['image'])
    output_path = os.path.join(dst, path)
    if os.path.exists(output_path):
        continue
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    data_numpy = zipreader.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    camera = db_rec['camera']
    K = np.array([
        [float(camera['fx']), 0, float(camera['cx'])], 
        [0, float(camera['fy']), float(camera['cy'])], 
        [0, 0, 1.], 
        ])
    distCoeffs = np.array([float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
    data_numpy = cv2.undistort(data_numpy, K, distCoeffs)
    #cv2.imwrite(output_path, data_numpy, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #cv2.imwrite(output_path, data_numpy)
    cv2.imwrite(output_path, data_numpy, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

