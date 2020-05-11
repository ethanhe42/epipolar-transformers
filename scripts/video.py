import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import glob

parser = argparse.ArgumentParser(description="PyTorch Keypoints Training")
parser.add_argument(
    "--src",
    default="outs/epipolar/keypoint_h36m_fixed/video/multiview_h36m_val/",
    help="source folder",
    type=str,
)
parser.add_argument(
    "--dst",
    default='outs/skeletons.mp4',
    help="dst video path",
    type=str,
)
parser.add_argument(
    "--fps",
    default=24,
    help="FPS",
    type=int,
)

args = parser.parse_args()
# image_folder = args.src

video_name = args.dst
if video_name.endswith('avi'):
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
elif video_name.endswith('mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
else:
    raise NotImplementedError


frameSize = (369*2, 369*2)
# newsize = (256*2, 256*2)

sources = [
    'outs/video_gt/multiview_h36m_val/*.png',
    'outs/benchmark/keypoint_h36m/video/multiview_h36m_val/*.png',
    '../multiview-human-pose-estimation-pytorch/output/multiview_h36m/multiview_pose_resnet_50/256_fusion/image_with_joints/validation_view_1*',
    'outs/epipolar/keypoint_h36m_fixed/video/multiview_h36m_val/*.png',
]
titles = [
    'ground truth',
    'baseline (48.7 mm)',
    'crossview ICCV\'19 (45.5 mm)',
    'epipolar transformer (33.1 mm)',
]
footnote = 'all methods are with image size 256x256, ResNet-50'

# font                   = cv2.FONT_HERSHEY_SIMPLEX
# font                   = 4 #FONT_HERSHEY_TRIPLEX
font                   = 2 #FONT_HERSHEY_DUPLEX
fontScale              = 0.5
fontColors              = [
    (0,0,0),
    (0,0,0),
    (0,0,0),
    (39, 159, 39),
]
lineType               = cv2.LINE_AA #2



locs = [
    (0, 0), 
    (0, frameSize[1] // 2),
    (frameSize[0] // 2, 0),
    (frameSize[0] // 2, frameSize[1] // 2),
]

images_list = []
for image_folder in sources:
    images = sorted([img for img in glob.glob(image_folder)])
    images_list.append(images)
    print(images[0])

video = cv2.VideoWriter(video_name, fourcc=fourcc, #0, 
    fps=args.fps, 
    frameSize=frameSize
    # frameSize=newsize #frameSize
    )

for i in tqdm(range(len(images_list[0]))):  
    try:
        img = np.zeros((frameSize[0],frameSize[1],3), np.uint8)
        img[...] = 255
        for images, loc, title, fontColor in zip(images_list, locs, titles, fontColors):
            image = images[i]
            frame = cv2.imread(image)
            height, width, layers = frame.shape

            if height != frameSize[0] // 2 or width != frameSize[1] // 2:
                # print(height, width, image)
                frame = frame[5:374, 36:405]
            img[loc[0]:loc[0]+frameSize[0] // 2, loc[1]:loc[1]+frameSize[1] // 2] = frame
            cv2.putText(img, title, 
                (loc[1] + 20, loc[0]+20), 
                font, 
                fontScale,
                fontColor,
                1,
                lineType)
        cv2.putText(img, footnote, 
            (240, frameSize[0] - 10), 
            font, 
            0.3,
            (255, 255, 255),
            1,
            lineType,
            )                
        # img = cv2.resize(img, newsize)
        video.write(img)
    except Exception as e:
        print(e)
        break
cv2.destroyAllWindows()
video.release()