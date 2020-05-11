import numpy as np
import scipy.ndimage
import skimage
import cv2

import torch

import matplotlib
from matplotlib import pylab as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# matplotlib.use('Agg')

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [
        (0, 1), (0, 4), #root hip
        (1, 2), (4, 5), #hip knee
        (2, 3), (5, 6), #knee ankle
        (0, 7), #root belly
        (7, 8), #belly neck
        (8, 9), #neck nose
        (9, 10), #nose head
        (8, 11), (8, 14), #neck shoulder
        (11, 12), (14, 15), #shoulder elbow
        (12, 13), (15, 16), #elbow wrist
        ],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

COLOR_DICT = {
    'coco': [
        (102, 0, 153), (153, 0, 102), (51, 0, 153), (153, 0, 153),  # head
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 102, 0), (153, 153, 0),  # right arm
        (0, 51, 153), (0, 0, 153),  # left leg
        (0, 153, 102), (0, 153, 153),  # right leg
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0)  # body
    ],

    'human36m': [
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # left leg
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102), (153, 0, 102),  # head
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0)   # left arm
    ],

    'kth': [
        (0, 153, 102), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153),  # left leg
        (153, 102, 0), (153, 153, 0),  # right arm
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), # body
        (102, 0, 153) # head
    ]
}

JOINT_NAMES_DICT = {
    'coco': {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
}

def draw_2d_pose(keypoints, ax, kind='human36m', keypoints_mask=None, point_size=8, line_width=3, radius=None, color=None):
    """
    Visualizes a 2d skeleton

    Args
        keypoints numpy array of shape (19, 2): pose to draw in CMU format.
        ax: matplotlib axis to draw on
    """
    connectivity = CONNECTIVITY_DICT[kind]

    color = 'blue' if color is None else color

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)


    # connections
    for i, (index_from, index_to) in enumerate(connectivity):
        if kind in COLOR_DICT:
            color = COLOR_DICT[kind][i]
        else:
            color = (0, 0, 255)        
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            xs, ys = [np.array([keypoints[index_from, j], keypoints[index_to, j]]) for j in range(2)]
            ax.plot(xs, ys, c=[c / 255. for c in color], lw=line_width, zorder=1)

    # points
    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], c='red', s=point_size, zorder=2)
    # if radius is not None:
    #     root_keypoint_index = 0
    #     xroot, yroot = keypoints[root_keypoint_index, 0], keypoints[root_keypoint_index, 1]

    #     ax.set_xlim([-radius + xroot, radius + xroot])
    #     ax.set_ylim([-radius + yroot, radius + yroot])

    # ax.set_aspect('equal')


def draw_2d_pose_cv2(keypoints, canvas, kind='cmu', keypoints_mask=None, point_size=2, point_color=(255, 255, 255), line_width=1, radius=None, color=None, anti_aliasing_scale=1):
    canvas = canvas.copy()

    shape = np.array(canvas.shape[:2])
    new_shape = shape * anti_aliasing_scale
    canvas = resize_image(canvas, tuple(new_shape))

    keypoints = keypoints * anti_aliasing_scale
    point_size = point_size * anti_aliasing_scale
    line_width = line_width * anti_aliasing_scale

    connectivity = CONNECTIVITY_DICT[kind]

    color = 'blue' if color is None else color

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    # connections
    for i, (index_from, index_to) in enumerate(connectivity):
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            pt_from = tuple(np.array(keypoints[index_from, :]).astype(int))
            pt_to = tuple(np.array(keypoints[index_to, :]).astype(int))

            if kind in COLOR_DICT:
                color = COLOR_DICT[kind][i]
            else:
                color = (0, 0, 255)

            cv2.line(canvas, pt_from, pt_to, color=color, thickness=line_width)

    if kind == 'coco':
        mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
        nose = keypoints[0, :]

        pt_from = tuple(np.array(nose).astype(int))
        pt_to = tuple(np.array(mid_collarbone).astype(int))

        if kind in COLOR_DICT:
            color = (153, 0, 51)
        else:
            color = (0, 0, 255)

        cv2.line(canvas, pt_from, pt_to, color=color, thickness=line_width)

    # points
    for pt in keypoints[keypoints_mask]:
        cv2.circle(canvas, tuple(pt.astype(int)), point_size, color=point_color, thickness=-1)

    canvas = resize_image(canvas, tuple(shape))

    return canvas


def draw_3d_pose(keypoints, ax, keypoints_mask=None, kind='cmu', radius=None, root=None, point_size=2, line_width=2, draw_connections=True):
    connectivity = CONNECTIVITY_DICT[kind]

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    if draw_connections:
        # Make connection matrix
        for i, joint in enumerate(connectivity):
            if keypoints_mask[joint[0]] and  keypoints_mask[joint[1]]:
                xs, ys, zs = [np.array([keypoints[joint[0], j], keypoints[joint[1], j]]) for j in range(3)]

                if kind in COLOR_DICT:
                    color = COLOR_DICT[kind][i]
                else:
                    color = (0, 0, 255)

                color = np.array(color) / 255

                ax.plot(xs, ys, zs, lw=line_width, c=color)

        if kind == 'coco':
            mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
            nose = keypoints[0, :]

            xs, ys, zs = [np.array([nose[j], mid_collarbone[j]]) for j in range(3)]

            if kind in COLOR_DICT:
                color = (153, 0, 51)
            else:
                color = (0, 0, 255)

            color = np.array(color) / 255

            ax.plot(xs, ys, zs, lw=line_width, c=color)


    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], keypoints[keypoints_mask][:, 2],
               s=point_size, c=np.array([230, 145, 56])/255, edgecolors='black')  # np.array([230, 145, 56])/255

    if radius is not None:
        if root is None:
            root = np.mean(keypoints, axis=0)
        xroot, yroot, zroot = root
        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
        ax.set_zlim([-radius + zroot, radius + zroot])

    ax.set_aspect('equal')


    # Get rid of the panes
    background_color = np.array([252, 252, 252]) / 255

    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_2d_pose(np.ones((17, 2)), ax)