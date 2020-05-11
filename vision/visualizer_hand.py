import numpy as np
import scipy.ndimage
import skimage
import cv2

import torch

import matplotlib
from matplotlib import pylab as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# matplotlib.use('Agg')

hand_colors = np.array([[0., 0., 0.5],
                    [0., 0., 0.73172906],
                    [0., 0., 0.96345811],
                    [0., 0.12745098, 1.],
                    [0., 0.33137255, 1.],
                    [0., 0.55098039, 1.],
                    [0., 0.75490196, 1.],
                    [0.06008855, 0.9745098, 0.90765338],
                    [0.22454143, 1., 0.74320051],
                    [0.40164453, 1., 0.56609741],
                    [0.56609741, 1., 0.40164453],
                    [0.74320051, 1., 0.22454143],
                    [0.90765338, 1., 0.06008855],
                    [1., 0.82861293, 0.],
                    [1., 0.63979666, 0.],
                    [1., 0.43645606, 0.],
                    [1., 0.2476398, 0.],
                    [0.96345811, 0.0442992, 0.],
                    [0.73172906, 0., 0.],
                    [0.5, 0., 0.]])

# define connections and hand_colors of the hand_bones
hand_bones = [((0, 1), hand_colors[0, :]),
            ((1, 2), hand_colors[1, :]),
            ((2, 3), hand_colors[2, :]),
            ((3, 20), hand_colors[3, :]),

            ((4, 5), hand_colors[4, :]),
            ((5, 6), hand_colors[5, :]),
            ((6, 7), hand_colors[6, :]),
            ((7, 20), hand_colors[7, :]),

            ((8, 9), hand_colors[8, :]),
            ((9, 10), hand_colors[9, :]),
            ((10, 11), hand_colors[10, :]),
            ((11, 20), hand_colors[11, :]),

            ((12, 13), hand_colors[12, :]),
            ((13, 14), hand_colors[13, :]),
            ((14, 15), hand_colors[14, :]),
            ((15, 20), hand_colors[15, :]),

            ((16, 17), hand_colors[16, :]),
            ((17, 18), hand_colors[17, :]),
            ((18, 19), hand_colors[18, :]),
            ((19, 20), hand_colors[19, :])]
# #RHD
# hand_colors = np.array([[0., 0., 0.5],
#                     [0., 0., 0.73172906],
#                     [0., 0., 0.96345811],
#                     [0., 0.12745098, 1.],
#                     [0., 0.33137255, 1.],
#                     [0., 0.55098039, 1.],
#                     [0., 0.75490196, 1.],
#                     [0.06008855, 0.9745098, 0.90765338],
#                     [0.22454143, 1., 0.74320051],
#                     [0.40164453, 1., 0.56609741],
#                     [0.56609741, 1., 0.40164453],
#                     [0.74320051, 1., 0.22454143],
#                     [0.90765338, 1., 0.06008855],
#                     [1., 0.82861293, 0.],
#                     [1., 0.63979666, 0.],
#                     [1., 0.43645606, 0.],
#                     [1., 0.2476398, 0.],
#                     [0.96345811, 0.0442992, 0.],
#                     [0.73172906, 0., 0.],
#                     [0.5, 0., 0.]])

# # define connections and hand_colors of the hand_bones
# hand_bones = [((1, 2), hand_colors[0, :]),
#             ((2, 3), hand_colors[1, :]),
#             ((3, 4), hand_colors[2, :]),
#             ((4, 0), hand_colors[3, :]),

#             ((5, 6), hand_colors[4, :]),
#             ((6, 7), hand_colors[5, :]),
#             ((7, 8), hand_colors[6, :]),
#             ((8, 0), hand_colors[7, :]),

#             ((9, 10), hand_colors[8, :]),
#             ((10, 11), hand_colors[9, :]),
#             ((11, 12), hand_colors[10, :]),
#             ((12, 0), hand_colors[11, :]),

#             ((13, 14), hand_colors[12, :]),
#             ((14, 15), hand_colors[13, :]),
#             ((15, 16), hand_colors[14, :]),
#             ((16, 0), hand_colors[15, :]),

#             ((17, 18), hand_colors[16, :]),
#             ((18, 19), hand_colors[17, :]),
#             ((19, 20), hand_colors[18, :]),
#             ((20, 0), hand_colors[19, :])]

def plot_hand_3d(coords_xyz, occlusion, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    for connection, color in hand_bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            if occlusion[0] * occlusion[1] ==0:
                continue
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            if occlusion[0] * occlusion[1] ==0:
                continue
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)
    axis.view_init(azim=-90., elev=90.)

def plot_single_hand_2d(keypoints, ax, occlusion=None, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    for connection, color in hand_bones:
        coord1 = keypoints[connection[0], :]
        coord2 = keypoints[connection[1], :]
        coords = np.stack([coord1, coord2])
        if (coords[:, 0] <= 1).any():
            continue
        if (coords[:, 1] <= 1).any():
            continue
        if occlusion is not None:
            if not occlusion[0] or not occlusion[1]:
                continue
        if color_fixed is None:
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            ax.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)

def plot_two_hand_2d(keypoints, ax, occlusion=None, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    plot_single_hand_2d(keypoints[:21], ax, None if occlusion is None else occlusion[:21], color_fixed, linewidth)
    plot_single_hand_2d(keypoints[21:], ax, None if occlusion is None else occlusion[21:], color_fixed, linewidth)

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_two_hand_2d(np.ones((42, 2)), ax, np.ones(42))