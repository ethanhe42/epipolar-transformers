if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from data.transforms.image import de_transform
from vision.multiview import coord2pix, pix2coord
import scipy
from matplotlib.patches import Circle
import time
import math

RGB_MATCHING_COLOR = '#0066cc'
BASELINE_MATCHING_COLOR = 'y'
OURS_MATCHING_COLOR = 'r'
GROUNDTRUTH_COLOR = 'g'


def de_normalize(pts, H, W, engine='numpy'):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    pts_ = pts.copy()
    if engine == 'torch':
        WH = torch.tensor([W, H], dtype=pts.dtype, device=pts.device)
        return (pts + 1) * (WH - 1) / 2.
    pts_[..., 0] = (pts[..., 0] + 1) * (W - 1) / 2.
    pts_[..., 1] = (pts[..., 1] + 1) * (H - 1) / 2.
    return pts_

def normalize(pts, H, W):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    pts_ = pts.copy()
    pts_[..., 0] = -1. + 2. * pts[..., 0] / (W - 1)
    pts_[..., 1] = -1. + 2. * pts[..., 1] / (H - 1)
    return pts_

def BGR2Lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def Lab2ab(image):
    _, A, B = cv2.split(image)
    return np.stack([A, B])

class Output(object):
    def __init__(self, pkl_path):
        with open(pkl_path,"rb") as f:
            output = pickle.load(f)

        img1 = output['img1'][0]
        img1 = de_transform(img1).transpose(1,2,0)
        img2 = output['img2'][0]
        img2 = de_transform(img2).transpose(1,2,0)

        self.img1 = img1[:, :, ::-1]
        self.img2 = img2[:, :, ::-1]

        img1_ab = Lab2ab(BGR2Lab(img1)).transpose(1,2,0)
        img2_ab = Lab2ab(BGR2Lab(img2)).transpose(1,2,0)
        self.img1_ab = img1_ab
        self.img2_ab = img2_ab

        self.depth = output['depth']
        self.corr_pos_pred = output['corr_pos_pred']
        self.sample_locs = output['sample_locs']
        self.img1_path = output['img1_path']
        self.img2_path = output['img2_path']
        self.camera = output['camera'][0]
        self.other_camera = output['other_camera'][0]
        self.heatmap_pred = output['heatmap_pred']
        self.batch_locs = output['batch_locs']
        self.points_2d = output['points-2d']

        self.H, self.W = img1.shape[:2]

    def calc_color_score(self, x, y):
        cx, cy = int(coord2pix(x, 4)), int(coord2pix(y, 4))
        ref_point = self.img1_ab[int(y), int(x), :]
        color_score = []
        max_score_id = None
        max_score = -1
        
        for i in range(0, 64):
            pos = self.sample_locs[i][int(cy)][int(cx)]
            depos = de_normalize(pos, self.H, self.W)
            source_point = self.img2_ab[int(depos[1]), int(depos[0]), :]
            color_score.append(np.dot(ref_point, source_point))
            if color_score[-1] > max_score:
                max_score = color_score[-1]
                max_score_id = (int(depos[0]), int(depos[1]))
                
        color_score = color_score / sum(color_score)
        return color_score, max_score_id

class Complex_Draw(object):
    def __init__(self, output, b_output):
        self.output = output
        self.b_output = b_output
        self.ref_img = output.img1
        assert output.img1_path == b_output.img1_path
    
    def draw_sample_ax(self, ax, x, y):
        output = self.output
        b_output = self.b_output
        cx, cy = int(coord2pix(x, 4)), int(coord2pix(y, 4))
        ax.clear()

        # update the line positions
        ax.imshow(self.ref_img)
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        circ = Circle((x, y), 3, color=GROUNDTRUTH_COLOR)
        ax.add_patch(circ)

        self.txt.set_text('x=%1.1f, y=%1.1f; g: groundtruth; y: baseline; r: prediction' % (x, y))

    def draw_dist_ax(self, ax, x, y):
        output = self.output
        cx, cy = int(coord2pix(x, 4)), int(coord2pix(y, 4))
        color_score, max_score_id = output.calc_color_score(x, y)
        xrange = np.arange(0, 64)
        ax.clear()

        lines_color = {
            'feat. matching': OURS_MATCHING_COLOR,
            'rgb matching' : '#0066cc',
            'non-fusion feat. matching': BASELINE_MATCHING_COLOR,
        }
        lines_data = {
            'feat. matching': output.depth[:, cy, cx],
            'rgb matching' : color_score,
            'non-fusion feat. matching': self.b_output.depth[:, cy, cx],
        }

        ax.clear()
        for label, line in lines_data.items():
            ax.plot(xrange[1:-1], line[1:-1], color=lines_color[label], label=label)
        ax.set_yscale('log')
        ax.set_ylabel('similarity (log)')
        ax.tick_params(bottom=False, top=True)
        ax.tick_params(labelbottom=False, labeltop=True)
        ax.legend()

        return max_score_id
    
    def draw_other_ax(self, ax, x, y, max_score_id, joint_id=None):
        output = self.output
        b_output = self.b_output
        cx, cy = int(coord2pix(x, 4)), int(coord2pix(y, 4))
        xx, yy = output.corr_pos_pred[cy][cx]
        bxx, byy = self.b_output.corr_pos_pred[cy][cx]

        ax.clear()
        ax.imshow(output.img2)
        circ = Circle(max_score_id, 3, color=RGB_MATCHING_COLOR)
        ax.add_patch(circ)
        
        # draw epipolar lines
        line_start1 = de_normalize(output.sample_locs[1][int(cy)][int(cx)], output.H, output.W)
        line_start2 = de_normalize(output.sample_locs[63][int(cy)][int(cx)], output.H, output.W)
        ax.plot([line_start1[0], line_start2[0]], [line_start1[1], line_start2[1]], alpha=0.5, color='b', zorder=1)
        
        # draw groundtruth points
        # for i in range(17):
        gx, gy = output.points_2d[output.other_camera][joint_id][0], output.points_2d[output.other_camera][joint_id][1]
        circ = Circle((gx, gy), 3, color=GROUNDTRUTH_COLOR, zorder=2)
        ax.add_patch(circ)

        # draw baseline predicted point
        circ = Circle((pix2coord(bxx, 4), pix2coord(byy, 4)), 3, color=BASELINE_MATCHING_COLOR, zorder=2)
        ax.add_patch(circ)

        # draw predicted point
        circ = Circle((pix2coord(xx, 4), pix2coord(yy, 4)), 3, color=OURS_MATCHING_COLOR, zorder=3)
        ax.add_patch(circ)

        def dist(x1, y1, x2, y2):
            return math.sqrt((x1 - x2)**2 + (y1-y2) **2)

        flag = True
        #  predicted - gt > baseline - gt
        if dist(pix2coord(xx, 4), pix2coord(yy,4), gx, gy)*1.5 > dist(pix2coord(bxx, 4), pix2coord(byy,4), gx, gy):
            flag = False
        #  predicted - gt > TH: 3
        if dist(pix2coord(bxx, 4), pix2coord(byy,4), gx, gy) < 5:
            flag = False

        if flag:
            print('img1 path: ', output.img1_path)
            print('img2 path: ', output.img2_path)
            print('pred - gt: ', dist(pix2coord(xx, 4), pix2coord(yy,4), gx, gy))
            print('baseline - gt', dist(pix2coord(bxx, 4), pix2coord(byy,4), gx, gy))

        txt = self.sample_ax.text(0, 0, '', va="bottom", ha="left")
        txt.set_text('g: groundtruth; y: baseline; r: our prediction')
        return flag
    
    def draw_heatmap_ax(self, ax):
        output = self.output
        ax.clear()
        ax.imshow(output.heatmap_pred.max(0))

    def draw(self, x, y, save_path, joint_id=None):

        self.fig, self.axs = plt.subplots(2, 2, squeeze=True, figsize=(12, 8))
        self.sample_ax = self.axs[0, 0]
        self.dist_ax = self.axs[0, 1]
        self.other_ax = self.axs[1, 0]
        self.heatmap_ax = self.axs[1, 1]

        self.lx = self.sample_ax.axhline(color='k')  # the horiz line
        self.ly = self.sample_ax.axvline(color='k')  # the vert line

        self.txt = self.sample_ax.text(0, 0, '', va="bottom", ha="left")

        output = self.output
        self.draw_sample_ax(self.sample_ax, x, y)
        max_score_id = self.draw_dist_ax(self.dist_ax, x, y)
        flag = self.draw_other_ax(self.other_ax, x, y, max_score_id, joint_id)
        if not flag:
            plt.close()
            return flag
        self.draw_heatmap_ax(self.heatmap_ax)

        plt.savefig(save_path) #, transparent=True)
        print('saved for ', save_path)
        return flag


class Easy_Draw(Complex_Draw):
    def __init__(self, output, b_output):
        self.output = output
        self.b_output = b_output
        self.ref_img = output.img1
        assert output.img1_path == b_output.img1_path
    
    def draw(self, x, y, save_path):
        self.fig, self.ax = plt.subplots(1, figsize=(12, 8))
        output = self.output
        self.draw_dist_ax(self.ax, x, y)
        plt.savefig(save_path, transparent=True)
        print('saved for ', save_path)


root_dir = "outs/epipolar/keypoint_h36m_fixed/visualizations/h36m/"
# for i in range(4,5):

i = 1
j = 2 

ours_pkl = root_dir + "output_{}.pkl".format(i)
baseline_pkl = root_dir + "output_baseline_{}.pkl".format(i)
complex_output = root_dir + "{}_joint{}_output.eps"
easy_output = root_dir + "easy_output/{}_joint{}_easy_output.eps"

output = Output(ours_pkl)
b_output = Output(baseline_pkl)

cd = Complex_Draw(output, b_output)
ed = Easy_Draw(output, b_output)

flag = cd.draw(x=output.points_2d[output.camera][j][0], y=output.points_2d[output.camera][j][1], save_path=complex_output.format(i, j), joint_id=j)
if flag:
    ed.draw(x=output.points_2d[output.camera][j][0], y=output.points_2d[output.camera][j][1], save_path=easy_output.format(i, j))

fig, ax = plt.subplots()
plt.imshow(output.img1)
ax.axis('off')
fig.savefig(root_dir+'original/{}_ref_img.eps'.format(i),bbox_inches='tight', pad_inches=0)
fig, ax = plt.subplots()
ax.axis('off')
plt.imshow(output.img2)
fig.savefig(root_dir+'original/{}_source_img.eps'.format(i),bbox_inches='tight', pad_inches=0)
print('saved original images')
