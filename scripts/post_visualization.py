import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.patches import Circle
import time
import pickle

def de_normalize(pts, H, W, engine='numpy'):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    if engine == 'torch':
        WH = torch.tensor([W, H], dtype=pts.dtype, device=pts.device)
        return (pts + 1) * (WH - 1) / 2.
    pts_ = pts.copy()
    pts_[..., 0] = (pts[..., 0] + 1) * (W - 1) / 2.
    pts_[..., 1] = (pts[..., 1] + 1) * (H - 1) / 2.
    return pts_

"""
    Load related information
"""

# pkl_name = 'output_1.pkl'
pkl_name = 'outs/epipolar/rgb/keypoint_HG11_no_other_gradoutput_0.pkl'

with open(pkl_name,"rb") as f:
    output = pickle.load(f)
print(output.keys())
img1 = output['img1']
img2 = output['img2']
RT = output['RT']
other_RT = output['other_RT']
depth = output['depth']
corr_pos_pred = output['corr_pos_pred']
sample_locs = output['sample_locs']
img1_path = output['img1_path']
img2_path = output['img2_path']
print(depth.shape)

ref_img = img1

H, W = ref_img.shape[:2]
print(img1_path)
print(img2_path)


"""
    Draw with Cursor
"""


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

# sample the probability cost-volume

class Cursor(object):
    def __init__(self, sample_ax, draw_ax):
        self.sample_ax = sample_ax
        self.draw_ax = draw_ax
        self.lx = sample_ax.axhline(color='k')  # the horiz line
        self.ly = sample_ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = sample_ax.text(0, 0, '', va="bottom", ha="left")

    def mouse_down(self, event):
        global ref_img, cost_volume, depth, corr_pos_pred, H, W
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        # draw probability
        pr_cost_volume = depth[:, int(y), int(x)]
        cost_volume_xs = np.arange(0, pr_cost_volume.shape[0])
        
        
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        
        xx, yy = corr_pos_pred[int(y)][int(x)]

        self.txt.set_text('x=%1.1f, y=%1.1f depth=%.5f\nCorr xx=%d, yy=%d' % (x, y, np.max(pr_cost_volume), xx, yy))
#         self.txt.set_text('x=%1.1f, y=%1.1f depth=%.5f' % (x, y, np.max(pr_cost_volume)))
        self.sample_ax.figure.canvas.draw()
    
        self.draw_ax.clear()
        self.draw_ax.plot(cost_volume_xs[1:-1], pr_cost_volume[1:-1], color='#fea83a', label='deep feature matching')
        
#         self.draw_ax.yaxis.set_major_formatter(StrMethodFormatter('%.1f'))
        self.draw_ax.set_yscale('log')
#         self.draw_ax.set_ylabel(r'probability (log) $\times 10^{-2}$')
#         self.draw_ax.set_ylabel(r'probability (log)')
        self.draw_ax.tick_params(bottom=False, top=True)
        self.draw_ax.tick_params(labelbottom=False, labeltop=True)
        self.draw_ax.figure.canvas.draw()
#         normalized_pr_cost_volume = (pr_cost_volume - pr_cost_volume.min())
#         normalized_pr_cost_volume = normalized_pr_cost_volume / normalized_pr_cost_volume.max()
        
        axs[1, 0].clear()
        axs[1, 0].imshow(img2)
        for i in range(1, 63):
            pos = sample_locs[i][int(y)][int(x)]
            depos = de_normalize(pos, H, W)
#             circ = Circle((int(depos[0]), int(depos[1])),1,color='r', alpha=normalized_pr_cost_volume[i])
            circ = Circle((int(depos[0]), int(depos[1])),1,color='y', alpha=0.5)
            axs[1, 0].add_patch(circ)
        
        
        circ = Circle((xx, yy),2,color='r')
        axs[1, 0].add_patch(circ)
        
        ref_point = ref_img[int(y), int(x), :]
        color_score = []
        max_score_id = None
        max_score = -1
        for i in range(0, 64):
            if (y > sample_locs.shape[1] or x > sample_locs.shape[0]):
                axs[1, 1].plot(cost_volume_xs[1:-1], pr_cost_volume[1:-1], color='#fea83a')
            pos = sample_locs[i][int(y)][int(x)]
            depos = de_normalize(pos, H, W)
            source_point = img2[int(depos[1]), int(depos[0]), :]
            color_score.append(np.dot(ref_point, source_point))
            if color_score[-1] > max_score:
                max_score = color_score[-1]
                max_score_id = (int(depos[0]), int(depos[1]))
                
                

        circ = Circle(max_score_id, 2, color='b')
        axs[1, 0].add_patch(circ)
        color_score = color_score / sum(color_score)

        axs[1, 1].clear()
        axs[1, 1]=self.draw_ax.twinx()
        axs[1, 1].set_yscale('log', basey=10) 
#         axs[1, 1].tick_params(axis='y', direction='inout')
        axs[1, 1].plot(cost_volume_xs[1:-1], color_score[1:-1], color='b', label='rgb matching')
        
        plt.savefig('output1.png',transparent=True)

        
fig, axs = plt.subplots(2, 2,  squeeze=True, figsize=(12, 8))
cus = Cursor(axs[0,0], axs[0,1])
axs[0,0].imshow(ref_img)


max_score = np.log(np.max(depth, axis=0))
print(max_score.shape)
print(max_score)
max_score = (max_score - max_score.min())
max_score = max_score / max_score.max()

fig.canvas.mpl_connect('button_press_event', cus.mouse_down)