import torch
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
from matplotlib.patches import Circle
import matplotlib.cm as cmap

class Cursor(object):
    def __init__(self, sample_ax, draw_ax):
        self.sample_ax = sample_ax
        self.draw_ax = draw_ax
        self.lx = sample_ax.axhline(color='k')  # the horiz line
        self.ly = sample_ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = sample_ax.text(0, 0, '', va="bottom", ha="left")

    def mouse_down(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.1f, y=%1.1f' % (x, y))
        self.sample_ax.figure.canvas.draw()
        for i in self.draw_ax:
            i.clear()
            i.figure.canvas.draw()
        
        self.sample_ax.imshow(ref_img)
        a, b, heatmap = heatmapat(x, y, weights[0])
        im1= self.draw_ax[1].imshow(heatmap, cmap=cmap.hot)
        self.draw_ax[1].set_title("%f~%f" % (a, b))
        a, b, heatmap = heatmapat(x, y, weights[1])
        im2= self.draw_ax[2].imshow(heatmap, cmap=cmap.hot)
        self.draw_ax[2].set_title("%f~%f" % (a, b))
        a, b, heatmap = heatmapat(x, y, weights[2])
        im3= self.draw_ax[3].imshow(heatmap, cmap=cmap.hot)
        self.draw_ax[3].set_title("%f~%f" % (a, b))
        # fig.colorbar(im2, ax=axs[0, 1])
        circ = Circle((x, y),2,color='r')
        axs[0, 0].add_patch(circ)
        plt.show()
        # plt.cla()
        # plt.close()
        # fig, axs = plt.subplots(2, 2,  squeeze=True, figsize=(12, 8))

def heatmapat(x, y, weights):
    heatmap = weights[int(x), int(y)]
    return heatmap.min(), heatmap.max(), (heatmap - weightsmin) / (weightsmax - weightsmin)

    # return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

parser = argparse.ArgumentParser(description="PyTorch Keypoints Training")
parser.add_argument(
    "--src",
    default="",
    help="source model",
    type=str,
)
args = parser.parse_args()

state_dict = torch.load(args.src)
#4096 x 4096
weights = []
weightsmin, weightsmax = 1000000, -1000000
for i in range(12):
    weight = state_dict['aggre_layer.aggre.%d.weight' % i].view(64,64,64,64).cpu().numpy()
    weightsmin = min(weight.min(), weightsmin)
    weightsmax = max(weight.max(), weightsmax)
    weights.append(weight)
print(weightsmin, weightsmax)
ref_img = np.zeros((64, 64, 3))
# img2 = np.zeros((100, 100, 3))
        
fig, axs = plt.subplots(2, 2,  squeeze=True, figsize=(12, 8))
cus = Cursor(axs[0,0], [axs[0,0], axs[0,1], axs[1,0], axs[1,1]])

fig.canvas.mpl_connect('button_press_event', cus.mouse_down)

plt.show()