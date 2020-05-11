import os.path, sys, re, cv2, glob, numpy as np
import os.path as osp
from tqdm import tqdm
from IPython import embed
import scipy
import matplotlib.pyplot as plt
from skimage.transform import resize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import auc
from matplotlib.patches import Circle

import torch

# from .ipv_vis import *
from vision.triangulation import triangulate
from vision.multiview import pix2coord, coord2pix
from core import cfg
from vision.multiview import de_normalize 
from vision.visualizer_human import draw_2d_pose
from vision.visualizer_hand import plot_hand_3d


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


class Cursor_for_epipolar_line(object):
    def __init__(self, sample_ax, draw_ax, sample_locs, H, W, axs, img2, outs):
        self.sample_ax = sample_ax
        self.draw_ax = draw_ax
        self.lx = sample_ax.axhline(color='k')  # the horiz line
        self.ly = sample_ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = sample_ax.text(0, 0, '', va="bottom", ha="left")
        self.sample_locs = sample_locs
        self.H = H
        self.W = W
        self.axs = axs
        self.img2 = img2
        self.outs = outs

    def mouse_down(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        # pr_cost_volume = self.depth[:, int(y), int(x)]
        # cost_volume_xs = np.arange(0, pr_cost_volume.shape[0])
        # xx, yy = self.corr_pos_pred[int(y)][int(x)]

        self.txt.set_text('x=%1.1f, y=%1.1f' % (x, y))
        self.sample_ax.figure.canvas.draw()
        for i in self.draw_ax:
            i.clear()
            i.figure.canvas.draw()
        
        self.axs[1, 0].clear()
        self.axs[1, 0].imshow(self.img2)
        inty, intx = int(y+0.5), int(x+0.5)
        print(self.sample_locs[:, inty, intx])
        _, _, _, debugsample_locs, intersections, mask, valid_intersections, start, vec = self.outs
        print(intx, inty)
        print('debugsample_locs', debugsample_locs[:, 0, inty, intx])
        print('intersections', intersections.view(-1, 64, 64, 4, 2)[0, inty, intx])
        print('mask', mask.view(-1, 64, 64, 4)[0, inty, intx])
        print('valid_intersections', valid_intersections.view(-1, 64, 64, 2, 2)[0, inty, intx])
        print('start', start.view(-1, 64, 64, 2)[0, inty, intx])
        print('vec', vec.view(-1, 64, 64, 2)[0, inty, intx])
        for i in range(64):
            # pos = self.sample_locs[i][int(y+0.5)][int(x+0.5)]
            pos = debugsample_locs[i, 0, inty, intx].cpu().numpy().copy()
            depos = de_normalize(pos, self.H, self.W)
            # circ = Circle((int(depos[0]), int(depos[1])),1,color='b', alpha=0.5)
            circ = Circle((depos[0], depos[1]), 1 , color='b', alpha=0.5)
            self.axs[1, 0].add_patch(circ)
        # circ = Circle((xx, yy),2,color='r')
        self.axs[1, 0].add_patch(circ)
        plt.show()



class Cursor_for_corrspondence(object):
    def __init__(self, sample_ax, draw_ax, depth, corr_pos_pred, sample_locs, H, W):
        self.sample_ax = sample_ax
        self.draw_ax = draw_ax
        self.lx = sample_ax.axhline(color='k')  # the horiz line
        self.ly = sample_ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = sample_ax.text(0, 0, '', va="bottom", ha="left")
        self.depth = depth
        self.corr_pos_pred = corr_pos_pred
        self.sample_locs = sample_locs
        self.H = H
        self.W = W

    def mouse_down(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        pr_cost_volume = self.depth[:, int(y), int(x)]
        cost_volume_xs = np.arange(0, pr_cost_volume.shape[0])
        xx, yy = self.corr_pos_pred[int(y)][int(x)]

        self.txt.set_text('x=%1.1f, y=%1.1f depth=%.5f\nCorr xx=%d, yy=%d' % (x, y, np.max(pr_cost_volume), xx, yy))
        self.sample_ax.figure.canvas.draw()
        for i in self.draw_ax:
            i.clear()
            i.figure.canvas.draw()
        
        axs[1, 0].clear()
        axs[1, 0].imshow(img2)
        for i in range(64):
            pos = sample_locs[i][int(y)][int(x)]
            depos = de_normalize(pos, H, W)
            circ = Circle((int(depos[0]), int(depos[1])),1,color='b', alpha=0.5)
            axs[1, 0].add_patch(circ)
        circ = Circle((xx, yy),2,color='r')
        axs[1, 0].add_patch(circ)
        plt.show()


def toimg(x):
    return x.squeeze().numpy().transpose([1,2,0])

def de_transform(img):
    img[..., 0, :, :] = img[..., 0, :, :] * 0.229 + 0.485
    img[..., 1, :, :] = img[..., 1, :, :] * 0.224 + 0.456
    img[..., 2, :, :] = img[..., 2, :, :] * 0.225 + 0.406
    return img

def draw_auc(predictions, pck, auc_path):

    max_threshold = 20
    thresholds = np.linspace(0, max_threshold, num=20)
    pck = np.sum(pck, axis=0)
    
    auc_value = auc(thresholds, pck) / max_threshold
    print('AUC: ', auc_value)

    plt.plot(thresholds, pck, 'r')
    plt.axis([0, 20, 0, 1])
    plt.savefig(auc_path)
    plt.show()

def get_point_cloud(img1, img2, KRT1, KRT2, RT1, RT2, corr_pos, score):
    """
        KRT: 
        corr_pos: feat_h x feat_w x 2
        score:    sample_size x feat_h x feat_w
    """

    y = np.arange(0, img1.shape[0]) # 128
    x = np.arange(0, img1.shape[1]) # 84

    grid_x, grid_y = np.meshgrid(x, y)

    grid_y = pix2coord(grid_y, cfg.BACKBONE.DOWNSAMPLE)
    grid_y = grid_y * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE
    grid_x = pix2coord(grid_x, cfg.BACKBONE.DOWNSAMPLE)
    grid_x = grid_x * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE
    # 2668 * 4076

    grid_corr = pix2coord(corr_pos, cfg.BACKBONE.DOWNSAMPLE)
    grid_corr = grid_corr * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE

    grid = np.stack((grid_x, grid_y))
    grid = grid.reshape(2, -1)
    grid_corr = grid_corr.reshape(-1, 2).transpose()


    from scipy.misc import imresize
    sample_size, fh, fw = score.shape
    resized_img2 = imresize(img2, (fh, fw))

    max_score = np.max(score.reshape(sample_size, -1), axis=0).reshape(fh, fw)
    select_pos1 = max_score > 0.02
    print('->', np.sum(select_pos1))
    select_pos2 = np.sum(resized_img2, axis=2) > 20
    print('->',np.sum(select_pos2))

    select_pos3 = np.sum(corr_pos, axis=2) > -50
    print('->',np.sum(select_pos2))

    select_pos = np.logical_and(select_pos3, select_pos2).reshape(-1)
    # select_pos = select_pos3
    print('-->',np.sum(select_pos))

    select_pos = select_pos.reshape(-1)
    select_img_point = resized_img2.reshape(fh*fw, 3)[select_pos, :]
    print(select_pos.shape)
    print('total pos', sum(select_pos))

    p3D = cv2.triangulatePoints(KRT2, KRT1, grid_corr[:,select_pos], grid[:,select_pos])
    # p3D = cv2.triangulatePoints(KRT2, KRT1, grid_corr, grid)

    # depth = np.ones((fh, fw)) * np.min((KRT1@p3D)[2, :])
    depth = np.ones((fh, fw)) * np.max((KRT1@p3D)[2, :])


    cnt = 0
    for i in range(fh):
        for j in range(fw):
            if not select_pos[i*fw+j]:
                continue
            p_homo = (KRT1 @ p3D[:, cnt]) 
            p = p_homo / p_homo[2]
            depth[int(coord2pix(p[1], 32)), int(coord2pix(p[0], 32))] = p_homo[2]
            cnt += 1

    p3D /= p3D[3]
    p3D = p3D[:3].squeeze()

    depth = (depth - depth.min()) / (depth.max() - depth.min()) + 1
    depth = np.log(depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    #######vis
    fig = plt.figure(1)
    ax1_1 = fig.add_subplot(331)
    ax1_1.imshow(img1)
    ax1_2 = fig.add_subplot(332)
    ax1_2.imshow(img2)

    w = corr_pos[:, :, 0]
    w = (w - w.min()) / (w.max() - w.min())
    ax1_1 = fig.add_subplot(334)
    ax1_1.imshow(w)

    w = corr_pos[:, :, 1]
    w = (w - w.min()) / (w.max() - w.min())
    ax1_1 = fig.add_subplot(335)
    ax1_1.imshow(w)

    # w1 = corr_pos[:, :, 0]
    # w1 = (w1 - w1.min()) / (w1.max() - w1.min())
    # w2 = corr_pos[:, :, 1]
    # w2 = (w2 - w2.min()) / (w2.max() - w2.min())
    # W = np.stack([w1, w2, np.ones(w2.shape)], axis=0)
    # ax2_1 = fig.add_subplot(336)
    # ax2_1.imshow(W.transpose(1,2,0))

    ax1_1 = fig.add_subplot(336)
    ax1_1.imshow(depth)


    w = select_pos1.reshape(fh,fw)
    # w = (w - w.min()) / (w.max() - w.min())
    ax2_1 = fig.add_subplot(337)
    ax2_1.imshow(w)

    w = select_pos2.reshape(fh,fw)
    # w = (w - w.min()) / (w.max() - w.min())
    ax2_1 = fig.add_subplot(338)
    ax2_1.imshow(w)

    w = select_pos.reshape(fh,fw)
    # w = (w - w.min()) / (w.max() - w.min())
    ax2_1 = fig.add_subplot(339)
    ax2_1.imshow(w)

    ####### end vis

    # w = select_img_point[:, :10000].reshape(-1, 100, 100).transpose(1,2,0)
    # w = (w - w.min()) / (w.max() - w.min())
    # ax2_1 = fig.add_subplot(326)
    # ax2_1.imshow(w)


    plt.show()
    return p3D, select_img_point

def visualization(cfg):
    if cfg.VIS.POINTCLOUD and 'h36m' not in cfg.OUTPUT_DIR:
        output_dir = cfg.OUTPUT_DIR
        dataset_names = cfg.DATASETS.TEST
        predictions = torch.load(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "predictions.pth"))
        print(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "predictions.pth"))

        cnt = 0
        # for inputs, pred in predictions:
        while True:
            inputs, pred =  predictions[cnt]
            heatmap = inputs.get('heatmap')
            points2d = inputs.get('points-2d')
            KRT = inputs.get('KRT')[0]
            RT = inputs.get('RT')[0]

            image_path = inputs.get('img-path')
            print('image path:', image_path)
            img = resize(plt.imread(image_path), (128, 84, 3))

            other_KRT = inputs.get('other_KRT')[0]
            other_RT = inputs.get('other_RT')[0]
            other_image_path = inputs.get('other_img_path')[0]
            print('other image path', other_image_path)
            other_img = resize(plt.imread(other_image_path), (128, 84, 3))

            heatmap_pred = pred.get('heatmap_pred')
            score_pred = pred.get('score_pred')
            corr_pos_pred = pred.get('corr_pos')
            sim = pred.get('depth')

            import pdb; pdb.set_trace()

            # p3D, img_pt = get_point_cloud(img, other_img, KRT, other_KRT, RT, other_RT, corr_pos_pred, sim)
            output = {
                # 'p3D': p3D,
                # 'img_pt': img_pt,
                'img1': img,
                'img2' : other_img,
                'img1_path': image_path,
                'img2_path': other_image_path,
                'RT'   : RT,
                'other_RT': other_RT,
                'corr_pos_pred': corr_pos_pred,
                'depth': sim,
            }
            if 'sample_locs' in pred:
                sample_locs = pred.get('sample_locs')
                output['sample_locs'] = sample_locs
            else:
                print('No sample_locs!!!!!')
            import pickle
            with open('baseline_' + "output_{:d}.pkl".format(cnt),"wb") as f:
                pickle.dump(output, f)
            print('saved! to ', 'baseline_' + "output_{:d}.pkl".format(cnt))
            cnt += 1
            # break
            # ipv_prepare(ipv)
            # ipv_draw_point_cloud(ipv, p3D, colors=img_pt, pt_size=1)
            # ipv.xyzlim(500)
            # ipv.show()

    if cfg.VIS.POINTCLOUD and 'h36m' in cfg.OUTPUT_DIR:
        output_dir = cfg.OUTPUT_DIR
        dataset_names = cfg.DATASETS.TEST

        baseline = "baseline" in cfg.VIS.SAVE_PRED_NAME
        name = "_baseline" if baseline else ""
        predictions = torch.load(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "predictions"+name+".pth"))
        print(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "predictions"+name+".pth"))

        cnt = 0
        # for inputs, pred in predictions:
        while True:
            inputs, pred =  predictions[cnt]
            print('input keys:')
            print(inputs.keys())

            print('pred keys:')
            print(pred.keys())

            heatmap = inputs.get('heatmap')
            other_heatmap = inputs.get('other_heatmap')
            points2d = inputs.get('points-2d')

            KRT = inputs.get('KRT')[0]
            camera = inputs.get('camera')
            other_camera = inputs.get('other_camera')


            image_path = inputs.get('img-path')[0]
            print(image_path)
            # image_path = 'images.zip@'
            image_file = osp.join("datasets", 'h36m',  'images.zip@', 'images',
                              image_path)
            # from utils import zipreader
            # data_numpy = zipreader.imread(
            #     image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            
            # img = data_numpy[:1000]
            # assert img.shape == (1000, 1000, 3), img.shape
            img = inputs.get('img')

            other_KRT = inputs.get('other_KRT')[0]
            # other_RT = inputs.get('other_RT')[0]
            other_image_path = inputs.get('other_img-path')[0]
            print('other image path', other_image_path)
            other_image_file = osp.join("datasets", 'h36m',  'images.zip@', 'images',
                              other_image_path)
            other_img = inputs.get('other_img')

            heatmap_pred = pred.get('heatmap_pred')
            score_pred = pred.get('score_pred')
            corr_pos_pred = pred.get('corr_pos')
            sim = pred.get('depth')
            batch_locs = pred.get('batch_locs')


            # p3D, img_pt = get_point_cloud(img, other_img, KRT, other_KRT, RT, other_RT, corr_pos_pred, sim)
            output = {
                # 'p3D': p3D,
                # 'img_pt': img_pt,
                'img1': img,
                'img2' : other_img,
                'img1_path': image_file,
                'img2_path': other_image_file,
                # 'RT'   : RT,
                # 'other_RT': other_RT,
                'heatmap': heatmap,
                'other_heatmap': other_heatmap,
                'points-2d': points2d,
                'corr_pos_pred': corr_pos_pred,
                'depth': sim,
                'heatmap_pred': heatmap_pred,
                'batch_locs': batch_locs,
                'camera': camera,
                'other_camera': other_camera,
            }
            if 'sample_locs' in pred:
                sample_locs = pred.get('sample_locs')
                output['sample_locs'] = sample_locs
            else:
                print('No sample_locs!!!!!')
            import pickle
            with open(cfg.OUTPUT_DIR + "/visualizations/h36m/output{}_{:d}.pkl".format(name, cnt),"wb") as f:
                pickle.dump(output,f)
            print('saved!')
            cnt += 1

# depth = output['depth']
# corr_pos_pred = output['corr_pos_pred']
# sample_locs = output['sample_locs']

    if cfg.EPIPOLAR.VIS:
        if 'h36m' in cfg.OUTPUT_DIR:
            from data.build import make_data_loader
            if cfg.VIS.MULTIVIEWH36M:
                data_loader = make_data_loader(cfg, is_train=True, force_shuffle=True)
            elif cfg.VIS.H36M:
                from data.datasets.joints_dataset import JointsDataset
                from data.datasets.multiview_h36m import MultiViewH36M
                data_loader = MultiViewH36M('datasets', 'validation', True)
                print(len(data_loader))
                for i in tqdm(range(len(data_loader))):
                    data_loader.__getitem__(i)
                data_loader = make_data_loader(cfg, is_train=False)[0]
                # data_loader = make_data_loader(cfg, is_train=True, force_shuffle=True)
                # data_loader = make_data_loader(cfg, is_train=False, force_shuffle=True)[0]

            # for idx, batchdata in enumerate(tqdm(data_loader)):
                if not cfg.VIS.MULTIVIEWH36M and not cfg.VIS.H36M:
                    cpu = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    from modeling.layers.epipolar import Epipolar
                    imgmodel = Epipolar()
                    debugmodel = Epipolar(debug=True)
                    KRT0 = batchdata['KRT'].squeeze()[None, 0]
                    KRT1 = batchdata['other_KRT'].squeeze()[None, 0]
                    # batchdata['img']: 1 x 4 x 3 x 256 x 256
                    input_img = batchdata['img'].squeeze()[None, 0, :, ::4, ::4]
                    input_other_img = batchdata['other_img'].squeeze()[None, 0, :, ::4, ::4]
                    outs = debugmodel(input_img, input_other_img, KRT0, KRT1)
                    H, W = input_img.shape[-2:]
                    print(H, W)
                    orig_img = de_transform(cpu(batchdata['img'].squeeze()[None, ...])[0][0])
                    orig_other_img = de_transform(cpu(batchdata['other_img'].squeeze()[None, ...])[0][0])
                    # outs = imgmodel(batchdata['heatmap'][:, 0], batchdata['heatmap'][:, 1], batchdata['KRT'][:, 0], batchdata['other_KRT'][:, 1])
                    out, sample_locs = imgmodel.imgforward_withdepth(input_img, input_other_img, KRT0, KRT1, outs[2][0])
                    if not cfg.VIS.CURSOR:
                        # show_img = de_transform(cpu(batchdata['img'][:, 0, :, ::4, ::4])[0][0])
                        # show_other_img = de_transform(cpu(batchdata['other_img'][:, 0, :, ::4, ::4])[0][0])
                        fig = plt.figure(1)
                        ax1 = fig.add_subplot(231)
                        ax2 = fig.add_subplot(232)
                        ax3 = fig.add_subplot(233)
                        ax4 = fig.add_subplot(234)
                        ax5 = fig.add_subplot(235)
                        ax1.imshow(orig_img[::-1].transpose((1,2,0)))
                        ax2.imshow(orig_other_img[::-1].transpose((1,2,0)))
                        ax3.imshow(cpu(batchdata['heatmap'])[0][0].sum(0))
                        ax4.imshow(cpu(batchdata['other_heatmap'])[0][0].sum(0))
                        # ax5.imshow(cpu(outs[0])[0].sum(0))
                        print(out.shape)
                        out_img = de_transform(cpu(out)[0, ::-1].transpose((1,2,0)))
                        ax5.imshow(out_img)
                        plt.show()
                    else:
                        print(sample_locs.shape) # 64 x 1 x H x W x 2
                        sample_locs = sample_locs[:, 0, :, :, :]
                        # import pdb; pdb.set_trace()

                        fig, axs = plt.subplots(2, 2)
                        cus = Cursor_for_epipolar_line(axs[0,0], [axs[0,1], axs[1,0], axs[1,1]], sample_locs, H, W, axs, \
                            cpu(input_other_img)[0, :, :, :][::-1].transpose((1,2,0)), outs)
                        axs[0, 0].imshow(cpu(input_img)[0, :, :, :][::-1].transpose((1,2,0)))
                        # prob_im = axs[1, 1].imshow(max_score)
                        fig.canvas.mpl_connect('button_press_event', cus.mouse_down)
                        plt.show()
        return 

    output_dir = cfg.OUTPUT_DIR
    dataset_names = cfg.DATASETS.TEST
    predictions = torch.load(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "predictions.pth"))
    pck = torch.load(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "pck.pth"))

    if cfg.VIS.AUC:
        auc_path = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_names[0], "auc.png")
        draw_auc(predictions, pck, auc_path)

    total = 0
    for inputs, pred in predictions:
        heatmap = inputs.get('heatmap')
        points2d = inputs.get('points-2d')
        hand_side = inputs.get('hand-side')
        img = inputs.get('img')
        can_3dpoints = inputs.get('can-points-3d')
        normed_3d = inputs.get('normed-points-3d')
        target_global = inputs.get('points-3d')
        rot_mat =  inputs.get('rotation')
        R_global = inputs.get('R')
        keypoint_scale = inputs.get('scale')
        visibility = inputs.get('visibility')
        unit = inputs.get('unit')
        image_path = inputs.get('img-path')

        can_pred = pred.get('can_pred')
        normed_pred = pred.get('normed_pred')
        heatmap_pred = pred.get('heatmap_pred')

        im = plt.imread(image_path)
        image = np.array(im, dtype=np.int)
        if cfg.DATASETS.TASK == 'keypoint':
            fig = plt.figure(1)
            ax1 = fig.add_subplot(331)
            ax2 = fig.add_subplot(332)
            ax3 = fig.add_subplot(333)
            #ax1.imshow(image)
            print(heatmap.min(), heatmap.max())
            print(heatmap_pred.min(), heatmap_pred.max())
            ax2.imshow(heatmap.sum(0).T)
            ax3.imshow(heatmap_pred.sum(0).T)
        else:
            total += 1

            visibility = visibility.squeeze()[..., None]
            can_3dpoints = can_3dpoints * visibility
            can_pred = can_pred * visibility

            normed_3d = normed_3d * visibility
            normed_pred = normed_pred * visibility

            delta = normed_pred - normed_3d
            print(delta)
            print('L1 err = ', np.abs(delta).sum())
            print('L2 err = ', ((delta**2).sum(-1)**0.5).mean())

            fig = plt.figure(1)
            ax1_1 = fig.add_subplot(331)
            ax1_2 = fig.add_subplot(332)
            #ax1_3 = fig.add_subplot(333)

            #ax2 = fig.add_subplot(222)
            ax2_1 = fig.add_subplot(334, projection='3d')
            ax2_2 = fig.add_subplot(335, projection='3d')
            ax2_3 = fig.add_subplot(336, projection='3d')

            ax3_1 = fig.add_subplot(337, projection='3d')
            ax3_2 = fig.add_subplot(338, projection='3d')
            ax3_3 = fig.add_subplot(333, projection='3d')
            
            ax1_1.imshow(image)
            ax1_2.imshow(image)
            #ax1_3.imshow(image)
            
            #ax2.imshow(image)

            plot_hand_3d(can_3dpoints, visibility, ax2_1)
            ax2_1.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view

            plot_hand_3d(can_pred, visibility, ax2_2)
            ax2_2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view

            plot_hand_3d(can_3dpoints, visibility, ax2_3)
            plot_hand_3d(can_pred, visibility, ax2_3)
            ax2_3.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            # ax3.set_xlim([-3, 3])
            # ax3.set_ylim([-3, 3])
            # ax3.set_zlim([-3, 3])
            


            plot_hand_3d(normed_3d, visibility, ax3_1)
            ax3_1.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view

            plot_hand_3d(normed_pred, visibility, ax3_2)
            ax3_2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view

            plot_hand_3d(normed_3d, visibility, ax3_3)
            plot_hand_3d(normed_pred, visibility, ax3_3)
            ax3_3.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            # ax3.set_xlim([-3, 3])
            # ax3.set_ylim([-3, 3])
            # ax3.set_zlim([-3, 3])
            
        plt.show()
        print("show")

