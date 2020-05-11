
import torch
from core import cfg

def EPEmean(out, target, keypoint_vis=None, keypoint_scale=1., unit=1):
    """
    mean error per endpoint in millonmeters
    inputs:
        out: Nx21x3
        target: Nx21x3
    """
    if unit is None:
        unit = 1
    if keypoint_scale is None:
        keypoint_scale = 1.
    MAX_DIST = cfg.TEST.EPEMEAN_MAX_DIST

    with torch.no_grad():
        try:
            err = ((out - target)**2).sum(-1)
        except Exception as e: 
            print(e)
            err = ((out.double() - target.double())**2).sum(-1)
            # print('out', out)
            # print('target', target)
        try:
            err = (err**0.5) * keypoint_scale * unit
        except:
            err = (err**0.5) * keypoint_scale[:, None] * unit
        # print('out')
        # print(out)
        # print('target')
        # print(target)
        # print(err[0])
        # print(err[err > MAX_DIST])
        err[err > MAX_DIST] = MAX_DIST
        perjointerr = err.clone()
        if keypoint_vis is not None:
            try:
                perjointerr[~keypoint_vis.squeeze(-1)] = 0.
            except:
                perjointerr[~keypoint_vis.byte().squeeze(-1)] = 0.
            # err = err[keypoint_vis.byte().squeeze(-1)]
        # print(err.max())
        # print(err[err > MAX_DIST])
        return err.mean(), perjointerr[0]

def EPEmean_gt(target, rot_mat, coord_xyz_rel_normed, side, keypoint_vis, keypoint_scale=1.):
    """ put this in model.py
    sanity check passed! (check whether gt target * rot matches gt normed)
    metric_dict['EPEmean_gt'] = EPEmean_gt(coord_xyz_can, rot_mat +0.1, coord_xyz_rel_normed, hand_side, keypoint_vis, keypoint_scale)
    """
    with torch.no_grad():
        target_mirrored = torch.stack([
                target[:, :, 0],
                target[:, :, 1],
               -target[:, :, 2]], 2)
        cond_right = torch.eq(side, torch.ones((1,)).type_as(side))
        target_flip = torch.where(cond_right.view(-1,1,1), target_mirrored, target)
        normed_gt = torch.matmul(target_flip, rot_mat)
    return EPEmean(normed_gt, coord_xyz_rel_normed, keypoint_vis, keypoint_scale)

def EPEmean_multiview_gt(out, target, keypoint_vis=None, keypoint_scale=1., unit=1):
    # sanity check: select nearest 3D joint to GT
    with torch.no_grad():
        err, _ = (((out - target)**2).sum(-1)**0.5).min(1)
        err = err * keypoint_scale * unit
        if keypoint_vis is not None:
            err = err[keypoint_vis.byte()]
        return err.mean()
