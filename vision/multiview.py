import numpy as np

import torch
import warnings

from core import cfg

def camera_center(KRT, engine='numpy'):
    """
    Args:
        KRT: 3x4
    """
    if engine == 'numpy':
        invA = np.linalg.inv(KRT[:, :3])
        return (-(np.dot(invA, KRT[:, 3]))), invA
    elif engine == 'torch':
        invA = torch.inverse(KRT[..., :3])
        center = -torch.matmul(invA, KRT[..., 3, None])
        out = torch.ones([center.shape[0], 4, 1], dtype=KRT.dtype, device=center.device)
        out[..., :3, :] = center
        return out, invA
    else:
        raise

def normalize(pts, H, W):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    if cfg.EPIPOLAR.USE_CORRECT_NORMALIZE:
        pts[..., 0] = -1. + 2. * pts[..., 0] / (W - 1)
        pts[..., 1] = -1. + 2. * pts[..., 1] / (H - 1)
    else:
        warnings.warn('using inaccurate normalize func', DeprecationWarning)
        pts[..., 0] = -1. + 2. * (pts[..., 0] + 0.5) / W
        pts[..., 1] = -1. + 2. * (pts[..., 1] + 0.5) / H
    return pts

def de_normalize(pts, H, W, engine='numpy'):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    if cfg.EPIPOLAR.USE_CORRECT_NORMALIZE:
        if engine == 'torch':
            WH = torch.tensor([W, H], dtype=pts.dtype, device=pts.device)
            return (pts + 1) * (WH - 1) / 2.
        pts[..., 0] = (pts[..., 0] + 1) * (W - 1) / 2.
        pts[..., 1] = (pts[..., 1] + 1) * (H - 1) / 2.
    else:
        warnings.warn('using inaccurate denormalize func', DeprecationWarning)
        if engine == 'torch':
            WH = torch.tensor([W, H], dtype=pts.dtype, device=pts.device)
            return (pts + 1) * WH / 2. - .5
        pts[..., 0] = (pts[..., 0] + 1) * W / 2. - 0.5
        pts[..., 1] = (pts[..., 1] + 1) * H / 2. - 0.5
    return pts

def neighbor_cameras(d):
    """
    Args:
        d: dictionary of KRT
    Return:
        rank: dictionary of tuple of (list of cameras sorted by distance, distance)
    """
    cams = list(d.keys())
    centers = {}
    for k0, v0 in d.items():
        center, _ = camera_center(v0)
        centers[k0] = center
        assert len(center) == 3
    rank = {}
    # cntmeandist = 0
    for k0, v0 in centers.items():
        dist = {}
        for k1, v1 in centers.items():
            dist[k1] = np.linalg.norm(v0 - v1)
        r = sorted(cams, key=lambda x: dist[x])
        sorteddist = np.array(sorted(dist.values()))
        assert r[0] == k0
        # exclude the camera its own
        rank[k0] = (r[1:], sorteddist[1:])
    return rank

def findFundamentalMat(P1, P2, engine='torch'):
    """
    Args:
        P1, P2: N x 3 x 4
    """
    if engine == 'torch':
        #TODO: pinv not the same as numpy
        print('P1', P1)
        print('P2', P2)
        # N x 4 x 3
        P1 = P1.view(-1, 3, 4)
        P2 = P2.view(-1, 3, 4)
        P1t = P1.transpose(1, 2)
        # N x 4 x 3
        # P1inv = torch.matmul(P1t, torch.inverse(torch.matmul(P1, P1t)))
        # P1inv = P1t @ torch.inverse(torch.matmul(P1, P1t))
        P1inv = torch.stack([i.pinverse() for i in P1])
        print('torch.inverse(torch.matmul(P1, P1t))', torch.inverse(torch.matmul(P1, P1t)))
        print('P1inv', P1inv)
        # N x 3 x 3
        P2P1inv = torch.matmul(P2, P1inv)
        print('P2P1inv', P2P1inv)        
        # N x 4
        C, _ = camera_center(P1, engine='torch')
        # N x 3 x 1
        # e2 = torch.matmul(P2, C).view(-1, 3, 1).expand(-1, -1, 3)
        # N x 3 x 3
        # F = torch.cross(e2, P2P1inv, dim=1)

        e2 = torch.matmul(P2, C).squeeze()
        print(e2)
        e2cross = torch.zeros_like(P2P1inv)
        e2cross[..., 0, 1] = -e2[..., 2]
        e2cross[..., 0, 2] = e2[..., 1]
        e2cross[..., 1, 2] = -e2[..., 0]
        e2cross[..., 1, 0] = e2[..., 2]
        e2cross[..., 2, 0] = -e2[..., 1]
        e2cross[..., 2, 1] = e2[..., 0]
        F = torch.matmul(e2cross, P2P1inv)
        return F.squeeze()
    else:
        print('P1', P1)
        print('P2', P2)
        #3x4
        # P1inv = np.linalg.pinv(P1)
        P1inv = P1.T @ np.linalg.inv(P1 @ P1.T)
        print('np.linalg.inv(P1 @ P1.T)', np.linalg.inv(P1 @ P1.T))
        print('P1inv', P1inv)
        P2P1inv = P2 @ P1inv
        print('P2P1inv', P2P1inv)        
        C, _ = camera_center(P1)
        C_ = np.ones(4, dtype=C.dtype)
        C_[:3] = C
        e2 = P2 @ C_
        print(e2)
        # F = np.cross(e2, P2P1inv)
        F = crossmat(e2) @ P2P1inv
        # print('P1', P1)
        # print('P2', P2)
        # print('e2', e2)
        # print('P2P1inv', P2P1inv)
        # print('e2 @ F', e2 @ F)
        return F

def crossmat(vec, engine='numpy'):
    return np.array([[0, -vec[2], vec[1]], 
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]], dtype=vec.dtype)

def pix2coord(x, downsample):
    """convert pixels indices to real coordinates for 3D 2D projection
    """
    return x * downsample + downsample / 2.0 - 0.5

def coord2pix(y, downsample):
    """convert real coordinates to pixels indices for 3D 2D projection
    """
    # x * downsample + downsample / 2.0 - 0.5 = y
    return (y + 0.5 - downsample / 2.0) / downsample

def project_point_radial(x, f, c, k, p, R=None, T=None):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    if R is None or T is None:
        xcam = x.T
    else:
        xcam = R.dot(x.T - T)
    y = xcam[:2] / xcam[2]

    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = 2 * p[0] * y[1] + 2 * p[1] * y[0]
    y = y * np.tile(radial + tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = (f * y) + c
    return ypixel.T

if __name__ == '__main__':
    a = torch.ones((2, 3,4))
    b = torch.ones((2, 3,4))
    a.random_()
    b.random_()
    # findFundamentalMat(a,b)
    R = np.array([[-0.91536173,  0.40180837,  0.02574754],
                [ 0.05154812,  0.18037357, -0.98224649],
                [-0.39931903, -0.89778361, -0.18581953]])
    print('crossmat')
    print(crossmat(np.arange(1,4.)))
    P1 = R.dot(np.ones((3,4)))
    P2 = np.ones((3,4))
    print(findFundamentalMat(P1, P2, engine='numpy'))
    print(findFundamentalMat(torch.tensor(P1, dtype=torch.double), torch.tensor(P2, dtype=torch.double)))
