import numpy as np
import math
import cv2
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

if __name__ != '__main__':
    from .multiview import camera_center, pix2coord, coord2pix
    from core import cfg

from .camera_model import CameraModel
from .multi_camera_system import MultiCameraSystem
from pymvg.ros_compat import sensor_msgs as sensor_msgs_compat

def toT(A):
    if isinstance(A, torch.FloatTensor):
        return A
    return torch.from_numpy(A)

def toV(A):
    return Variable(toT(A))

def pytTriangulateDLT_getuv(M, i, p):
    d = torch.mm(p, M[2, :].unsqueeze(0))
    d = M[i, :].expand( (p.shape[0], M.shape[1]) ) - d
    return d.unsqueeze(1)

def pytTriangulateDLT_multi(Ms, ps):
    ds = list()
    for u in range( Ms.size(0) ):
        ds.append( pytTriangulateDLT_getuv(Ms[u,:,:], 0, ps[u, :, 0:1]) )
        ds.append( pytTriangulateDLT_getuv(Ms[u,:,:], 1, ps[u, :, 1:2]) )
    Ab = torch.cat( ds, dim=1 )
    A = Ab[:, :, :3]
    At = torch.transpose(A, 1, 2)
    AtA = torch.matmul(At, A)
    tpinv = [t.inverse() for t in torch.unbind(AtA)]
    invAtA = torch.stack(tpinv)
    P = torch.matmul( invAtA, torch.matmul(At, -Ab[:, :, 3].unsqueeze(2)) )
    return P

def pytTriangulateDLT(M1, M2, p1, p2):
    d1 = pytTriangulateDLT_getuv(M1, 0, p1[:, 0:1])
    d2 = pytTriangulateDLT_getuv(M1, 1, p1[:, 1:2])
    d3 = pytTriangulateDLT_getuv(M2, 0, p2[:, 0:1])
    d4 = pytTriangulateDLT_getuv(M2, 1, p2[:, 1:2])
    Ab = torch.cat( (d1, d2, d3, d4), dim=1 )
    A = Ab[:, :, :3]
    At = torch.transpose(A, 1, 2)
    AtA = torch.matmul(At, A)
    tpinv = [t.inverse() for t in torch.unbind(AtA)]
    invAtA = torch.stack(tpinv)
    P = torch.matmul( invAtA, torch.matmul(At, -Ab[:, :, 3].unsqueeze(2)) )
    return P

def pytTriangulateNLR_calcGrad(M, p, Pt):
    m1P = torch.matmul(M[0, :3].unsqueeze(0), Pt) + M[0, 3]
    m2P = torch.matmul(M[1, :3].unsqueeze(0), Pt) + M[1, 3]
    m3P = torch.matmul(M[2, :3].unsqueeze(0), Pt) + M[2, 3]
    m3P_sq = m3P * m3P
    x = torch.div(m1P, m3P)
    y = torch.div(m2P, m3P)
    e_u = p[:, 0:1].unsqueeze(2) - x
    e_v = p[:, 1:2].unsqueeze(2) - y
    grad1 = -2 * e_u * torch.div(M[0,:3] * m3P - M[2,:3] * m1P, m3P_sq)
    grad2 = -2 * e_v * torch.div(M[1,:3] * m3P - M[2,:3] * m2P, m3P_sq)
    return grad1+grad2, e_u*e_u + e_v*e_v, torch.stack((x, y)).squeeze(2).squeeze(2)

def pytTriangulateNLR(M1, M2, p1, p2, Pt):
    Pt = Pt.clone()
    lr = 0.001
    for it in range(10000):
        grad1, l1, dc = pytTriangulateNLR_calcGrad(M1, p1, Pt)
        grad2, l2, dc = pytTriangulateNLR_calcGrad(M2, p2, Pt)
        grad = torch.transpose(grad1+grad2, 1, 2) * lr
        if grad.abs().max().data[0] < 1e-4:
            print("Non-linear refinement finished in %d iterations"%(it))
            break
        #print "%d %g %g %g %g"%(it, (l1+l2).data[0], Pt.data[0][0][0], Pt.data[0][1][0], Pt.data[0][2][0])
        Pt -= grad
    return Pt

def point2line(p3D, x1, x2):
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    d1 = x1 - p3D
    d2 = x2 - p3D
    d3 = x1 - x2
    #dist2 = np.linalg.norm( np.cross( d1, d2 ) ) / np.linalg.norm(d3)
    #expanded so it is faster..
    cro = (d1[1]*d2[2]-d1[2]*d2[1], d1[2]*d2[0]-d1[0]*d2[2], d1[0]*d2[1]-d1[1]*d2[0])
    return math.sqrt(cro[0]*cro[0] + cro[1]*cro[1] + cro[2]*cro[2]) / math.sqrt(d3[0]*d3[0] + d3[1]*d3[1] + d3[2]*d3[2])

RANSAC_ITER = 100

def triangulate(pts, KRTs, confs):
    """
    Args:
        pts: view x 42 x 2
        KRTs: view x 3 x 4
        confs: view x 42
    Return:
        42 x 3
    """
    if torch.is_tensor(pts): pts = pts.cpu().numpy()
    if torch.is_tensor(KRTs): KRTs = KRTs.cpu().numpy()
    if torch.is_tensor(confs): confs = confs.cpu().numpy()

    cam_centers = []
    invAs = []
    for M in KRTs:
        cam_center, invA = camera_center(M)
        cam_centers.append(cam_center)
        invAs.append(invA)
    cam_centers = np.array(cam_centers)
    invAs = np.array(invAs)
    # view x 3
    ret = []
    for cands, conf in zip(pts.transpose((1,0,2)), confs.T):
        best = 0
        bestconf = 0
        best3D = [0, 0, 0]
        selected_idx = conf > cfg.KEYPOINT.CONF_THRES
        if selected_idx.sum() <= 1:
            ret.append(best3D)
            continue
        cands = cands[selected_idx]
        KRT = KRTs[selected_idx]

        for _ in range(RANSAC_ITER):
            a = random.randint(0, len(cands)-1)
            b = random.randint(0, len(cands)-1)
            if a == b:
                continue

            p3D = cv2.triangulatePoints( KRT[a], KRT[b], cands[a], cands[b])
            p3D /= p3D[3]
            p3D = p3D[:3].squeeze()

            acc = 0
            for cand, cam_center, invA in zip(cands, cam_centers[selected_idx], invAs[selected_idx]):
                x1 = np.dot(invA, np.append(cand, 1)) + cam_center
                dist = point2line(p3D, x1, cam_center)
                if dist < cfg.KEYPOINT.RANSAC_THRES:
                    acc += 1

            if acc > best:
                best = acc
                best3D = p3D
        ret.append(best3D)
    return np.array(ret)

def cv2triangulate(KRT0, KRT1, pts0, pts1):
    p3D = cv2.triangulatePoints(KRT0, KRT1, pts0, pts1)
    p3D /= p3D[3]
    p3D = p3D[:3].squeeze()
    return p3D

def triangulate_refine(pts, KRTs, Ks, RTs, confs):
    """
    Args:
        pts: view x 42 x 2
        KRTs: view x 3 x 4
        confs: view x 42
    Return:
        42 x 3
    """
    camera_system = build_multi_camera_system(Ks, RTs)

    if torch.is_tensor(pts): pts = pts.cpu().numpy()
    if torch.is_tensor(KRTs): KRTs = KRTs.cpu().numpy()
    if torch.is_tensor(Ks): Ks = Ks.cpu().numpy()
    if torch.is_tensor(RTs): RTs = RTs.cpu().numpy()
    if torch.is_tensor(confs): confs = confs.cpu().numpy()
    assert len(confs.T) == pts.shape[1]

    cam_centers = []
    invAs = []
    for M in KRTs:
        cam_center, invA = camera_center(M)
        cam_centers.append(cam_center)
        invAs.append(invA)
    cam_centers = np.array(cam_centers)
    invAs = np.array(invAs)
    # view x 3
    ret = []
    for all_cands, conf in zip(pts.transpose((1,0,2)), confs.T):
        best = 0
        bestconf = 0
        best3D = [0, 0, 0]
        bestinliers = []
        selected_idx = conf > cfg.KEYPOINT.CONF_THRES
        if selected_idx.sum() <= 1:
            ret.append(best3D)
            continue
        cands = all_cands[selected_idx]
        KRT = KRTs[selected_idx]

        for _ in range(RANSAC_ITER):
            a = random.randint(0, len(cands)-1)
            b = random.randint(0, len(cands)-1)
            if a == b:
                continue

            p3D = cv2triangulate( KRT[a], KRT[b], cands[a], cands[b])

            acc = 0
            inliers = []
            for pid, cand, cam_center, invA in zip(np.where(selected_idx)[0], cands, cam_centers[selected_idx], invAs[selected_idx]):
                x1 = np.dot(invA, np.append(cand, 1)) + cam_center
                dist = point2line(p3D, x1, cam_center)
                if dist < cfg.KEYPOINT.RANSAC_THRES:
                    acc += 1
                    inliers.append(pid)


            if acc > best:
                bestinliers = inliers
                best = acc
                best3D = p3D

        points_2d_set = []
        if len(bestinliers) > 1:
            for j in bestinliers:
                points_2d = all_cands[j]
                points_2d_set.append((str(j), points_2d))
            best3D = triangulate_one_point(camera_system, points_2d_set).squeeze()
        ret.append(best3D)
    return np.array(ret)

def triangulate_epipolar(pts, KRTs, Ks, RTs, confs, corr_pos, otherKRTs, dlt=False):
    """
    Args:
        pts: view x 42 x 2
        KRTs: view x 3 x 4
        confs: view x 42
    Return:
        42 x 3
    """
    camera_system = build_multi_camera_system(Ks, RTs)

    if torch.is_tensor(pts): pts = pts.cpu().numpy()
    if torch.is_tensor(KRTs): KRTs = KRTs.cpu().numpy()
    if torch.is_tensor(otherKRTs): otherKRTs = otherKRTs.cpu().numpy()
    if torch.is_tensor(Ks): Ks = Ks.cpu().numpy()
    if torch.is_tensor(RTs): RTs = RTs.cpu().numpy()
    if torch.is_tensor(confs): confs = confs.cpu().numpy()
    if torch.is_tensor(corr_pos): corr_pos = corr_pos.cpu().numpy()
    assert len(confs.T) == pts.shape[1]
    assert 'epipolar' in cfg.BACKBONE.BODY

    cam_centers = []
    invAs = []
    for M in KRTs:
        cam_center, invA = camera_center(M)
        cam_centers.append(cam_center)
        invAs.append(invA)
    cam_centers = np.array(cam_centers)
    invAs = np.array(invAs)
    # view x 3
    ret = []
    for all_cands, conf in zip(pts.transpose((1,0,2)), confs.T):
        best = 0
        bestconf = 0
        best3D = [0, 0, 0]
        bestinliers = []
        selected_idx = conf > cfg.KEYPOINT.CONF_THRES
        if selected_idx.sum() == 0:
            print('correspondence by max point + epipolar')
            selected_idx = np.zeros_like(selected_idx)
            selected_idx[conf.argmax()] = True
        elif selected_idx.sum() == 1:
            print('correspondence by 1 point + epipolar')
        if selected_idx.sum() == 1:
            # 1 x 2
            cands = all_cands[selected_idx]
            pix_locs = coord2pix(cands / cfg.DATASETS.IMAGE_RESIZE / cfg.DATASETS.PREDICT_RESIZE,
                cfg.BACKBONE.DOWNSAMPLE).squeeze()
            other_locs = corr_pos[selected_idx].squeeze()[int(pix_locs[1]), int(pix_locs[0])]
            other_locs = pix2coord(other_locs, cfg.BACKBONE.DOWNSAMPLE)   # 128 -> 512
            other_locs = other_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE # ->1024->4096
            otherKRT = otherKRTs[selected_idx].squeeze()
            KRT = KRTs[selected_idx].squeeze()
            best3D = cv2triangulate(KRT, otherKRT, cands.squeeze(), other_locs)
            ret.append(best3D)
            continue
        
        cands = all_cands[selected_idx]
        KRT = KRTs[selected_idx]

        if dlt:
            points_2d_set = []
            for j in np.where(selected_idx)[0]:
                points_2d_set.append((str(j), all_cands[j]))
            best3D = triangulate_one_point(camera_system, points_2d_set).squeeze()
            ret.append(best3D)
            continue

        # too few points no need to ransac
        if pts.shape[1] < RANSAC_ITER**.5:
            for a in range(len(cands)):
                for b in range(len(cands)):
                    if a == b:
                        continue
                    p3D = cv2triangulate(KRT[a], KRT[b], cands[a], cands[b])
                    acc = 0
                    inliers = []
                    for pid, cand, cam_center, invA in zip(np.where(selected_idx)[0], cands, cam_centers[selected_idx], invAs[selected_idx]):
                        x1 = np.dot(invA, np.append(cand, 1)) + cam_center
                        dist = point2line(p3D, x1, cam_center)
                        if dist < cfg.KEYPOINT.RANSAC_THRES:
                            acc += 1
                            inliers.append(pid)
                    if acc > best:
                        bestinliers = inliers
                        best = acc
                        best3D = p3D
        else:
            for _ in range(RANSAC_ITER):
                a, b = np.random.choice(len(cands), 2, replace=False)
                # a = random.randint(0, len(cands)-1)
                # b = random.randint(0, len(cands)-1)
                # if a == b:
                #     continue
                p3D = cv2triangulate( KRT[a], KRT[b], cands[a], cands[b])
                acc = 0
                inliers = []
                for pid, cand, cam_center, invA in zip(np.where(selected_idx)[0], cands, cam_centers[selected_idx], invAs[selected_idx]):
                    x1 = np.dot(invA, np.append(cand, 1)) + cam_center
                    dist = point2line(p3D, x1, cam_center)
                    if dist < cfg.KEYPOINT.RANSAC_THRES:
                        acc += 1
                        inliers.append(pid)
                if acc > best:
                    bestinliers = inliers
                    best = acc
                    best3D = p3D

        if len(bestinliers) > 2:
            points_2d_set = []
            for j in bestinliers:
                points_2d_set.append((str(j), all_cands[j]))
            best3D = triangulate_one_point(camera_system, points_2d_set).squeeze()
        ret.append(best3D)
    return np.array(ret)

def build_multi_camera_system(Ks, RTs):
    """
    Build a multi-camera system with pymvg package for triangulation

    Args:
        Ks, RTs: list of camera parameters
    Returns:
        cams_system: a multi-cameras system
    """
    pymvg_cameras = []
    for name, (K, RT) in enumerate(zip(Ks, RTs)):
        P = np.zeros( (3,4) )
        P[:3,:3]=K

        distortion_coefficients = np.zeros((5,))
        i = sensor_msgs_compat.msg.CameraInfo()
        i.width = None
        i.height = None
        i.D = [float(val) for val in distortion_coefficients]
        i.K = list(K.flatten())
        i.R = list(np.eye(3).flatten())
        i.P = list(P.flatten())

        camera = CameraModel._from_parts(
            translation=RT[:, -1], 
            rotation=RT[:, :-1], 
            intrinsics=i,
            name=str(name))
        # camera = CameraModel.load_camera_from_M(KRT, name=str(name))
        pymvg_cameras.append(camera)
    return MultiCameraSystem(pymvg_cameras)


def triangulate_one_point(camera_system, points_2d_set):
    """
    Triangulate 3d point in world coordinates with multi-views 2d points

    Args:
        camera_system: pymvg camera system
        points_2d_set: list of structure (camera_name, point2d)
    Returns:
        points_3d: 3x1 point in world coordinates
    """
    # try:
    points_3d = camera_system.find3d(points_2d_set)
    # except:
    #     print(points_2d_set)
    return points_3d


def triangulate_pymvg(pts, Ks, RTs, confs):
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com), modified by Yihui He
# ------------------------------------------------------------------------------
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        KRTs: a list of camera parameters, each corresponding to
                       one prediction in pts
        pts: ndarray of shape nxkx2, len(cameras) == n
    Returns:
        poses3d: ndarray of shape n/nviews x k x 3
    """
    if torch.is_tensor(pts): pts = pts.cpu().numpy()
    if torch.is_tensor(Ks): Ks = Ks.cpu().numpy()
    if torch.is_tensor(RTs): RTs = RTs.cpu().numpy()
    if torch.is_tensor(confs): confs = confs.cpu().numpy()

    njoints = pts.shape[1]
    camera_system = build_multi_camera_system(Ks, RTs)
    p3D = np.zeros((njoints, 3))
    for k, conf in enumerate(confs.T):
        confthresh = cfg.KEYPOINT.CONF_THRES
        while True:
            selected_idx = np.where(conf > confthresh)[0]
            if confthresh < -1:
                break
            if len(selected_idx) <= 1:
                confthresh -= 0.05
                print('conf too high, decrease to', confthresh)
            else:
                break
        points_2d_set = []
        for j in selected_idx:
            points_2d = pts[j, k, :]
            points_2d_set.append((str(j), points_2d))
        p3D[k, :] = triangulate_one_point(camera_system, points_2d_set).T
    return p3D

if __name__ == '__main__':
    KRTs = 2*[np.eye(3, 4)]
    Ks = 2*[np.eye(3, 3)]
    RTs = 2*[np.eye(3, 4)]
    pts = np.zeros((2, 5, 2))
    confs = np.ones((2, 5))
    print(triangulate_pymvg(pts, Ks, RTs, confs))

