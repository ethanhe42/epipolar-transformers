import numpy as np
import os.path as osp

def palm_coord(keypoint_xyz):
    """
    input: 42x3 or 42x2
    """
    palm_coord_l = (0.5*(keypoint_xyz[0 , :] + keypoint_xyz[12, :]))[None, ...]
    palm_coord_r = (0.5*(keypoint_xyz[21, :] + keypoint_xyz[33, :]))[None, ...]
    return np.vstack([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]])

def palm_coord_singlehand(keypoint_xyz):
    """
    input: 21x3 or 21x2
    """
    palm_coord_l = (0.5*(keypoint_xyz[0 , :] + keypoint_xyz[12, :]))[None, ...]
    return np.vstack([palm_coord_l, keypoint_xyz[1:21, :]])

def flip_hand(coords_xyz_canonical):
    """ Flips the given canonical coordinates
        The returned coordinates represent those of a left hand.
        Inputs:
            coords_xyz_canonical: Nx3 matrix, containing the coordinates for each of the N keypoints
    """
    # mirror along y axis
    coords_xyz_canonical_mirrored = coords_xyz_canonical.copy()
    coords_xyz_canonical_mirrored[..., 2] = -coords_xyz_canonical_mirrored[..., 2]
    return coords_xyz_canonical_mirrored

def canonical_trafo(coords_xyz , DEBUG=False):
    """
    coords_xyz: 21x3
    """
    assert coords_xyz.shape == (21, 3), coords_xyz.shape
    ROOT_NODE_ID = 0  # Node that will be at 0/0/0: 0=palm keypoint (root)
    ALIGN_NODE_ID = 12  # Node that will be at 0/-D/0: 12=beginning of middle finger
    ROT_NODE_ID = 20  # Node that will be at z=0, x>0; 20: Beginning of pinky

    # 1. Translate the whole set s.t. the root kp is located in the origin
    #3x1
    # trans = coords_xyz[ROOT_NODE_ID, :]
    # coords_xyz_t = coords_xyz - trans
    coords_xyz_t = coords_xyz

    # 2. Rotate and scale keypoints such that the root bone is of unit length and aligned with the y axis
    #3
    p = coords_xyz_t[ALIGN_NODE_ID, :]

    # Rotate point into the yz-plane
    alpha = atan2(p[0], p[1])
    #alpha = np.arctan2(p[1], p[0])
    rot_mat = _get_rot_mat_z(alpha)
    #21x3
    coords_xyz_t_r1 = coords_xyz_t.dot(rot_mat)
    total_rot_mat = rot_mat

    # Rotate point within the yz-plane onto the xy-plane
    p1 = coords_xyz_t_r1[ALIGN_NODE_ID]
    beta = -atan2(p1[2], p1[1])
    #beta  = np.arctan2(p1[1], p1[2])
    rot_mat = _get_rot_mat_x(beta + np.pi)
    coords_xyz_t_r2 = coords_xyz_t_r1.dot(rot_mat)
    total_rot_mat = total_rot_mat.dot(rot_mat)

    # 3. Rotate keypoints such that rotation along the y-axis is defined
    p2 = coords_xyz_t_r2[ROT_NODE_ID]
    gamma = atan2(p2[2], p2[0])
    #gamma   = np.arctan2(p2[0], p2[2])
    rot_mat = _get_rot_mat_y(gamma)
    coords_xyz_normed = coords_xyz_t_r2.dot(rot_mat)
    total_rot_mat = total_rot_mat.dot(rot_mat)
    total_rot_mat = np.linalg.inv(total_rot_mat)
    return coords_xyz_normed, total_rot_mat

def pts_normalize_rot(coords_xyzo , DEBUG=False):
    """
    coords_xyzo: 4x21
    """
    assert coords_xyzo.shape == (4, 21), coords_xyzo.shape
    ROOT_NODE_ID = 20  # Node that will be at 0/0/0: 0=palm keypoint (root)
    ALIGN_NODE_ID = 11  # Node that will be at 0/-D/0: 12=beginning of middle finger
    ROT_NODE_ID = 19  # Node that will be at z=0, x>0; 20: Beginning of pinky

    coords_o = coords_xyzo[-1, :] 
    out_range = np.any(np.abs(coords_xyzo[:3]) > 2000, axis=0)
    coords_o[out_range] = 0.

    for i in [10, ROOT_NODE_ID, ROT_NODE_ID, ALIGN_NODE_ID]:
        if coords_o[i] < 0.5:
            return coords_xyzo, None, 0, False


    #3x21
    coords_xyz = coords_xyzo[:3]

    #0. scale
    scale = np.linalg.norm(coords_xyz[:, 11] - coords_xyz[:, 10])
    coords_xyz  /= scale

    # 1. Translate the whole set s.t. the root kp is located in the origin
    #3x1
    trans = coords_xyz[:, ROOT_NODE_ID, None]
    coords_xyz_t = coords_xyz - trans

    # 2. Rotate and scale keypoints such that the root bone is of unit length and aligned with the y axis
    #3
    p = coords_xyz_t[:, ALIGN_NODE_ID]

    # Rotate point into the yz-plane
    alpha = atan2(p[0], p[1])
    #alpha = np.arctan2(p[1], p[0])
    rot_mat = _get_rot_mat_z(alpha)
    #21x3
    coords_xyz_t_r1 = coords_xyz_t.T.dot(rot_mat)
    total_rot_mat = rot_mat

    # Rotate point within the yz-plane onto the xy-plane
    p1 = coords_xyz_t_r1[ALIGN_NODE_ID]
    beta = -atan2(p1[2], p1[1])
    #beta  = np.arctan2(p1[1], p1[2])
    rot_mat = _get_rot_mat_x(beta + np.pi)
    coords_xyz_t_r2 = coords_xyz_t_r1.dot(rot_mat)
    total_rot_mat = total_rot_mat.dot(rot_mat)
    if DEBUG:
        print(np.hstack((np.arange(21)[:,None], coords_xyz_t_r1[:, :3])))
        print(np.hstack((np.arange(21)[:,None], coords_xyz_t_r2[:, :3])))

    # 3. Rotate keypoints such that rotation along the y-axis is defined
    p2 = coords_xyz_t_r2[ROT_NODE_ID]
    gamma = atan2(p2[2], p2[0])
    #gamma   = np.arctan2(p2[0], p2[2])
    rot_mat = _get_rot_mat_y(gamma)
    coords_xyz_normed = coords_xyz_t_r2.dot(rot_mat)
    total_rot_mat = total_rot_mat.dot(rot_mat)



    return np.vstack((coords_xyz_normed.T, coords_o[None, :])), total_rot_mat, scale, True 

def atan2(y, x):
    """ My implementation of atan2 in tensorflow.  Returns in -pi .. pi."""
    tan = np.arctan(y / (x + 1e-8))  # this returns in -pi/2 .. pi/2

    # correct quadrant error
    correction = np.pi if x + 1e-8 < 0. else 0
    tan_c = tan + correction  # this returns in -pi/2 .. 3pi/2

    # bring to positive values
    correction = 2*np.pi if tan_c < 0.0 else 0.
    tan_zero_2pi = tan_c + correction  # this returns in 0 .. 2pi

    # make symmetric
    correction = -2*np.pi if tan_zero_2pi > np.pi else 0.
    tan_final = tan_zero_2pi + correction  # this returns in -pi .. pi
    return tan_final

def _get_rot_mat_x(angle):
    """ Returns a 3D rotation matrix. """
    return np.array([[1            , 0             ,  0            ],
                     [0            ,  np.cos(angle),  np.sin(angle)],
                     [0            , -np.sin(angle),  np.cos(angle)]])

def _get_rot_mat_y(angle):
    """ Returns a 3D rotation matrix. """
    return np.array([[np.cos(angle), 0            , -np.sin(angle)],
                     [0            , 1            , 0             ],
                     [np.sin(angle), 0            ,  np.cos(angle)]])


def _get_rot_mat_z(angle):
    """ Returns a 3D rotation matrix. """
    return np.array([[ np.cos(angle), np.sin(angle), 0],
                     [-np.sin(angle), np.cos(angle), 0],
                     [0,              0,             1]])
