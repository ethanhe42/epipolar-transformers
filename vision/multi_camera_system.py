import os
import json
from collections import OrderedDict
import warnings

import numpy as np

from .camera_model import CameraModel
from .camera_model import is_rotation_matrix
from pymvg.util import pretty_json_dump, normalize_M, \
     parse_radfile, my_rq, center
from pymvg.align import estsimt

class MultiCameraSystem:
    def __init__(self,cameras):
        self._cameras=OrderedDict()
        for camera in cameras:
            self.append(camera)

    def append(self,camera):
        assert isinstance(camera, CameraModel)
        name = camera.name
        if name in self._cameras:
            raise ValueError('Cannot create MultiCameraSystem with '
                             'multiple identically-named cameras.')
        self._cameras[name]=camera

    @classmethod
    def from_dict(cls, d):
        cam_dict_list = d['camera_system']
        cams = [CameraModel.from_dict(cd) for cd in cam_dict_list]
        return MultiCameraSystem( cameras=cams )

    def get_pymvg_str( self ):
        d = self.to_dict()
        d['__pymvg_file_version__']='1.0'
        buf = pretty_json_dump(d)
        return buf

    def save_to_pymvg_file( self, fname ):
        buf = self.get_pymvg_str()
        with open(fname,mode='w') as fd:
            fd.write(buf)

    @classmethod
    def from_pymvg_str(cls, buf):
        d = json.loads(buf)
        assert d['__pymvg_file_version__']=='1.0'
        cam_dict_list = d['camera_system']
        cams = [CameraModel.from_dict(cd) for cd in cam_dict_list]
        return MultiCameraSystem( cameras=cams )

    @classmethod
    def from_pymvg_file(cls, fname):
        with open(fname,mode='r') as fd:
            buf = fd.read()
        return MultiCameraSystem.from_pymvg_str(buf)

    @classmethod
    def from_mcsc(cls, dirname ):
        '''create MultiCameraSystem from output directory of MultiCamSelfCal'''

        # FIXME: This is a bit convoluted because it's been converted
        # from multiple layers of internal code. It should really be
        # simplified and cleaned up.

        do_normalize_pmat=True

        all_Pmat = {}
        all_Res = {}
        all_K = {}
        all_distortion = {}

        opj = os.path.join

        with open(opj(dirname,'camera_order.txt'),mode='r') as fd:
            cam_ids = fd.read().strip().split('\n')

        with open(os.path.join(dirname,'Res.dat'),'r') as res_fd:
            for i, cam_id in enumerate(cam_ids):
                fname = 'camera%d.Pmat.cal'%(i+1)
                pmat = np.loadtxt(opj(dirname,fname)) # 3 rows x 4 columns
                if do_normalize_pmat:
                    pmat_orig = pmat
                    pmat = normalize_M(pmat)
                all_Pmat[cam_id] = pmat
                all_Res[cam_id] = map(int,res_fd.readline().split())

        # load non linear parameters
        rad_files = [ f for f in os.listdir(dirname) if f.endswith('.rad') ]
        for cam_id_enum, cam_id in enumerate(cam_ids):
            filename = os.path.join(dirname,
                                    'basename%d.rad'%(cam_id_enum+1,))
            if os.path.exists(filename):
                K, distortion = parse_radfile(filename)
                all_K[cam_id] = K
                all_distortion[cam_id] = distortion
            else:
                if len(rad_files):
                    raise RuntimeError(
                        '.rad files present but none named "%s"'%filename)
                warnings.warn('no non-linear data (e.g. radial distortion) '
                              'in calibration for %s'%cam_id)
                all_K[cam_id] = None
                all_distortion[cam_id] = None

        cameras = []
        for cam_id in cam_ids:
            w,h = all_Res[cam_id]
            Pmat = all_Pmat[cam_id]
            M = Pmat[:,:3]
            K,R = my_rq(M)
            if not is_rotation_matrix(R):
                # RQ may return left-handed rotation matrix. Make right-handed.
                R2 = -R
                K2 = -K
                assert np.allclose(np.dot(K2,R2), np.dot(K,R))
                K,R = K2,R2

            P = np.zeros((3,4))
            P[:3,:3] = K
            KK = all_K[cam_id] # from rad file or None
            distortion = all_distortion[cam_id]

            # (ab)use PyMVG's rectification to do coordinate transform
            # for MCSC's undistortion.

            # The intrinsic parameters used for 3D -> 2D.
            ex = P[0,0]
            bx = P[0,2]
            Sx = P[0,3]
            ey = P[1,1]
            by = P[1,2]
            Sy = P[1,3]

            if KK is None:
                rect = np.eye(3)
                KK = P[:,:3]
            else:
                # Parameters used to define undistortion coordinates.
                fx = KK[0,0]
                fy = KK[1,1]
                cx = KK[0,2]
                cy = KK[1,2]

                rect = np.array([[ ex/fx,     0, (bx+Sx-cx)/fx ],
                                 [     0, ey/fy, (by+Sy-cy)/fy ],
                                 [     0,     0,       1       ]]).T

            if distortion is None:
                distortion = np.zeros((5,))

            C = center(Pmat)
            rot = R
            t = -np.dot(rot, C)[:,0]

            d = {'width':w,
                 'height':h,
                 'P':P,
                 'K':KK,
                 'R':rect,
                 'translation':t,
                 'Q':rot,
                 'D':distortion,
                 'name':cam_id,
                 }
            cam = CameraModel.from_dict(d)
            cameras.append( cam )
        return MultiCameraSystem( cameras=cameras )

    def __eq__(self, other):
        assert isinstance( self, MultiCameraSystem )
        if not isinstance( other, MultiCameraSystem ):
            return False
        if len(self.get_names()) != len(other.get_names()):
            return False
        for name in self.get_names():
            if self._cameras[name] != other._cameras[name]:
                return False
        return True

    def __ne__(self,other):
        return not (self==other)

    def get_names(self):
        result = list(self._cameras.keys())
        return result

    def get_camera_dict(self):
        return self._cameras

    def get_camera(self,name):
        return self._cameras[name]

    def to_dict(self):
        return {'camera_system':
                [self._cameras[name].to_dict() for name in self._cameras]}

    def find3d(self,pts,undistort=True):
        """Find 3D coordinate using all data given
        Implements a linear triangulation method to find a 3D
        point. For example, see Hartley & Zisserman section 12.2
        (p.312).
        By default, this function will undistort 2D points before
        finding a 3D point.
        """
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        # Construct matrices
        A=[]
        for name,xy in pts:
            cam = self._cameras[name]
            if undistort:
                xy = cam.undistort( [xy] )
            Pmat = cam.get_M() # Pmat is 3 rows x 4 columns
            row2 = Pmat[2,:]
            x,y = xy[0,:]
            A.append( x*row2 - Pmat[0,:] )
            A.append( y*row2 - Pmat[1,:] )

        # Calculate best point
        A=np.array(A)
        u,d,vt=np.linalg.svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize
        return X

    def find2d(self,camera_name,xyz,distorted=True):
        cam = self._cameras[camera_name]

        xyz = np.array(xyz)
        rank1 = xyz.ndim==1

        xyz = np.atleast_2d(xyz)
        pix = cam.project_3d_to_pixel( xyz, distorted=distorted ).T

        if rank1:
            # convert back to rank1
            pix = pix[:,0]
        return pix

    def get_aligned_copy(self, other):
        """return copy of self that is scaled, translated, and rotated to best match other"""
        assert isinstance( other, MultiCameraSystem)

        orig_names = self.get_names()
        new_names = other.get_names()
        names = set(orig_names).intersection( new_names )
        if len(names) < 3:
            raise ValueError('need 3 or more cameras in common to align.')
        orig_points = np.array([ self._cameras[name].get_camcenter() for name in names ]).T
        new_points = np.array([ other._cameras[name].get_camcenter() for name in names ]).T

        s,R,t = estsimt(orig_points,new_points)
        assert is_rotation_matrix(R)

        new_cams = []
        for name in self.get_names():
            orig_cam = self._cameras[name]
            new_cam = orig_cam.get_aligned_camera(s,R,t)
            new_cams.append( new_cam )
        result = MultiCameraSystem(new_cams)
        return result

def build_example_system(n=6,z=5.0):
    base = CameraModel.load_camera_default()

    x = np.linspace(0, 2*n, n)
    theta = np.linspace(0, 2*np.pi, n)
    cams = []
    for i in range(n):
        # cameras are spaced parallel to the x axis
        center = np.array( (x[i], 0.0, z) )

        # cameras are looking at +y
        lookat = center + np.array( (0,1,0))

        # camera up direction rotates around the y axis
        up = -np.sin(theta[i]), 0, np.cos(theta[i])

        cam = base.get_view_camera(center,lookat,up)
        cam.name = 'theta: %.0f'%( np.degrees(theta[i]) )
        cams.append(cam)

    system = MultiCameraSystem(cams)
    return system