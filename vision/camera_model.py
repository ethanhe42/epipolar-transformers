import numpy as np
import yaml

from pymvg.util import _undistort, np2plain, \
     Bunch, plain_vec, my_rq, center, normalize, \
     point_msg_to_tuple, parse_rotation_msg, _cam_str, is_string
from pymvg.quaternions import quaternion_matrix, quaternion_from_matrix
from pymvg.ros_compat import sensor_msgs as sensor_msgs_compat

import warnings

# Define matrices to point ROS transforms such than +Z is directly
# ahead and +X is right.
rot_90 = np.array( [[ 0,0,1],
                    [ -1,0,0],
                    [ 0,-1,0]], dtype=np.float)
rot_90i = np.linalg.pinv(rot_90)

def is_rotation_matrix(R,eps=1e-5):
    # check if rotation matrix is really a pure rotation matrix

    # test: inverse is transpose
    testI = np.dot(R.T,R)
    # if not np.allclose( testI, np.eye(len(R)) ):
    #     print('R.T @ R', testI)
    #     return False

    # test: determinant is unity
    dr = np.linalg.det(R)
    if not (abs(dr-1.0)<eps):
        print('det', dr)
        return False

    # test: has one eigenvalue of unity
    l, W = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < eps)[0] # XXX do we need to check for complex part?
    if not len(i):
        print('eig', l, W)
        return False
    return True

def get_rotation_matrix_and_quaternion(rotation):
    rotation_orig = rotation
    rotation = np.array(rotation)
    if rotation.ndim==2:
        assert rotation.shape==(3,3)
        if not np.alltrue(np.isnan( rotation )):
            assert is_rotation_matrix(rotation)

        rmat = rotation

        rnew = np.eye(4)
        rnew[:3,:3] = rmat
        rquat = quaternion_from_matrix(rnew)
        if not np.alltrue(np.isnan( rquat )):
            R2 = quaternion_matrix(rquat)[:3,:3]
            assert np.allclose(rmat,R2)
    elif rotation.ndim==0:
        assert rotation.dtype == object
        rotation = (rotation_orig.x,
                    rotation_orig.y,
                    rotation_orig.z,
                    rotation_orig.w)
        return get_rotation_matrix_and_quaternion(rotation)
    else:
        assert rotation.ndim==1
        assert rotation.shape==(4,)
        rquat = rotation
        rmat = quaternion_matrix(rquat)[:3,:3]

        if not np.alltrue(np.isnan( rmat )):
            assert is_rotation_matrix(rmat)

    return rmat, rquat

def warn_deprecation(attr):
    warnings.warn('CameraModel property %r will be removed'%attr, DeprecationWarning)

class CameraModel(object):
    """an implementation of the Camera Model used by ROS and OpenCV
    Tranformations: We can think about the overall projection to 2D in
    two steps. Step 1 takes 3D world coordinates and, with a simple
    matrix multiplication and perspective division, projects them to
    undistorted 2D coordinates. Step 2 takes these undistorted 2D
    coordinates and distorts them so they are 'distorted' and match up
    with a real camera with radial distortion, for example.
    3D world --(step1)----> undistorted 2D ---(step2)----> distorted 2D
    Step 1 is accomplished by making the world coordinates a
    homogeneous vector of length 4, multiplying by a 3x4 matrix M
    (built from P, R and t) to get values [r,s,t] in which the
    undistorted 2D coordinates are [r/t, s/t]. (The implementation is
    vectorized so that in fact many points at once can be
    transformed.)
    Step 2 is somewhat complicated in that it allows a separate focal
    length and camera center to be used for distortion. Undistorted 2D
    coordinates are transformed first to uncorrected normalized image
    coordinates using parameters from P, then corrected using a
    rectification matrix. These corrected normalized image coordinates
    are then used in conjunction with the distortion model to create
    distorted normalized pixels which are finally transformed to
    distorted image pixels by K.
    Coordinate system: the camera is looking at +Z, with +X rightward
    and +Y down. For more information, see
    http://www.ros.org/wiki/image_pipeline/CameraInfo
    As noted on the link above, this differs from the coordinate
    system of Harley and Zisserman, which has Z forward, Y up, and X
    to the left (looking towards +Z).'
    """
    __slots__ = [
        # basic properties
        'name', # a string with the camera name
        'width','height', # the size of the image

        # extrinsic parameters
        '_rquat', # the rotation quaternion, np.array with shape (4,)
        '_camcenter', # the center of the camera, np.array with shape (3,)

        # intrinsic parameters
        '_opencv_compatible',
        # these intrinsic parameters specified like OpenCV
        'P', # used for undistorted<->normalized, np.array with shape (3,4)
        'K', # used for distorted<->normalized, a scaled version of P[:3,:3], np.array with shape (3,3)

        # (The scaling of K, with the default alpha=0, is such that
        # every pixel in the undistorted image is valid, thus throwing
        # away some pixels. With alpha=1, P==K and all pixels in the
        # original image are in the undistorted image.)

        # the distortion model
        'distortion', # (distortion params) the distortion, np.array with shape (5,1) or (8,1)
        'rect', # (distortion params) the rectification, None or np.array with shape (3,3)

        '_cache', # cached computational results
        ]
    AXIS_FORWARD = np.array((0,0,1),dtype=np.float)
    AXIS_UP = np.array((0,-1,0),dtype=np.float)
    AXIS_RIGHT = np.array((1,0,0),dtype=np.float)

    # --- start of CameraModel constructors ------------------------------------

    def __init__(self, name, width, height, _rquat, _camcenter, P, K, distortion, rect):
        self.name = name
        self.width = width
        self.height = height
        self._rquat = _rquat
        self._camcenter = _camcenter
        self.P = P
        eps = 1e-8
        if abs(self.P[2,2]-1.0) > eps:
            raise ValueError('matrix P must have element (2,2) near 1.0')
        self.K = K
        self.distortion = distortion
        self.rect = rect

        self._opencv_compatible = (self.P[0,1]==0)
        assert np.allclose( P[:,3], np.zeros((3,)))
        self._cache = {}
        self._cache['Q'] = self.get_Q()
        self._cache['translation'] = self.get_translation()
        self._cache['Qt'] = self.get_Qt()
        self._cache['M'] = self.get_M()
        self._cache['Q_inv'] = self.get_Q_inv()
        self._cache['t_inv'] = self.get_t_inv()

    def __getstate__(self):
        """allow CameraModel to be pickled"""
        return self.to_dict()

    def __setstate__(self,state):
        tmp = CameraModel.from_dict(state)
        for attr in self.__slots__:
            setattr( self, attr, getattr( tmp, attr ) )

    @classmethod
    def _from_parts(cls,
                      translation=None,
                      camcenter=None,
                      rotation=None,
                      intrinsics=None,
                      name=None,
                      ):
        """Instantiate a Camera Model.
        params
        ------
        translation : converted to np.array with shape (3,)
          the translational position of the camera (note: not the camera center)
        camcenter : converted to np.array with shape (3,)
          the camera center (mutually exclusive with translation parameter)
        rotation : converted to np.array with shape (4,) or (3,3)
          the camera orientation as a quaternion or a 3x3 rotation vector
        intrinsics : a ROS CameraInfo message
          the intrinsic camera calibration
        name : string
          the name of the camera
        """
        if translation is not None and camcenter is not None:
            raise RuntimeError('translation and camcenter arguments are mutually exclusive')
        set_camcenter_from_translation = True
        if camcenter is not None:
            set_camcenter_from_translation = False
            camcenter = np.array(camcenter)
            assert camcenter.ndim==1
            assert camcenter.shape[0]==3
        if translation is None:
            translation = (0,0,0)
        if rotation is None:
            rotation = np.eye(3)
        if name is None:
            name = 'camera'

        rmat, rquat = get_rotation_matrix_and_quaternion(rotation)

        if set_camcenter_from_translation:
            t = np.array(translation)
            t.shape = 3,1
            camcenter = -np.dot( rmat.T, t )[:,0]
            del t

        _rquat = rquat

        if 1:
            # Initialize the camera calibration from a CameraInfo message.
            msg = intrinsics
            width = msg.width
            height = msg.height
            shape = (msg.height, msg.width)

            P = np.array(msg.P,dtype=np.float)
            P.shape = (3,4)
            if not np.allclose(P[:,3], np.zeros((3,))):
                raise NotImplementedError('not tested when 4th column of P is nonzero')

            K = np.array( msg.K, dtype=np.float)
            K.shape = (3,3)
            assert K.ndim == 2

            distortion = np.array(msg.D, dtype=np.float)
            if len(distortion) == 5:
                distortion.shape = (5,)
            elif len(distortion) == 8:
                distortion.shape = (8,)
            else:
                raise ValueError('distortion can have only 5 or 8 entries')

            assert distortion.ndim==1

            if msg.R is None:
                rect = None
            else:
                rect = np.array( msg.R, dtype=np.float )
                rect.shape = (3,3)
                if np.allclose(rect,np.eye(3)):
                    rect = None

        denom = P[2,2]
        if denom != 1.0:
            if rect is not None:
                warnings.warn(
                    'A non-normalized P matrix and a rectification matrix were '
                    'supplied. This case is not well tested and the behavior '
                    'should be considered undefined.')
            P = P/denom # normalize
        result = cls(name, width, height, _rquat, camcenter, P, K, distortion, rect)
        return result

    @classmethod
    def from_dict(cls, d, extrinsics_required=True ):
        translation = None
        rotation = None

        if 'image_height' in d:
            # format saved in ~/.ros/camera_info/<camera_name>.yaml
            #only needs w,h,P,K,D,R
            c = sensor_msgs_compat.msg.CameraInfo(
                height=d['image_height'],
                width=d['image_width'],
                P=d['projection_matrix']['data'],
                K=d['camera_matrix']['data'],
                D=d['distortion_coefficients']['data'],
                R=d['rectification_matrix']['data'])
            name = d['camera_name']
        else:
            # format saved by roslib.message.strify_message( sensor_msgs.msg.CameraInfo() )
            c = sensor_msgs_compat.msg.CameraInfo(
                height = d['height'],
                width = d['width'],
                P=d['P'],
                K=d['K'],
                D=d['D'],
                R=d['R'])
            name = d.get('name',None)
            translation = d.get('translation',None)
            rotation = d.get('Q',None)

        if translation is None or rotation is None:
            if extrinsics_required:
                raise ValueError('extrinsic parameters are required, but not provided')

        result = cls._from_parts(translation=translation,
                                   rotation=rotation,
                                   intrinsics=c,
                                   name=name,
                                   )
        return result

    @classmethod
    def load_camera_from_file( cls, fname, extrinsics_required=True ):
        if fname.endswith('.bag'):
            raise NotImplementedError('cannot open .bag file directly. Open '
                                      'with ROS and call '
                                      'CameraModel.load_camera_from_opened_bagfile(bag, ...)')
        elif (fname.endswith('.yaml') or
              fname.endswith('.json')):
            if fname.endswith('.yaml'):
                with open(fname,'r') as f:
                    d = yaml.safe_load(f)
            else:
                assert fname.endswith('.json')
                with open(fname,'r') as f:
                    d = json.load(f)
            return cls.from_dict(d, extrinsics_required=extrinsics_required)
        else:
            raise ValueError("only supports: .bag .yaml .json")

    @classmethod
    def load_camera_from_opened_bagfile( cls, bag, extrinsics_required=True ): # pragma: no cover
        """factory function for class CameraModel
        arguments
        ---------
        bag - an opened rosbag.Bag instance
        extrinsics_required - are extrinsic parameters required
        """
        camera_name = None
        translation = None
        rotation = None
        intrinsics = None

        for topic, msg, t in bag.read_messages():
            if 1:
                parts = topic.split('/')
                if parts[0]=='':
                    parts = parts[1:]
                topic = parts[-1]
                parts = parts[:-1]
                if len(parts)>1:
                    this_camera_name = '/'.join(parts)
                else:
                    this_camera_name = parts[0]
                # empty, this_camera_name, topic = parts
                # assert empty==''
            if camera_name is None:
                camera_name = this_camera_name
            else:
                assert this_camera_name == camera_name

            if topic == 'tf':
                translation = msg.translation
                rotation = msg.rotation # quaternion
            elif topic == 'matrix_tf':
                translation = msg.translation
                rotation = msg.rotation # matrix
            elif topic == 'camera_info':
                intrinsics = msg
            else:
                warnings.warn('skipping message topic %r'%topic)
                continue

        bag.close()

        if translation is None or rotation is None:
            if extrinsics_required:
                raise ValueError('no extrinsic parameters in bag file')
            else:
                translation = (np.nan, np.nan, np.nan)
                rotation = (np.nan, np.nan, np.nan, np.nan)
        else:
            translation = point_msg_to_tuple(translation)
            rotation = parse_rotation_msg(rotation)

        if intrinsics is None:
            raise ValueError('no intrinsic parameters in bag file')

        result = cls._from_parts(translation=translation,
                                   rotation=rotation,
                                   intrinsics=intrinsics,
                                   name=camera_name,
                                   )
        return result


    @classmethod
    def load_camera_from_M( cls, pmat, width=None, height=None, name='cam',
                            distortion_coefficients=None,
                            _depth=0, eps=1e-15 ):
        """create CameraModel instance from a camera matrix M"""
        pmat = np.array(pmat)
        assert pmat.shape==(3,4)
        M = pmat[:,:3]
        K,R = my_rq(M)
        if not is_rotation_matrix(R):
            # RQ may return left-handed rotation matrix. Make right-handed.
            assert np.allclose(np.dot(-K,-R), np.dot(K,R))
            R = -R
            K = -K
        a = K[2,2]
        if a==0:
            warnings.warn('ill-conditioned intrinsic camera parameters')
        else:
            if abs(a-1.0) > eps:
                if _depth > 0:
                    raise ValueError('cannot scale this pmat: %s'%( repr(pmat,)))
                new_pmat = pmat/a
                cam = cls.load_camera_from_M( new_pmat, width=width, height=height, name=name, _depth=_depth+1)
                return cam

        camcenter = center(pmat)[:,0]

        P = np.zeros( (3,4) )
        P[:3,:3]=K

        if distortion_coefficients is None:
            distortion_coefficients = np.zeros((5,))
        else:
            distortion_coefficients = np.array(distortion_coefficients)
            assert distortion_coefficients.shape == (5,)

        i = sensor_msgs_compat.msg.CameraInfo()
        i.width = width
        i.height = height
        i.D = [float(val) for val in distortion_coefficients]
        i.K = list(K.flatten())
        i.R = list(np.eye(3).flatten())
        i.P = list(P.flatten())
        result = cls._from_parts(camcenter = camcenter,
                                   rotation = R,
                                   intrinsics = i,
                                   name=name)
        return result

    @classmethod
    def load_camera_default(cls):
        pmat = np.array( [[ 300,   0, 320, 0],
                          [   0, 300, 240, 0],
                          [   0,   0,   1, 0]])
        return cls.load_camera_from_M( pmat, width=640, height=480, name='cam')

    @classmethod
    def load_camera_from_ROS_tf( cls,
                                 translation=None,
                                 rotation=None,
                                 **kwargs):
        rmatx, rquatx = get_rotation_matrix_and_quaternion(rotation)
        rmat = np.dot( rot_90i,rmatx)
        rmat, rquat = get_rotation_matrix_and_quaternion(rmat)
        if hasattr(translation,'x'):
            translation = (translation.x, translation.y, translation.z)
        C = np.array(translation)
        C.shape = (3,)

        return cls._from_parts(camcenter=C, rotation=rquat, **kwargs)

    @classmethod
    def load_camera_simple( cls,
                            fov_x_degrees=30.0,
                            width=640, height=480,
                            eye=(0,0,0),
                            lookat=(0,0,-1),
                            up=None,
                            name='simple',
                            distortion_coefficients=None,
                            ):
        aspect = float(width)/float(height)
        fov_y_degrees = fov_x_degrees/aspect
        f = (width/2.0) / np.tan(np.radians(fov_x_degrees)/2.0)
        cx = width/2.0
        cy = height/2.0
        M = np.array( [[ f, 0, cx, 0],
                       [ 0, f, cy, 0],
                       [ 0, 0,  1, 0]])
        c1 = cls.load_camera_from_M( M, width=width, height=height, name=name,
                                     distortion_coefficients=distortion_coefficients)
        c2 = c1.get_view_camera( eye=eye, lookat=lookat, up=up)
        return c2

    # --- end of CameraModel constructors --------------------------------------

    def __str__(self):
        return _cam_str(self.to_dict())

    def __eq__(self,other):
        assert isinstance( self, CameraModel )
        if not isinstance( other, CameraModel ):
            return False
        d1 = self.to_dict()
        d2 = other.to_dict()
        for k in d1:
            if k not in d2:
                return False
            v1 = d1[k]
            v2 = d2[k]
            if is_string(v1):
                if not v1==v2:
                    return False
            elif v1 is None:
                if not v2 is None:
                    return False
            else:
                if not np.allclose(np.array(v1), np.array(v2)):
                    return False
        for k in d2:
            if k not in d1:
                return False
        return True

    def __ne__(self,other):
        return not (self==other)

    def to_dict(self):
        d = {}
        d['name'] =self.name
        d['height'] = self.height
        d['width'] = self.width
        d['P'] = np2plain(self.P)
        d['K'] = np2plain(self.K)
        d['D'] = np2plain(self.distortion[:])
        if self.rect is None:
            d['R'] = np2plain(np.eye(3))
        else:
            d['R'] = np2plain(self.rect)
        d['translation']=np2plain(self._cache['translation'])
        d['Q']=np2plain(self.get_Q())
        return d

    # -------------------------------------------------
    # properties / getters

    def get_Q(self):
        R = quaternion_matrix(self._rquat)[:3,:3]
        return R
    def _get_Q(self):
        warn_deprecation( 'Q' )
        return self.get_Q()
    Q = property(_get_Q)

    def get_Qt(self):
        Q = self._cache['Q']
        t = np.array(self._cache['translation'],copy=True)
        t.shape = 3,1
        Rt = np.hstack((Q,t))
        return Rt
    def _get_Qt(self):
        warn_deprecation( 'Qt' )
        return self.get_Qt()
    Qt = property(_get_Qt)

    def get_M(self):
        Qt = self._cache['Qt']
        P33 = self.P[:,:3]
        M = np.dot( P33, Qt )
        return M
    def _get_M(self):
        warn_deprecation( 'M' )
        return self.get_M()
    M = property(_get_M)

    def get_translation(self):
        Q = self._cache['Q']
        C = np.array(self._camcenter)
        C.shape = (3,1)
        t = -np.dot(Q, C)[:,0]
        return t
    def _get_translation(self):
        warn_deprecation( 'translation' )
        return self.get_translation()
    translation = property(_get_translation)

    def get_Q_inv(self):
        Q = self._cache['Q']
        return np.linalg.pinv(Q)
    def _get_Q_inv(self):
        warn_deprecation( 'Q_inv')
        return self.get_Q_inv()
    Q_inv = property(_get_Q_inv)

    def get_t_inv(self):
        ti = np.array(self._camcenter)
        ti.shape = 3,1
        return ti
    def _get_t_inv(self):
        warn_deprecation( 't_inv' )
        return self.get_t_inv()
    t_inv = property(_get_t_inv)

    # -------------------------------------------------
    # other getters

    def is_opencv_compatible(self):
        """True iff there is no skew"""
        return self._opencv_compatible

    def is_distorted_and_skewed(self,max_skew_ratio=1e15):
        """True if pixels are skewed and distorted"""

        if self.is_skewed(max_skew_ratio=max_skew_ratio):
            if np.sum(abs(self.distortion)) != 0.0:
                return True
        return False

    def is_skewed(self,max_skew_ratio=1e15):
        """True if pixels are skewed"""

        # With default value, if skew is 15 orders of magnitude less
        # than focal length, return False.

        skew = self.P[0,1]
        fx = self.P[0,0]

        if abs(skew) > (abs(fx)/max_skew_ratio):
            return True
        return False

    def get_name(self):
        return self.name

    def get_extrinsics_as_bunch(self):
        msg = Bunch()
        msg.translation = Bunch()
        msg.rotation = Bunch()
        translation = self._cache['translation']
        for i in range(3):
            setattr(msg.translation,'xyz'[i], translation[i] )
        for i in range(4):
            setattr(msg.rotation,'xyzw'[i], self._rquat[i] )
        return msg

    def get_ROS_tf(self):
        rmatx = self.get_Q()
        rmat = np.dot( rot_90,rmatx)
        rmat2, rquat2 = get_rotation_matrix_and_quaternion(rmat)
        return self.get_camcenter(), rquat2

    def get_intrinsics_as_bunch(self):
        i = Bunch()
        i.height = self.height
        i.width = self.width
        assert len(self.distortion) == 5
        i.distortion_model = 'plumb_bob'
        i.D = plain_vec(self.distortion.flatten())
        i.K = plain_vec(self.K.flatten())
        i.R = plain_vec(self.get_rect().flatten())
        i.P = plain_vec(self.P.flatten())
        return i

    def get_camcenter(self):
        t_inv = self._cache['t_inv']
        return t_inv[:,0] # drop dimension

    def get_lookat(self,distance=1.0):
        world_coords = self.project_camera_frame_to_3d( [distance*self.AXIS_FORWARD] )
        world_coords.shape = (3,) # drop dimension
        return world_coords

    def get_up(self,distance=1.0):
        world_coords = self.project_camera_frame_to_3d( [distance*self.AXIS_UP] )
        world_coords.shape = (3,) # drop dimension
        return world_coords-self._camcenter

    def get_right(self,distance=1.0):
        cam_coords = np.array([[distance,0,0]])
        world_coords = self.project_camera_frame_to_3d( [distance*self.AXIS_RIGHT] )
        world_coords.shape = (3,) # drop dimension
        return world_coords-self._camcenter

    def get_view(self):
        return self.get_camcenter(), self.get_lookat(), self.get_up()

    def get_rotation_quat(self):
        return np.array(self._rquat)

    def get_rotation(self):
        return self._cache['Q']

    def get_K(self):
        return self.K

    def get_D(self):
        return self.distortion

    def get_rect(self):
        if self.rect is None:
            return np.eye(3)
        else:
            return self.rect

    def get_P(self):
        return self.P

    def save_to_bagfile(self,fname,roslib): # pragma: no cover
        """save CameraModel to ROS bag file
        arguments
        ---------
        fname - filename or file descriptor to save to
        roslib - the roslib module
        """

        roslib.load_manifest('rosbag')
        roslib.load_manifest('sensor_msgs')
        roslib.load_manifest('geometry_msgs')

        import sensor_msgs.msg
        import geometry_msgs
        import rosbag

        bagout = rosbag.Bag(fname, 'w')

        msg = extrinsics = geometry_msgs.msg.Transform()
        b = self.get_extrinsics_as_bunch()
        for name in 'xyz':
            setattr(msg.translation, name, getattr(b.translation,name) )
        for name in 'xyzw':
            setattr(msg.rotation, name, getattr(b.rotation,name) )

        topic = self.name + '/tf'
        bagout.write(topic, extrinsics)

        msg = intrinsics = sensor_msgs.msg.CameraInfo()
        b = self.get_intrinsics_as_bunch()
        # these are from image_geometry ROS package in the utest.cpp file
        msg.height = b.height
        msg.width = b.width
        msg.distortion_model = b.distortion_model
        msg.D = b.D
        msg.K = b.K
        msg.R = b.R
        msg.P = b.P

        topic = self.name.encode('ascii') + '/camera_info'
        bagout.write(topic, intrinsics)

        bagout.close()

    def save_intrinsics_to_yamlfile(self,fname):
        b = self.get_intrinsics_as_bunch()
        d = b.__dict__
        buf = yaml.dump(d)
        with open(fname,'w') as fd:
            fd.write( buf )

    def get_mirror_camera(self,axis='lr',hold_center=False):
        """return a copy of this camera whose x coordinate is (image_width-x)
        arguments
        ---------
        axis - string. Specifies the axis of the mirroring, either 'lr' or 'ud'.
        hold_center - boolean. Preserve the optical center?
        """
        assert axis in ['lr','ud']
        # Keep extrinsic coordinates, but flip intrinsic
        # parameter so that a mirror image is rendered.
        i = self.get_intrinsics_as_bunch()
        if axis=='lr':
            i.K[0] = -i.K[0]
            i.P[0] = -i.P[0]

            if not hold_center:
                i.P[1] = -i.P[1]

                i.K[2] = (self.width-i.K[2])
                i.P[2] = (self.width-i.P[2])
        else:
            # axis=='ud'
            i.K[4] = -i.K[4]
            i.P[5] = -i.P[5]

            if not hold_center:
                i.K[5] = (self.height-i.K[5])
                i.P[6] = (self.height-i.P[6])

        translation = self._cache['translation']
        Q = self._cache['Q']
        camnew = self._from_parts(
                              translation = translation,
                              rotation = Q,
                              intrinsics = i,
                              name = self.name + '_mirror',
                              )
        return camnew

    def get_flipped_camera(self):
        """return a copy of this camera looking in the opposite direction
        The returned camera has the same 3D->2D projection. (The
        2D->3D projection results in a vector in the opposite
        direction.)
        """
        cc, la, up = self.get_view()
        lv = la-cc # look vector

        lv2 = -lv
        up2 = -up
        la2 = cc+lv2

        camnew = self.get_view_camera(cc, la2, up2).get_mirror_camera(hold_center=True)
        camnew.distortion[3] = -camnew.distortion[3]

        if camnew.rect is not None:
            camnew.rect[0,:] = -camnew.rect[0,:]
            camnew.distortion[3] = -camnew.distortion[3]
            camnew.K[0,0] = -camnew.K[0,0]
        return camnew

    def get_view_camera(self, eye, lookat, up=None):
        """return a copy of this camera with new extrinsic coordinates"""
        eye = np.array(eye); eye.shape=(3,)
        lookat = np.array(lookat); lookat.shape=(3,)
        gen_up = False
        if up is None:
            up = np.array((0,-1,0))
            gen_up = True
        else:
            up = np.array(up)
            assert up.ndim==1
            assert up.shape[0]==3
        lv = lookat - eye
        f = normalize(lv)
        old_settings = np.seterr(invalid='ignore')
        s = normalize( np.cross( f, up ))
        np.seterr(**old_settings)
        if np.isnan(s[0]) and gen_up:
            up = np.array((0,0,1))
            s = normalize( np.cross( f, up ))
        assert not np.isnan(s[0]), 'invalid up vector'
        u = normalize( np.cross( f, s ))
        R = np.array( [[ s[0], u[0], f[0]],
                       [ s[1], u[1], f[1]],
                       [ s[2], u[2], f[2]]]).T

        eye.shape = (3,1)
        t = -np.dot(R,eye)

        result = self._from_parts(
                             translation=t,
                             rotation=R,
                             intrinsics=self.get_intrinsics_as_bunch(),
                             name=self.name,
                             )
        return result

    def get_aligned_camera(self, scale, rotation, translation):
        """return a copy of this camera with new extrinsic coordinates"""
        s,R,t = scale, rotation, translation
        cc, la, up = self.get_view()
        f = la-cc

        X = np.linalg.inv( R )

        fa = np.dot( f, X )
        cca0 = cc*s
        cca = np.dot( cca0, X )
        laa = cca+fa
        up2 = np.dot( up, X )

        cca2 = cca+t
        laa2 = laa+t

        return self.get_view_camera(cca2, laa2, up2)

    # --------------------------------------------------
    # image coordinate operations

    def undistort(self, nparr):
        # See http://opencv.willowgarage.com/documentation/cpp/camera_calibration_and_3d_reconstruction.html#cv-undistortpoints

        # Parse inputs
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==2

        u = nparr[:,0]
        v = nparr[:,1]

        # prepare parameters
        K = self.get_K()

        fx = K[0,0]
        cx = K[0,2]
        fy = K[1,1]
        cy = K[1,2]

        # P=[fx' 0 cx' tx; 0 fy' cy' ty; 0 0 1 tz]

        P = self.get_P()
        fxp = P[0,0]
        cxp = P[0,2]
        fyp = P[1,1]
        cyp = P[1,2]

        # Apply intrinsic parameters to get normalized, distorted coordinates
        xpp = (u-cx)/fx
        ypp = (v-cy)/fy

        # Undistort
        (xp,yp) = _undistort( xpp, ypp, self.get_D() )

        # Now rectify
        R = self.rect
        if R is None:
            x = xp
            y = yp
        else:
            assert R.shape==(3,3)
            Rti = np.linalg.inv(R.T)
            uh = np.vstack( (xp,yp,np.ones_like(xp)) )
            XYWt = np.dot(Rti, uh)
            X = XYWt[0,:]
            Y = XYWt[1,:]
            W = XYWt[2,:]
            x = X/W
            y = Y/W

        # Finally, get (undistorted) pixel coordinates
        up = x*fxp + cxp
        vp = y*fyp + cyp

        return np.vstack( (up,vp) ).T

    def distort(self, nparr):
        # See http://opencv.willowgarage.com/documentation/cpp/camera_calibration_and_3d_reconstruction.html#cv-undistortpoints

        # Based on code in pinhole_camera_model.cpp of ROS image_geometry package.

        # Parse inputs
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==2

        uv_rect_x = nparr[:,0]
        uv_rect_y = nparr[:,1]

        # prepare parameters
        P = self.get_P()

        fx = P[0,0]
        cx = P[0,2]
        Tx = P[0,3]
        fy = P[1,1]
        cy = P[1,2]
        Ty = P[1,3]

        x = (uv_rect_x - cx - Tx)/fx
        y = (uv_rect_y - cy - Ty)/fy

        if self.rect is not None:
            R = self.rect.T
            xy1 = np.vstack((x,y,np.ones_like(x)))
            X,Y,W = np.dot(R, xy1)
            xp = X/W
            yp = Y/W
        else:
            xp = x
            yp = y
        r2 = xp*xp + yp*yp
        r4 = r2*r2
        r6 = r4*r2
        a1 = 2*xp*yp
        D = self.distortion
        k1 = D[0]; k2=D[1]; p1=D[2]; p2=D[3]; k3=D[4]
        barrel = 1 + k1*r2 + k2*r4 + k3*r6
        if len(D)==8:
            barrel /= (1.0 + D[5]*r2 + D[6]*r4 + D[7]*r6)
        xpp = xp*barrel + p1*a1 + p2*(r2+2*(xp*xp))
        ypp = yp*barrel + p1*(r2+2*(yp*yp)) + p2*a1

        K = self.get_K()
        kfx = K[0,0]
        kcx = K[0,2]
        kfy = K[1,1]
        kcy = K[1,2]

        u = xpp*kfx + kcx
        v = ypp*kfy + kcy
        return np.vstack( (u,v) ).T

    # --------------------------------------------------
    # 3D <-> image coordinate operations

    def project_pixel_to_camera_frame(self, nparr, distorted=True, distance=1.0 ):
        if distorted:
            nparr = self.undistort(nparr)
        # now nparr is undistorted (aka rectified) 2d point data

        # Parse inputs
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==2
        uv_rect_x = nparr[:,0]
        uv_rect_y = nparr[:,1]

        P = self.P/self.P[2,2] # normalize

        # transform to 3D point in camera frame at z=1
        y = (uv_rect_y            - P[1,2] - P[1,3]) / P[1,1]
        x = (uv_rect_x - P[0,1]*y - P[0,2] - P[0,3]) / P[0,0]
        z = np.ones_like(x)
        ray_cam = np.vstack((x,y,z))
        rl = np.sqrt(np.sum(ray_cam**2,axis=0))
        ray_cam = distance*(ray_cam/rl) # normalize then scale
        return ray_cam.T

    def project_camera_frame_to_pixel(self, pts3d, distorted=True):
        pts3d = np.array(pts3d,copy=False)
        assert pts3d.ndim==2
        assert pts3d.shape[1]==3

        # homogeneous and transposed
        pts3d_h = np.empty( (4,pts3d.shape[0]) )
        pts3d_h[:3,:] = pts3d.T
        pts3d_h[3] = 1

        # undistorted homogeneous image coords
        cc = np.dot(self.P, pts3d_h)

        # project
        pc = cc[:2]/cc[2]
        u, v = pc

        if distorted:
            # distort (the currently undistorted) image coordinates
            nparr = np.vstack((u,v)).T
            u,v = self.distort( nparr ).T
        return np.vstack((u,v)).T

    def project_pixel_to_3d_ray(self, nparr, distorted=True, distance=1.0 ):
        ray_cam = self.project_pixel_to_camera_frame( nparr, distorted=distorted, distance=distance )
        # transform to world frame
        return self.project_camera_frame_to_3d( ray_cam )

    def project_3d_to_pixel(self, pts3d, distorted=True):
        pts3d = np.array(pts3d,copy=False)
        assert pts3d.ndim==2
        assert pts3d.shape[1]==3

        # homogeneous and transposed
        pts3d_h = np.empty( (4,pts3d.shape[0]) )
        pts3d_h[:3,:] = pts3d.T
        pts3d_h[3] = 1

        M = self._cache['M']

        # undistorted homogeneous image coords
        cc = np.dot(M, pts3d_h)

        # project
        pc = cc[:2]/cc[2]
        u, v = pc

        if distorted:
            # distort (the currently undistorted) image coordinates
            nparr = np.vstack((u,v)).T
            u,v = self.distort( nparr ).T
        return np.vstack((u,v)).T

    def project_camera_frame_to_3d(self, pts3d):
        """take 3D coordinates in camera frame and convert to world frame"""
        Q_inv = self._cache['Q_inv']
        cam_coords = np.array(pts3d).T
        t = self.get_translation()
        t.shape = (3,1)
        world_coords = np.dot(Q_inv, cam_coords - t)
        return world_coords.T

    def project_3d_to_camera_frame(self, pts3d):
        """take 3D coordinates in world frame and convert to camera frame"""
        pts3d = np.array(pts3d)
        assert pts3d.ndim==2
        assert pts3d.shape[1]==3

        # homogeneous and transposed
        pts3d_h = np.empty( (4,pts3d.shape[0]) )
        pts3d_h[:3,:] = pts3d.T
        pts3d_h[3] = 1

        # undistorted homogeneous image coords
        Qt = self._cache['Qt']
        cc = np.dot(Qt, pts3d_h)

        return cc.T

    # --------------------------------------------
    # misc. helpers

    def camcenter_like(self,nparr):
        """create numpy array of camcenters like another array"""
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==3
        t_inv = self._cache['t_inv']
        return np.zeros( nparr.shape ) + t_inv.T