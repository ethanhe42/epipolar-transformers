import os

from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.

# ---------------------------------------------------------------------------- #
# backbones options
# ---------------------------------------------------------------------------- #
_C.BACKBONE = CN()
_C.BACKBONE.ENABLED = False
#ResNets: R-18,34,50,101,152
#HG, HG1, HG11
_C.BACKBONE.BODY = 'R-50'
_C.BACKBONE.PRETRAINED = True
_C.BACKBONE.PRETRAINED_WEIGHTS = ''
_C.BACKBONE.DOWNSAMPLE = 4
_C.BACKBONE.BN_MOMENTUM = 0.1
_C.BACKBONE.SYNC_BN = False

# ---------------------------------------------------------------------------- #
# Lifting Net options
# ---------------------------------------------------------------------------- #
_C.LIFTING = CN()
_C.LIFTING.ENABLED = False
_C.LIFTING.VIEW_ON = False
_C.LIFTING.FLIP_ON = False
_C.LIFTING.CROP_SIZE = 256
_C.LIFTING.IMAGE_SIZE = 320

# average loss over all keypoints
_C.LIFTING.AVELOSS_KP = False

# sanity check: select nearest 3D joint to GT
_C.LIFTING.MULTIVIEW_UPPERBOUND = False
_C.LIFTING.MULTIVIEW_MEDIUM = True

# ---------------------------------------------------------------------------- #
# Keypoint detector options
# ---------------------------------------------------------------------------- #
_C.KEYPOINT = CN()
_C.KEYPOINT.ENABLED = False
_C.KEYPOINT.SIGMA = 25.
_C.KEYPOINT.NUM_PTS = 21
_C.KEYPOINT.ROOTIDX = 0
_C.KEYPOINT.HEATMAP_SIZE = (224, 224)
# number of views to use
_C.KEYPOINT.NUM_CAM = 0
_C.KEYPOINT.NFEATS = 256
# naive, pymvg, refine, epipolar, epipolar_dlt, rpsm
_C.KEYPOINT.TRIANGULATION = 'naive'
_C.KEYPOINT.CONF_THRES = 0.05
_C.KEYPOINT.RANSAC_THRES = 3
# mse, joint, smoothmse
_C.KEYPOINT.LOSS = 'mse'
# calculate loss for each joint
_C.KEYPOINT.LOSS_PER_JOINT = True


# ---------------------------------------------------------------------------- #
# Epipolar options
# ---------------------------------------------------------------------------- #
_C.EPIPOLAR = CN()
# visualization
_C.EPIPOLAR.VIS = False
# random select a view from K nearest neighbors
# 0: use range
# >0: use TOPK
# <0: use baseline range like (1059, 200) which means 1059 +- 200
_C.EPIPOLAR.TOPK = 1
_C.EPIPOLAR.TOPK_RANGE = (1, 2)
# way to combine features on epipolar line
# max: select the most similar
# avg: weighted average based on similarity
_C.EPIPOLAR.ATTENTION = 'max'
_C.EPIPOLAR.SIMILARITY = 'dot'
# cos, dot
_C.EPIPOLAR.SAMPLESIZE = 64
_C.EPIPOLAR.SOFTMAX_ENABLED = True
_C.EPIPOLAR.SOFTMAXSCALE = 1 / _C.EPIPOLAR.SAMPLESIZE**.5
_C.EPIPOLAR.SOFTMAXBETA = True
# merge features early or late
_C.EPIPOLAR.MERGE = 'early'
# only use other view's image
_C.EPIPOLAR.OTHER_ONLY = False
# gradient on other view
_C.EPIPOLAR.OTHER_GRAD = ('other1', 'other2')
# share weights between reference view and other view
_C.EPIPOLAR.SHARE_WEIGHTS = False
# share weights between reference view and other view
# can parameterize 'z', 'theta', 'phi', 'g'
_C.EPIPOLAR.PARAMETERIZED = ()
_C.EPIPOLAR.ZRESIDUAL = False
# test all neighbouring views and adopt the best according to confidence
_C.EPIPOLAR.MULTITEST = False
_C.EPIPOLAR.WARPEDHEATMAP = False
# learn prior for each pair of views
_C.EPIPOLAR.PRIOR = False
_C.EPIPOLAR.PRIORMUL = False

_C.EPIPOLAR.REPROJECT_LOSS_WEIGHT = 0.
_C.EPIPOLAR.SIM_LOSS_WEIGHT = 0.
# load model from single view pretrained model
_C.EPIPOLAR.PRETRAINED = True

# find corrspondence based on 'feature' or 'rgb'
_C.EPIPOLAR.FIND_CORR = 'feature'

_C.EPIPOLAR.BOTTLENECK = 1
_C.EPIPOLAR.POOLING = False

_C.EPIPOLAR.USE_CORRECT_NORMALIZE = False

# ---------------------------------------------------------------------------- #
# pictorial structure options
# ---------------------------------------------------------------------------- #
_C.PICT_STRUCT = CN()
_C.PICT_STRUCT.FIRST_NBINS = 16
_C.PICT_STRUCT.PAIRWISE_FILE = 'datasets/h36m/pairwise.pkl'
_C.PICT_STRUCT.RECUR_NBINS = 2
_C.PICT_STRUCT.RECUR_DEPTH = 10
_C.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
_C.PICT_STRUCT.GRID_SIZE = 2000
_C.PICT_STRUCT.DEBUG = False
_C.PICT_STRUCT.TEST_PAIRWISE = False
_C.PICT_STRUCT.SHOW_ORIIMG = False
_C.PICT_STRUCT.SHOW_CROPIMG = False
_C.PICT_STRUCT.SHOW_HEATIMG = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
# percentage of data to use
_C.DATASETS.COMPLETENESS = 1.

# task can be the one of the followings: 
#-----
# lifting:                   2D heatmap to 3D canonical joints
# lifting_rot:               2D heatmap to 3D local joints (learn lifting and rot_mat seperately)
# img_lifting_rot:           RGB image to 3D local joints (learn lifting and rot_mat seperately)
# lifting_direct:            direct lifting to 3D local joints without rot_mat
# keypoint:                  2D keypoint detection
# keypoint_lifting_rot:      2D learned heatmap to 3D local joints (learn lifting and rot_mat seperately)
# keypoint_lifting_direct:      2D learned heatmap, direct lifting to 3D local joints without rot_mat
#-----
# multiview_keypoint:        multiview 2D keypoint detection
# multiview_img_lifting_rot: multiview RGB image to 3D local joints (learn lifting and rot_mat seperately)
_C.DATASETS.TASK = 'lifting'

# if true, do nothing; if false, use the palm (kp_0 + kp_12)/2 as 0-th keypoint 
_C.DATASETS.WRIST_COORD = False
# image size in H, W format
_C.DATASETS.IMAGE_SIZE = (512, 336)
_C.DATASETS.CROP_AFTER_RESIZE = False
_C.DATASETS.CROP_SIZE = (512, 320)
# image resize ratio
_C.DATASETS.IMAGE_RESIZE = 2.
# upscale to original image size for prediction
_C.DATASETS.PREDICT_RESIZE = 4.

_C.DATASETS.INCLUDE_GREY_IMGS = True
# INCLUDE CAMERAS, if empty, use all
_C.DATASETS.CAMERAS = ()

#jpg, zip, undistoredzip
_C.DATASETS.DATA_FORMAT = 'jpg'
#augmentation
_C.DATASETS.ROT_FACTOR = 0
_C.DATASETS.SCALE_FACTOR = 0.

_C.DATASETS.H36M = CN()
# use 2d joints to compute more accurate 3d joints during testing
_C.DATASETS.H36M.REAL3D = True
# map keypoints from 17 to 20, which is used in MPII
_C.DATASETS.H36M.MAPPING = True
# filter damaged annotations
_C.DATASETS.H36M.FILTER_DAMAGE = True
# sample train set
_C.DATASETS.H36M.TRAIN_SAMPLE = 5
# sample test set
_C.DATASETS.H36M.TEST_SAMPLE = 64
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 20
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.BENCHMARK = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# sgd, adam
_C.SOLVER.OPTIMIZER = 'sgd'
_C.SOLVER.SCHEDULER = 'multistep'

_C.SOLVER.FINETUNE = False
# freeze backbone while finetuning
_C.SOLVER.FINETUNE_FREEZE = True
_C.SOLVER.MAX_EPOCHS = 40
_C.SOLVER.STEPS = (20, 30,)

_C.SOLVER.BASE_LR = 1e-3

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0000

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.CHECKPOINT_PERIOD = 2

# Number of images per batch
_C.SOLVER.IMS_PER_BATCH = 8
# iter before update
_C.SOLVER.BATCH_MUL = 1

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.THRESHOLDS = (1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100)
_C.TEST.MAX_TH = 20
_C.TEST.PCK = True
_C.TEST.EPEMEAN_MAX_DIST = 150 #if a point is in GT but not detected, penalty 150mm
# recompute bn with frozen params
_C.TEST.RECOMPUTE_BN = False
# use train mode for testing
_C.TEST.TRAIN_BN = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.SEED = 0

_C.OUTPUT_DIR = 'outs'
_C.FOLDER_NAME = 'outs/.'

_C.WEIGHTS = ""
_C.WEIGHTS_PREFIX = "module."
_C.WEIGHTS_PREFIX_REPLACE = ""
_C.WEIGHTS_LOAD_OPT = True
_C.WEIGHTS_ALLOW_DIFF_PREFIX = False
# skip loading optimizer
#_C.LOAD_WEIGHTS_ONLY = True

_C.DEVICE = 'cuda'

_C.TENSORBOARD = CN()
_C.TENSORBOARD.USE = True
_C.TENSORBOARD.COMMENT = ""
# log / iterations
_C.LOG_FREQ = 100
# eval / epochs
_C.EVAL_FREQ = 4

# inference without training 
_C.DOTRAIN = True
_C.DOTEST = True

#visualization
_C.VIS = CN()
_C.VIS.DOVIS = True
_C.VIS.SAVE_PRED = False
_C.VIS.SAVE_PRED_NAME = "predictions.pth"
_C.VIS.SAVE_PRED_FREQ = 100
_C.VIS.SAVE_PRED_LIMIT = -1
_C.VIS.MULTIVIEW = False
_C.VIS.POINTCLOUD = False
_C.VIS.AUC = False
_C.VIS.H36M = False
_C.VIS.VIDEO = False
_C.VIS.VIDEO_GT = False
_C.VIS.MULTIVIEWH36M = False
_C.VIS.EPIPOLAR_LINE = False
_C.VIS.CURSOR = False
_C.VIS.FLOPS = False