from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.BACKBONE = "resnet50"
_C.MODEL.WEIGHT = 'imagenet'
#_C.MODEL = CN()
# RESNET
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.PRETRAINED = True
_C.MODEL.RESNET.RES5_STRIDE = 2
_C.MODEL.RESNET.RES5_DILATION = 1

# SEPCONVLSTM
_C.MODEL.SEPCONVLSTM = CN()
_C.MODEL.SEPCONVLSTM.INPUT_DIM = 512
_C.MODEL.SEPCONVLSTM.HIDDEN_DIM = 128

# SEG_HEAD
_C.MODEL.SEG_HEAD = CN()
_C.MODEL.SEG_HEAD.INPUT_CHANNELS = 128

# FUSE
_C.MODEL.FUSE = CN()
_C.MODEL.FUSE.INPUT_CHANNELS = 128
_C.MODEL.FUSE.OUTPUT_CHANNELS = 128

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.HEIGHT = 224

_C.INPUT.WIDTH = 224
# Size of the image during training
#_C.INPUT.SIZE_TRAIN = (224, 224)
# Size of the image during test
#_C.INPUT.SIZE_TEST = (224, 224)
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 6

_C.LOSS = CN()
#_C.LOSS.LOSS_TYPE = ['BCE']]

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (300000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 2
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.LRSCHEDULER = "step"
_C.SOLVER.TARGET_LR = 0.0002
_C.SOLVER.POWER = 0.9
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 4

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
