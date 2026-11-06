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

cfg = CN()

cfg.MODEL = CN()
# Using cuda or cpu for training
cfg.MODEL.DEVICE = "cuda:0"
# ID number of GPU
cfg.MODEL.DEVICE_ID = '0'
# Name of backbone
cfg.MODEL.NAME = 'resnet50_nl'  #transformer resnet50_nl
# Last stride of backbone
cfg.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
cfg.MODEL.PRETRAIN_PATH = '/data2/chenzhenghan/home/pretrained/resnet50-19c8e357.pth' #vit_base_ics_cfs_lup.pth resnet50-19c8e357.pth
cfg.MODEL.PRETRAIN_HW_RATIO = 2
cfg.MODEL.PRETRAIN_DIR = './log/market1501/AGW-mosaic'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self', refer using pretrained resnet50 or pretrained re-id baseline respectively.
cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with center loss, options: 'bnneck' or 'no'
cfg.MODEL.CENTER_LOSS = 'off'
cfg.MODEL.CENTER_FEAT_DIM = 2048  #768 2048



# Pose-Guided-KPR
cfg.MODEL.POSE_GUIDE = 'off'    
cfg.MODEL.POSE_RADIUS = 5        # Keypoint Expansion Radius (pixels)
cfg.MODEL.POSE_LAMBDA = 1.0    # Guidance Loss Weight
cfg.MODEL.POSE_MIN_CONF = 0.3    # Minimum Confidence Threshold for Key Points
cfg.MODEL.POSE_SHORT_SIDE = 256     
cfg.MODEL.POSE_STARTEPOCH = 30


#DCMA
cfg.MODEL.DCMA = 'off'
cfg.MODEL.DCMA_LAMBDA = 0.1




# If train with weighted regularized triplet loss, options: 'on', 'off'
cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET = 'on'
# If train with generalized mean pooling, options: 'on', 'off'
cfg.MODEL.GENERALIZED_MEAN_POOL = 'on'

# gan mode 'the type of GAN objective. [vanilla] # [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'
cfg.MODEL.GAN_MODE = 'vanilla'
# # of gen filters in the last conv layer
cfg.MODEL.NGF = 64
# # of discrim filters in the first conv layer
cfg.MODEL.NDF = 64
# specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
cfg.MODEL.NETD = 'basic'
# specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
cfg.MODEL.NETG = 'unet_128'
# only used if netD==n_layers
cfg.MODEL.N_LAYERS_D = 3
# instance normalization or batch normalization [batch] # [instance | batch | none]
cfg.MODEL.NORM = 'batch'
# network initialization [normal | xavier | kaiming | orthogonal]
cfg.MODEL.INIT_TYPE = 'normal'
# scaling factor for normal, xavier and orthogonal.
cfg.MODEL.INIT_GAIN = 0.02
# no dropout for the generator
cfg.MODEL.NO_DROPOUT = 'store_true'
# the size of image buffer that stores previously generated images
cfg.MODEL.POOL_SIZE = 50
# which epoch to load? set to latest to use latest cached model
cfg.MODEL.EPOCH = 'best'
# train mode options: 'C', 'GCR'
cfg.MODEL.MODE = 'GCR'
# train input mode options: 'origin', 'protected', 'both'
cfg.MODEL.OUTPUT_MODE = 'both'
cfg.MODEL.MI_PERIOD = 20
cfg.MODEL.VAL_R1 = 0.30
cfg.MODEL.PSNR = 1.0
cfg.MODEL.SSIM = 0.05

# If train with arcface loss, options: 'True', 'False'
cfg.MODEL.COS_LAYER = False

cfg.MODEL.DROPOUT_RATE = 0.0
# Reduce feature dim
cfg.MODEL.REDUCE_FEAT_DIM = False
cfg.MODEL.FEAT_DIM = 512
# If train with BNNeck, options: 'bnneck' or 'no'
cfg.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
cfg.MODEL.IF_WITH_CENTER = 'no'

cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.MODEL.ID_LOSS_WEIGHT = 1.0
cfg.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

# Transformer setting
cfg.MODEL.DROP_PATH = 0.1
cfg.MODEL.DROP_OUT = 0.0
cfg.MODEL.ATT_DROP_RATE = 0.0
cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
cfg.MODEL.STRIDE_SIZE = [16, 16]
cfg.MODEL.GEM_POOLING = False
cfg.MODEL.STEM_CONV = True

# JPM Parameter
cfg.MODEL.JPM = False
cfg.MODEL.SHIFT_NUM = 5
cfg.MODEL.SHUFFLE_GROUP = 2
cfg.MODEL.DEVIDE_LENGTH = 4
cfg.MODEL.RE_ARRANGE = True

# SIE Parameter
cfg.MODEL.SIE_COE = 3.0
cfg.MODEL.SIE_CAMERA = False
cfg.MODEL.SIE_VIEW = False

cfg.MODEL.TRIAL = 0
# dual or one
cfg.MODEL.SCHEMA = "dual"
# on or off
cfg.MODEL.ANON_LOSS = 'on'

cfg.MODEL.FUSION='off'

cfg.MODEL.TS_SCALE = 0.5

cfg.MODEL.TO_SCALE = 1
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Size of the image
cfg.INPUT.IMG_SIZE = [256, 128]
# Random probability for image horizontal flip
cfg.INPUT.PROB = 0.5
# Random probability for random erasing
cfg.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
cfg.INPUT.PADDING = 10
# direction
cfg.INPUT.DIRECTION = 'AtoB'
cfg.INPUT.TYPE = 'mosaic' # noise, blur, mosaic
cfg.INPUT.RADIUS = 24.0  # 0.5 for noise, 12.0 for blur, 24.0 for mosaic

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
cfg.DATASETS.NAMES = ('sysu')
# Root directory where datasets should be used (and downloaded if not found)
cfg.DATASETS.ROOT_DIR = ('./data/sysu/')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATALOADER = CN()
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 8    #previous 8
# If use PK sampler for data loading
cfg.DATALOADER.PK_SAMPLER = 'on'
# Number of id for one batch
cfg.DATALOADER.NUM_INSTANCE = 8
# Number of sample per id
cfg.DATALOADER.NUM_POS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# Name of optimizer
cfg.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
cfg.SOLVER.MAX_EPOCHS = 120
# Base learning rate
cfg.SOLVER.BASE_LR = 0.00035
# Momentum
cfg.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
cfg.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
cfg.SOLVER.CENTER_LR = 0.5
cfg.SOLVER.ANON_LR = 0.5
# Balanced weight of center loss
cfg.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
cfg.SOLVER.WEIGHT_DECAY = 0.0005

# decay rate of learning rate
cfg.SOLVER.GAMMA = 0.1
# decay step of learning rate
cfg.SOLVER.STEPS = [40, 80]

# warm up factor
cfg.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
cfg.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
cfg.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
cfg.SOLVER.CHECKPOINT_PERIOD = 60
# iteration of display training log
cfg.SOLVER.LOG_PERIOD = 50
# epoch number of validation
cfg.SOLVER.EVAL_PERIOD = 3

cfg.SOLVER.VAL_PERIOD = 5

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
cfg.SOLVER.IMS_PER_BATCH = 80

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
# Number of images per batch during test
cfg.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'on','off'
cfg.TEST.RE_RANKING = 'off'
# Path to trained model
cfg.TEST.WEIGHT = "path"
# Whether feature is nomalized before test, if on, it is equivalent to cosine distance
cfg.TEST.FEAT_NORM = 'on'
cfg.TEST.EVALUATE_ONLY = 'off'
# Whether to evaluate on partial re-id dataset
cfg.TEST.PARTIAL_REID = 'off'
# gallery type  options: 'origin', 'protected', 'both'
cfg.TEST.GALLERY_MODE = 'both'
# recovery with key
cfg.TEST.RECOVERY_WITH_KEY = True
# retrieve with key
cfg.TEST.RETRIEVE_WITH_KEY = True
# query mode
cfg.TEST.QUERY_MODE = 'both'
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
cfg.TEST.NECK_FEAT = 'after'
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
cfg.OUTPUT_DIR = './log/sysu/img2'
cfg.PRETRAINED_TEACHER = '/data0/yb_data/pretrained'
