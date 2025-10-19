"""Configuration for BCSS segmentation."""

import os

# Data paths
DATA_DIR = './BCSS'
TRAIN_IMAGE_PATH = os.path.join(DATA_DIR, 'train/')
VAL_IMAGE_PATH = os.path.join(DATA_DIR, 'val/')
TRAIN_MASK_PATH = os.path.join(DATA_DIR, 'train_mask/')
VAL_MASK_PATH = os.path.join(DATA_DIR, 'val_mask/')
TEST_IMAGE_PATH = os.path.join(DATA_DIR, 'test/')

# Normalization values (ImageNet standard)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Model parameters
N_CHANNELS = 3
N_CLASSES = 3
USE_ATTENTION = True

# Training parameters
BATCH_SIZE = 48
ACCUMULATION_STEPS = 2
NUM_WORKERS = 8
MAX_LR = 5e-4  # 改為 5e-4 (原本 1e-3 太大)
MAX_EPOCHS = 50  # 改為 50 epochs (原本 100 太多)
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP = 1.0
LOSS_WEIGHTS = (0.4, 0.6)  # (CE weight, Dice weight)

# Class weights for handling imbalance (背景權重 0.2)
CLASS_WEIGHTS = [0.2, 1.0, 1.0]  # [background, class1, class2]

# Post-processing parameters
# Note: For 224×224 images, object size guidelines:
#   - 10-20 pixels: ~1×1 to 4.5×4.5 (very small, likely noise)
#   - 20-50 pixels: ~4.5×4.5 to 7×7 (small objects)
#   - 50-100 pixels: ~7×7 to 10×10 (medium-small objects)
#   - 100+ pixels: 10×10+ (substantial objects)
POST_PROCESS_MIN_SIZE = 20  # 移除小於此像素的物體 (推薦: 20-50)
POST_PROCESS_KERNEL_SIZE = 3  # 形態學操作的 kernel 大小 (推薦: 3 or 5)
POST_PROCESS_FILL_HOLES = True  # 是否填充空洞
USE_TTA = True  # 是否使用測試時增強 (Test-Time Augmentation)

# Alternative: Class-specific min_size (if needed)
# POST_PROCESS_MIN_SIZE_PER_CLASS = {
#     0: 0,    # background (not used)
#     1: 30,   # class 1
#     2: 30,   # class 2
# }

# Paths
CHECKPOINT_DIR = 'ckpt'
LOG_DIR = 'logs'
OUTPUT_FILE = 'output.csv'
