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
BASE_C = 64  # 基礎通道數
USE_ATTENTION = False  # 是否使用 Attention 機制 (對方的模型不使用)
USE_DROPOUT = False  # 是否使用 Dropout
DROPOUT_RATE = 0.1  # Dropout 比率 (當 USE_DROPOUT=True 時生效)

# Training parameters
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
NUM_WORKERS = 8
MAX_LR = 3e-4  # 改為 3e-4 (原本 1e-3 太大)
MAX_EPOCHS = 70
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP = 1.0
LOSS_WEIGHTS = (0.4, 0.6)  # (CE weight, Dice weight)

# Mixed Precision Training
USE_AMP = True  # 是否使用混合精度訓練
AMP_DTYPE = 'bfloat16'  # 'float16' or 'bfloat16' (bf16 更穩定)

# Class weights for handling imbalance (背景權重 0.2)
CLASS_WEIGHTS = [0.2, 1.0, 1.0]  # [background, class1, class2]

# Post-processing parameters (使用對方的形態學方法)
# Morphological operations: Open (remove noise) + Close (fill holes)
POST_PROCESS_METHOD = 'morphological'  # 'morphological' 或 'none'
POST_PROCESS_KERNEL_SIZE = 5  # 形態學操作的 kernel 大小 (3-7)
POST_PROCESS_ITERATIONS = 1  # 形態學操作的迭代次數
USE_TTA = True  # 是否使用測試時增強 (Test-Time Augmentation)

# Paths
CHECKPOINT_DIR = 'ckpt'
LOG_DIR = 'logs'
OUTPUT_FILE = 'output.csv'
