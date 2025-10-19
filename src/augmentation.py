"""Data augmentation pipelines."""

import albumentations as A


def get_train_transforms():
    """
    Get training data augmentation pipeline.
    
    Enhanced augmentation (~2.5x stronger) for medical image segmentation.
    """
    return A.Compose([
        # 基本幾何變換 (提高機率)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 增強幾何變換
        A.ShiftScaleRotate(
            shift_limit=0.1,        # 增加平移 (0.05 -> 0.1)
            scale_limit=0.15,       # 增加縮放 (0.1 -> 0.15)
            rotate_limit=20,        # 增加旋轉 (10 -> 20)
            border_mode=0,
            p=0.5                   # 提高機率 (0.3 -> 0.5)
        ),
        
        # 增強顏色變換
        A.RandomBrightnessContrast(
            brightness_limit=0.2,   # 增強 (0.1 -> 0.2)
            contrast_limit=0.2,     # 增強 (0.1 -> 0.2)
            p=0.5                   # 提高機率 (0.3 -> 0.5)
        ),
        
        # 新增更多增強
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),  # 增強 (15 -> 20)
        
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
        ], p=0.3),
        
        A.CLAHE(p=0.2),  # 對比度限制自適應直方圖均衡化
        A.Blur(blur_limit=3, p=0.1),
    ])


def get_val_transforms():
    """
    Get validation data augmentation pipeline.
    
    No augmentation for validation/test data.
    """
    return A.Compose([])
