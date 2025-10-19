"""Dataset classes for BCSS semantic segmentation."""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class BCSSDataset(Dataset):
    """Dataset class for loading BCSS images and masks with augmentation."""

    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        """
        Initialize BCSS dataset.
        
        Args:
            img_path: Path to image directory
            mask_path: Path to mask directory
            X: Array of image IDs
            mean: Normalization mean values
            std: Normalization std values
            transform: Albumentations transform pipeline
        """
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load image and mask
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        # Apply augmentation
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        # Normalize
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        return img, mask


class BCSSDatasetTest(Dataset):
    """Test dataset without masks."""

    def __init__(self, img_path, X, mean, std):
        """
        Initialize test dataset.
        
        Args:
            img_path: Path to image directory
            X: Array of image IDs
            mean: Normalization mean values
            std: Normalization std values
        """
        self.img_path = img_path
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        
        return img, self.X[idx]


def create_df(IMAGE_PATH):
    """
    Create a DataFrame of image IDs from a directory.
    
    Args:
        IMAGE_PATH: Path to image directory
        
    Returns:
        DataFrame with 'id' column containing image filenames
    """
    name = []
    
    # Check if path exists
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Path does not exist: {IMAGE_PATH}")
        return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))
    
    # Walk through directory and find PNG files
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            if filename.endswith('.png'):
                name.append(filename.split('.')[0])
    
    print(f"Found {len(name)} images in {IMAGE_PATH}")
    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))
