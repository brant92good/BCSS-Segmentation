#!/usr/bin/env python3
"""
僅進行預測的腳本 - 跳過訓練，直接使用已保存的最佳模型
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from config import (
    TEST_IMAGE_PATH, MEAN, STD, N_CHANNELS, N_CLASSES, USE_ATTENTION,
    OUTPUT_FILE, POST_PROCESS_MIN_SIZE, POST_PROCESS_KERNEL_SIZE,
    POST_PROCESS_FILL_HOLES, USE_TTA
)

from dataset import BCSSDatasetTest, create_df
from model import UNet
from utils import setup_gpu, load_checkpoint
from postprocess import post_process_mask, test_time_augmentation


def predict_image(model, image, device):
    """Predict segmentation mask for single image."""
    model.eval()
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0).numpy()
    
    return masked


def predict_test_set(model, device, X_test):
    """Predict on test set and prepare submission with post-processing."""
    print("\n" + "="*80)
    print("PREDICTING ON TEST SET")
    print("="*80)
    print(f"Post-processing enabled: min_size={POST_PROCESS_MIN_SIZE}, kernel={POST_PROCESS_KERNEL_SIZE}")
    print(f"Test-Time Augmentation: {USE_TTA}")
    
    test_set = BCSSDatasetTest(TEST_IMAGE_PATH, X_test, MEAN, STD)
    
    data = []
    for i in tqdm(range(len(test_set)), desc="Predicting"):
        img, filename = test_set[i]
        
        # Predict with optional TTA
        if USE_TTA:
            pred_mask = test_time_augmentation(model, img, device, num_augments=4)
        else:
            pred_mask = predict_image(model, img, device)
        
        # Apply post-processing (後處理)
        pred_mask = post_process_mask(
            pred_mask,
            min_size=POST_PROCESS_MIN_SIZE,
            kernel_size=POST_PROCESS_KERNEL_SIZE,
            fill_holes_flag=POST_PROCESS_FILL_HOLES
        )
        
        data.append({'index': filename, 'pred_mask': pred_mask.tolist()})
    
    df = pd.DataFrame(data)
    os.makedirs('output', exist_ok=True)
    output_path = os.path.join('output', OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return output_path


def main():
    """Main execution for prediction only."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using trained BCSS model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--no-tta', action='store_true', 
                       help='Disable test-time augmentation for faster prediction')
    args = parser.parse_args()
    
    # Setup
    device, has_gpu = setup_gpu()
    print(f"Using device: {device}")
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    print(f"\n📦 Loading model from: {args.model}")
    
    # Load model
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, use_attention=USE_ATTENTION)
    checkpoint = load_checkpoint(model, args.model, device=device)
    model.to(device)
    model.eval()
    
    # Print model info
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    if 'val_iou' in checkpoint:
        print(f"   Val mIoU: {checkpoint['val_iou']:.4f}")
    
    # Override TTA if specified
    global USE_TTA
    if args.no_tta:
        USE_TTA = False
        print("\n⚠️  Test-Time Augmentation disabled (faster but may be less accurate)")
    
    # Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    test_df = create_df(TEST_IMAGE_PATH)
    X_test = test_df['id'].to_numpy()
    print(f"Total Test Images: {len(X_test)}")
    
    if len(X_test) == 0:
        print("❌ No test images found!")
        sys.exit(1)
    
    # Predict
    csv_path = predict_test_set(model, device, X_test)
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE")
    print("="*80)
    print(f"Output file: {csv_path}")
    print()


if __name__ == '__main__':
    main()
