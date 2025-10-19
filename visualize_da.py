"""Data augmentation visualization script."""

import os
import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from config import TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, MEAN, STD, N_CLASSES
from src.dataset import create_df
from src.augmentation import get_train_transforms


def visualize_augmentations(img_idx=0, num_examples=4, output_dir='output'):
    """
    Visualize original image/mask and augmented variations.
    
    Args:
        img_idx: Index of training image to visualize
        num_examples: Number of augmented examples to show
        output_dir: Directory to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image and mask IDs
    train_df = create_df(TRAIN_IMAGE_PATH)
    X_train = train_df['id'].to_numpy()
    
    if img_idx >= len(X_train):
        print(f"Image index {img_idx} out of range. Max index: {len(X_train)-1}")
        return
    
    # Load original image and mask
    img_path = TRAIN_IMAGE_PATH + X_train[img_idx] + '.png'
    mask_path = TRAIN_MASK_PATH + X_train[img_idx] + '.png'
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Image or mask not found for index {img_idx}")
        return
    
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Get augmentation pipeline
    transforms_train = get_train_transforms()
    
    # Create figure
    fig, axes = plt.subplots(2, num_examples + 1, figsize=(4 * (num_examples + 1), 8))
    
    # Color map for masks
    colors = ['black', 'red', 'green', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors[:N_CLASSES])
    
    # Column 0: Original
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_mask, cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    axes[1, 0].set_title('Original Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Columns 1+: Augmented examples
    for col in range(1, num_examples + 1):
        aug = transforms_train(image=original_img.copy(), mask=original_mask.copy())
        aug_img = aug['image']
        aug_mask = aug['mask']
        
        axes[0, col].imshow(aug_img)
        axes[0, col].set_title(f'Augmented #{col}', fontsize=11)
        axes[0, col].axis('off')
        
        axes[1, col].imshow(aug_mask, cmap=cmap, vmin=0, vmax=N_CLASSES-1)
        axes[1, col].set_title(f'Mask #{col}', fontsize=11)
        axes[1, col].axis('off')
    
    plt.suptitle(f'Data Augmentation Examples: {X_train[img_idx]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'augmentation_example_{img_idx}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    print(f"Image ID: {X_train[img_idx]}")
    print(f"Image shape: {original_img.shape}")
    print(f"Mask classes: {np.unique(original_mask)}")
    
    plt.close()


def visualize_augmentations_grid(num_images=9, output_dir='output'):
    """
    Visualize augmentations for multiple images in a grid.
    
    Args:
        num_images: Number of images to show
        output_dir: Directory to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image IDs
    train_df = create_df(TRAIN_IMAGE_PATH)
    X_train = train_df['id'].to_numpy()
    
    if num_images > len(X_train):
        num_images = len(X_train)
        print(f"Limiting to {num_images} available images")
    
    # Get augmentation pipeline
    transforms_train = get_train_transforms()
    
    # Create grid
    sqrt_n = int(np.sqrt(num_images))
    fig, axes = plt.subplots(sqrt_n, sqrt_n, figsize=(3 * sqrt_n, 3 * sqrt_n))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx in range(num_images):
        img_path = TRAIN_IMAGE_PATH + X_train[idx] + '.png'
        mask_path = TRAIN_MASK_PATH + X_train[idx] + '.png'
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Skipping image {idx}: file not found")
            axes[idx].axis('off')
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply random augmentation
        aug = transforms_train(image=img, mask=mask)
        aug_img = aug['image']
        
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Augmented Sample {idx}', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Data Augmentation Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'augmentation_grid.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Grid visualization saved to {save_path}")
    
    plt.close()


def visualize_dataset_statistics(output_dir='output'):
    """
    Visualize dataset statistics (image shapes, mask distributions).
    
    Args:
        output_dir: Directory to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = create_df(TRAIN_IMAGE_PATH)
    X_train = train_df['id'].to_numpy()
    
    # Sample statistics
    sample_size = min(100, len(X_train))
    class_counts = np.zeros(N_CLASSES)
    image_heights = []
    image_widths = []
    
    print(f"Analyzing {sample_size} images...")
    for idx in range(sample_size):
        img_path = TRAIN_IMAGE_PATH + X_train[idx] + '.png'
        mask_path = TRAIN_MASK_PATH + X_train[idx] + '.png'
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            image_heights.append(img.shape[0])
            image_widths.append(img.shape[1])
            
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u < N_CLASSES:
                    class_counts[u] += c
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Image dimensions
    axes[0, 0].hist([image_heights, image_widths], label=['Height', 'Width'], bins=20)
    axes[0, 0].set_title('Image Dimensions Distribution')
    axes[0, 0].set_xlabel('Pixels')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid()
    
    # Class distribution
    class_names = ['Background', 'Class 1', 'Class 2'][:N_CLASSES]
    axes[0, 1].bar(class_names, class_counts)
    axes[0, 1].set_title('Pixel Count by Class')
    axes[0, 1].set_ylabel('Pixel Count')
    axes[0, 1].grid(axis='y')
    
    # Class proportion pie chart
    axes[1, 0].pie(class_counts, labels=class_names, autopct='%1.1f%%')
    axes[1, 0].set_title('Class Proportion')
    
    # Dataset split
    train_count = len(create_df(TRAIN_IMAGE_PATH))
    val_count = len(create_df(TRAIN_IMAGE_PATH.replace('train', 'val')))
    axes[1, 1].bar(['Train', 'Val'], [train_count, val_count])
    axes[1, 1].set_title('Dataset Split')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].grid(axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'dataset_statistics.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Statistics saved to {save_path}")
    
    plt.close()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Visualize data augmentation')
    parser.add_argument('--mode', choices=['single', 'grid', 'stats'], default='single',
                        help='Visualization mode')
    parser.add_argument('--idx', type=int, default=0, help='Image index to visualize (single mode)')
    parser.add_argument('--num-examples', type=int, default=4, help='Number of augmented examples')
    parser.add_argument('--num-images', type=int, default=9, help='Number of images in grid')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DATA AUGMENTATION VISUALIZATION")
    print("="*80)
    
    if args.mode == 'single':
        print(f"Visualizing single image (index={args.idx})")
        visualize_augmentations(
            img_idx=args.idx,
            num_examples=args.num_examples,
            output_dir=args.output_dir
        )
    elif args.mode == 'grid':
        print(f"Visualizing grid ({args.num_images} images)")
        visualize_augmentations_grid(
            num_images=args.num_images,
            output_dir=args.output_dir
        )
    elif args.mode == 'stats':
        print("Visualizing dataset statistics")
        visualize_dataset_statistics(output_dir=args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    main()
