"""Post-processing utilities for semantic segmentation."""

import numpy as np
import cv2
from scipy import ndimage


def remove_small_objects(mask, min_size=50):
    """
    Remove small connected components from mask.
    
    Args:
        mask: 2D numpy array of predicted mask
        min_size: Minimum object size to keep (in pixels)
        
    Returns:
        Cleaned mask
    """
    output = np.zeros_like(mask)
    
    for class_id in np.unique(mask):
        if class_id == 0:  # Skip background
            continue
            
        binary_mask = (mask == class_id).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Keep only large components
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = class_id
    
    return output


def morphological_cleanup(mask, kernel_size=3):
    """
    Apply morphological operations to clean up mask.
    
    Args:
        mask: 2D numpy array of predicted mask
        kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    output = np.zeros_like(mask)
    
    for class_id in np.unique(mask):
        if class_id == 0:  # Skip background
            continue
            
        binary_mask = (mask == class_id).astype(np.uint8)
        
        # Close small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        output[binary_mask > 0] = class_id
    
    return output


def fill_holes(mask):
    """
    Fill holes in each class mask.
    
    Args:
        mask: 2D numpy array of predicted mask
        
    Returns:
        Mask with filled holes
    """
    output = np.zeros_like(mask)
    
    for class_id in np.unique(mask):
        if class_id == 0:  # Skip background
            continue
            
        binary_mask = (mask == class_id).astype(bool)
        filled = ndimage.binary_fill_holes(binary_mask)
        output[filled] = class_id
    
    return output


def post_process_mask(mask, min_size=30, kernel_size=3, fill_holes_flag=True, 
                     min_size_per_class=None):
    """
    Apply full post-processing pipeline to mask.
    
    Args:
        mask: 2D numpy array of predicted mask
        min_size: Minimum object size to keep (default for all classes)
        kernel_size: Size of morphological kernel
        fill_holes_flag: Whether to fill holes
        min_size_per_class: Optional dict {class_id: min_size} for class-specific thresholds
        
    Returns:
        Post-processed mask
        
    Examples:
        # Same threshold for all classes
        mask = post_process_mask(mask, min_size=30)
        
        # Different threshold per class
        mask = post_process_mask(mask, min_size_per_class={1: 20, 2: 50})
    """
    # Step 1: Morphological cleanup
    mask = morphological_cleanup(mask, kernel_size)
    
    # Step 2: Fill holes
    if fill_holes_flag:
        mask = fill_holes(mask)
    
    # Step 3: Remove small objects
    if min_size_per_class is not None:
        # Use class-specific thresholds
        output = np.zeros_like(mask)
        for class_id, class_min_size in min_size_per_class.items():
            if class_id == 0:  # Skip background
                continue
            binary_mask = (mask == class_id).astype(np.uint8)
            cleaned = remove_small_objects(
                np.where(binary_mask > 0, class_id, 0), 
                min_size=class_min_size
            )
            output[cleaned == class_id] = class_id
        mask = output
    else:
        # Use same threshold for all classes
        mask = remove_small_objects(mask, min_size)
    
    return mask


def test_time_augmentation(model, image, device, num_augments=4):
    """
    Apply test-time augmentation for better predictions.
    
    Args:
        model: Trained model
        image: Input image tensor [C, H, W]
        device: Device to run on
        num_augments: Number of augmentations
        
    Returns:
        Averaged prediction mask
    """
    import torch
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        img = image.unsqueeze(0).to(device)
        output = model(img)
        predictions.append(torch.softmax(output, dim=1).cpu())
        
        if num_augments >= 2:
            # Horizontal flip
            img_flip = torch.flip(image, [2]).unsqueeze(0).to(device)
            output_flip = model(img_flip)
            output_flip = torch.flip(output_flip, [3])
            predictions.append(torch.softmax(output_flip, dim=1).cpu())
        
        if num_augments >= 3:
            # Vertical flip
            img_vflip = torch.flip(image, [1]).unsqueeze(0).to(device)
            output_vflip = model(img_vflip)
            output_vflip = torch.flip(output_vflip, [2])
            predictions.append(torch.softmax(output_vflip, dim=1).cpu())
        
        if num_augments >= 4:
            # Both flips
            img_both = torch.flip(image, [1, 2]).unsqueeze(0).to(device)
            output_both = model(img_both)
            output_both = torch.flip(output_both, [2, 3])
            predictions.append(torch.softmax(output_both, dim=1).cpu())
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    mask = torch.argmax(avg_pred, dim=1).squeeze(0).numpy()
    
    return mask
