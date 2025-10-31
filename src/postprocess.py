"""Post-processing utilities for semantic segmentation - Morphological method."""

import numpy as np
import cv2


def morphological_post_process(mask, kernel_size=8, iterations=1, n_classes=3):
    """
    使用形態學操作清理 mask (對方的方法)
    
    Process:
    1. Open operation: Remove small noise
    2. Close operation: Fill small holes
    
    Args:
        mask: 2D numpy array of predicted mask
        kernel_size: Size of morphological kernel (default: 8)
        iterations: Number of iterations for morphological operations
        n_classes: Number of classes (including background)
        
    Returns:
        Cleaned mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    final_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for cls in range(1, n_classes):  # Skip background (0)
        # Extract binary mask for this class
        binary_mask = (mask == cls).astype(np.uint8)
        
        # Open: Remove small noise
        opened_mask = cv2.morphologyEx(
            binary_mask, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=iterations
        )
        
        # Close: Fill small holes
        closed_mask = cv2.morphologyEx(
            opened_mask, 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=iterations
        )
        
        # Assign back to final mask
        final_mask[closed_mask == 1] = cls
    
    return final_mask


def post_process_mask(mask, method='morphological', kernel_size=8, iterations=1, n_classes=3):
    """
    Apply post-processing to mask.
    
    Args:
        mask: 2D numpy array of predicted mask
        method: 'morphological' or 'none'
        kernel_size: Size of morphological kernel
        iterations: Number of iterations
        n_classes: Number of classes
        
    Returns:
        Post-processed mask
    """
    if method == 'morphological':
        return morphological_post_process(mask, kernel_size, iterations, n_classes)
    else:
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
