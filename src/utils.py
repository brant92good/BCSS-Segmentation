"""Utility functions."""

import os
import torch
import numpy as np
import pynvml


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_gpu_stats():
    """Get GPU memory usage and utilization."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_percent = (mem_info.used / mem_info.total) * 100
        mem_used_gb = mem_info.used / (1024 ** 3)
        return mem_used_percent, mem_used_gb
    except:
        return 0.0, 0.0


def setup_gpu():
    """Setup GPU and return device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        pynvml.nvmlInit()
        has_gpu = torch.cuda.is_available()
    except:
        has_gpu = False
    
    return device, has_gpu


def create_checkpoint_dir(base_dir='ckpt'):
    """Create timestamped checkpoint directory."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(model_save_dir, exist_ok=True)
    return model_save_dir, timestamp


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics (val_loss, val_iou, etc.)
        save_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint.update(metrics)
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to restore state
        device: Device to load on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def predict_image(model, image, device):
    """
    Predict segmentation mask for single image.
    
    Args:
        model: Trained model
        image: Input image tensor
        device: Device to run on
        
    Returns:
        Prediction mask
    """
    model.eval()
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0).numpy()
    
    return masked
