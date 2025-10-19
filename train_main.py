"""Main training script with validation and Kaggle upload."""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from config import (
    TRAIN_IMAGE_PATH, VAL_IMAGE_PATH, TRAIN_MASK_PATH, VAL_MASK_PATH, TEST_IMAGE_PATH,
    MEAN, STD, N_CHANNELS, N_CLASSES, USE_ATTENTION,
    BATCH_SIZE, ACCUMULATION_STEPS, NUM_WORKERS, MAX_LR, MAX_EPOCHS,
    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, GRAD_CLIP, LOSS_WEIGHTS,
    CHECKPOINT_DIR, LOG_DIR, OUTPUT_FILE,
    CLASS_WEIGHTS, POST_PROCESS_MIN_SIZE, POST_PROCESS_KERNEL_SIZE,
    POST_PROCESS_FILL_HOLES, USE_TTA
)

from dataset import BCSSDataset, BCSSDatasetTest, create_df
from model import UNet
from losses import DiceLoss, WeightedCrossEntropyLoss, pixel_accuracy, mIoU
from augmentation import get_train_transforms, get_val_transforms
from utils import (
    get_lr, get_gpu_stats, setup_gpu, create_checkpoint_dir,
    save_checkpoint, load_checkpoint, predict_image
)
from postprocess import post_process_mask, test_time_augmentation


def verify_data_paths():
    """Verify that all required data paths exist."""
    paths = {
        'Train': TRAIN_IMAGE_PATH,
        'Val': VAL_IMAGE_PATH,
        'Train Mask': TRAIN_MASK_PATH,
        'Val Mask': VAL_MASK_PATH,
    }
    
    all_exist = True
    for path_name, path in paths.items():
        if os.path.exists(path):
            num_files = len([f for f in os.listdir(path) if f.endswith('.png')])
            print(f"✓ {path_name} path exists with {num_files} files")
        else:
            print(f"✗ {path_name} path NOT FOUND: {path}")
            all_exist = False
    
    if not all_exist:
        raise ValueError("Some data paths are missing!")
    
    return all_exist


def setup_data():
    """Load and prepare datasets."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Create dataframes
    train_df = create_df(TRAIN_IMAGE_PATH)
    val_df = create_df(VAL_IMAGE_PATH)
    
    print(f'Total Train Images: {len(train_df)}')
    print(f'Total Val Images: {len(val_df)}')
    
    if len(train_df) == 0:
        raise ValueError(f"No training images found!")
    if len(val_df) == 0:
        raise ValueError(f"No validation images found!")
    
    X_train = train_df['id'].to_numpy()
    X_val = val_df['id'].to_numpy()
    
    # Create datasets
    transforms_train = get_train_transforms()
    transforms_val = get_val_transforms()
    
    train_set = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, MEAN, STD, transforms_train)
    val_set = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, X_val, MEAN, STD, transforms_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=16, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, val_loader, X_train, X_val


def train_epoch(model, train_loader, criterion1, criterion2, optimizer, scheduler,
                device, has_gpu, writer, epoch, grad_clip, loss_weights, accumulation_steps):
    """Train for one epoch."""
    model.train()
    running_loss = 0
    iou_score = 0
    accuracy = 0
    
    for i, data in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
        image, mask = data
        image = image.to(device)
        mask = mask.to(device)
        
        output = model(image)
        
        # Compute weighted loss
        loss1 = criterion1(output, mask)
        loss2 = criterion2(output, mask)
        loss = loss_weights[0] * loss1 + loss_weights[1] * loss2
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        
        iou_score += mIoU(output, mask)
        accuracy += pixel_accuracy(output, mask)
        
        loss.backward()
        
        # Update weights every accumulation_steps batches
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
            lrs = get_lr(optimizer)
            scheduler.step()
        
        running_loss += loss.item() * accumulation_steps
        
        # Log to TensorBoard
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/batch', loss.item() * accumulation_steps, global_step)
        writer.add_scalar('Loss/CE_batch', loss1.item(), global_step)
        writer.add_scalar('Loss/Dice_batch', loss2.item(), global_step)
        writer.add_scalar('LearningRate/batch', get_lr(optimizer), global_step)
        
        if has_gpu:
            gpu_mem_percent, gpu_mem_gb = get_gpu_stats()
            writer.add_scalar('Hardware/GPU_Memory_Percent', gpu_mem_percent, global_step)
            writer.add_scalar('Hardware/GPU_Memory_GB', gpu_mem_gb, global_step)
        
        if (i + 1) % 5 == 0:
            writer.flush()
    
    train_loss_avg = running_loss / len(train_loader)
    train_iou_avg = iou_score / len(train_loader)
    train_acc_avg = accuracy / len(train_loader)
    
    return train_loss_avg, train_iou_avg, train_acc_avg


def val_epoch(model, val_loader, criterion1, criterion2, device, writer, epoch, loss_weights):
    """Validate for one epoch."""
    model.eval()
    test_loss = 0
    test_accuracy = 0
    val_iou_score = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch}")):
            image, mask = data
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            
            val_iou_score += mIoU(output, mask)
            test_accuracy += pixel_accuracy(output, mask)
            
            loss1 = criterion1(output, mask)
            loss2 = criterion2(output, mask)
            loss = loss_weights[0] * loss1 + loss_weights[1] * loss2
            test_loss += loss.item()
    
    val_loss_avg = test_loss / len(val_loader)
    val_iou_avg = val_iou_score / len(val_loader)
    val_acc_avg = test_accuracy / len(val_loader)
    
    # Log to TensorBoard
    writer.add_scalar('Loss/val', val_loss_avg, epoch)
    writer.add_scalar('mIoU/val', val_iou_avg, epoch)
    writer.add_scalar('Accuracy/val', val_acc_avg, epoch)
    
    return val_loss_avg, val_iou_avg, val_acc_avg


def train(model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler,
          device, has_gpu, model_save_dir, timestamp):
    """Main training loop."""
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    # Initialize TensorBoard
    writer = SummaryWriter(os.path.join(LOG_DIR, f'BCSS_UNet_{timestamp}'))
    
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    
    min_loss = np.inf
    best_val_iou = 0.0
    epochs_no_improve = 0
    best_model_path = None
    
    model.to(device)
    fit_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        
        # Train
        train_loss_avg, train_iou_avg, train_acc_avg = train_epoch(
            model, train_loader, criterion1, criterion2, optimizer, scheduler,
            device, has_gpu, writer, epoch, GRAD_CLIP, LOSS_WEIGHTS, ACCUMULATION_STEPS
        )
        
        # Validate
        val_loss_avg, val_iou_avg, val_acc_avg = val_epoch(
            model, val_loader, criterion1, criterion2, device, writer, epoch, LOSS_WEIGHTS
        )
        
        # Logging
        train_losses.append(train_loss_avg)
        test_losses.append(val_loss_avg)
        train_iou.append(train_iou_avg)
        val_iou.append(val_iou_avg)
        train_acc.append(train_acc_avg)
        val_acc.append(val_acc_avg)
        
        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('mIoU/train', train_iou_avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc_avg, epoch)
        
        elapsed = time.time() - since
        
        print(f"Epoch:{epoch+1}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss_avg:.3f} | Val Loss: {val_loss_avg:.3f} | "
              f"Train mIoU: {train_iou_avg:.3f} | Val mIoU: {val_iou_avg:.3f} | "
              f"Train Acc: {train_acc_avg:.3f} | Val Acc: {val_acc_avg:.3f} | "
              f"Time: {elapsed/60:.2f}m")
        
        # Save best model
        if val_loss_avg < min_loss:
            print(f'Loss Decreasing.. {min_loss:.3f} >> {val_loss_avg:.3f}')
            min_loss = val_loss_avg
            best_val_iou = val_iou_avg
            epochs_no_improve = 0
            
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_model_path = os.path.join(
                model_save_dir,
                f'best_model_loss-{val_loss_avg:.4f}_mIoU-{val_iou_avg:.3f}.pt'
            )
            save_checkpoint(
                model, optimizer, epoch,
                {'val_loss': val_loss_avg, 'val_iou': val_iou_avg},
                best_model_path
            )
            print(f'Model saved to: {best_model_path}')
        else:
            epochs_no_improve += 1
            print(f'Early stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}')
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs')
            print(f'Best val loss: {min_loss:.4f}, Best mIoU: {best_val_iou:.3f}')
            break
    
    history = {
        'train_loss': train_losses, 'val_loss': test_losses,
        'train_miou': train_iou, 'val_miou': val_iou,
        'train_acc': train_acc, 'val_acc': val_acc,
    }
    
    writer.close()
    total_time = time.time() - fit_time
    print(f'Total training time: {total_time/60:.2f} minutes')
    print(f'Best model saved at: {best_model_path}')
    
    return history, best_model_path


def plot_training_history(history, save_dir='logs'):
    """Plot training history."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='o')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid()
    
    # mIoU
    axes[1].plot(history['train_miou'], label='Train mIoU', marker='*')
    axes[1].plot(history['val_miou'], label='Val mIoU', marker='*')
    axes[1].set_title('mIoU per Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid()
    
    # Accuracy
    axes[2].plot(history['train_acc'], label='Train Acc', marker='*')
    axes[2].plot(history['val_acc'], label='Val Acc', marker='*')
    axes[2].set_title('Accuracy per Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=100)
    print(f"Training history saved to {save_path}")
    plt.close()


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


def submit_to_kaggle(csv_path):
    """Submit predictions to Kaggle."""
    print("\n" + "="*80)
    print("SUBMITTING TO KAGGLE")
    print("="*80)
    
    try:
        import subprocess
        result = subprocess.run(
            [
                'kaggle', 'competitions', 'submit',
                '-c', 'lab-4-semantic-segmentation-on-bcss-639003',
                '-f', csv_path,
                '-m', f'UNet Attention Model - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
            ],
            capture_output=True,
            text=True
        )
        print("Kaggle submission output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Kaggle submission failed: {e}")
        print(f"Please manually upload {csv_path} to Kaggle")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Train BCSS semantic segmentation model')
    parser.add_argument('--no-submit', action='store_true', help='Skip Kaggle submission')
    args = parser.parse_args()
    
    # Setup
    device, has_gpu = setup_gpu()
    print(f"Using device: {device}")
    
    # Verify data
    verify_data_paths()
    
    # Create checkpoint directory
    model_save_dir, timestamp = create_checkpoint_dir(CHECKPOINT_DIR)
    print(f"\nModels will be saved to: {model_save_dir}")
    print(f"Timestamp: {timestamp}")
    
    # Load data
    train_loader, val_loader, X_train, X_val = setup_data()
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, use_attention=USE_ATTENTION)
    print(f"Model initialized with {N_CLASSES} classes")
    
    # Setup training with class weights (背景權重 0.2)
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    print(f"Class weights: {CLASS_WEIGHTS} (背景權重降低到 {CLASS_WEIGHTS[0]})")
    
    criterion1 = WeightedCrossEntropyLoss(weight=class_weights)
    criterion2 = DiceLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    
    effective_steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=MAX_EPOCHS,
        steps_per_epoch=effective_steps_per_epoch,
        pct_start=0.10, div_factor=10.0, final_div_factor=1e4,
        anneal_strategy='cos'
    )
    
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"Max LR: {MAX_LR}, Weight decay: {WEIGHT_DECAY}")
    
    # Train
    history, best_model_path = train(
        model, train_loader, val_loader, criterion1, criterion2,
        optimizer, scheduler, device, has_gpu, model_save_dir, timestamp
    )
    
    # Plot history
    plot_training_history(history)
    
    # Load best model for inference
    print(f"\nLoading best model from: {best_model_path}")
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, use_attention=USE_ATTENTION)
    load_checkpoint(model, best_model_path, device=device)
    model.to(device)
    model.eval()
    
    # Predict on test set
    test_df = create_df(TEST_IMAGE_PATH)
    X_test = test_df['id'].to_numpy()
    csv_path = predict_test_set(model, device, X_test)
    
    # Submit to Kaggle
    if not args.no_submit:
        submit_to_kaggle(csv_path)
    else:
        print(f"Skipping Kaggle submission. Results saved to {csv_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
