"""Loss functions and metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss for handling class imbalance."""
    
    def __init__(self, weight=None):
        """
        Initialize Weighted CE Loss.
        
        Args:
            weight: Tensor of class weights [background, class1, class2, ...]
                   Example: [0.2, 1.0, 1.0] means background has 0.2 weight
        """
        super().__init__()
        self.weight = weight
        
    def forward(self, inputs, targets):
        """
        Calculate weighted cross-entropy loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Target values
            
        Returns:
            Loss value
        """
        return F.cross_entropy(inputs, targets, weight=self.weight)


class DiceLoss(nn.Module):
    """Focal Dice loss for handling class imbalance."""

    def __init__(self, gamma=2.0):
        """
        Initialize Focal Dice Loss.
        
        Args:
            gamma: Focal parameter for focusing on hard examples
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets, eps=1e-6):
        """
        Calculate focal dice loss.

        Args:
            inputs: Model predictions (logits)
            targets: Target values
            eps: Stability factor
            
        Returns:
            Loss value
        """
        num_classes = inputs.size(1)
        probas = F.softmax(inputs, dim=1)
        true_1_hot = F.one_hot(targets.long(), num_classes=num_classes)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float().to(inputs.device)

        dims = (0,) + tuple(range(2, targets.ndimension() + 1))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice = (2. * intersection + eps) / (cardinality + eps)
        
        focal_weight = (1 - dice) ** self.gamma
        dice_loss = 1.0 - dice
        
        return (focal_weight * dice_loss).mean()


def pixel_accuracy(output, mask):
    """
    Calculate pixel accuracy.
    
    Args:
        output: Model output logits
        mask: Ground truth mask
        
    Returns:
        Pixel accuracy ratio
    """
    with torch.no_grad():
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, n_classes=3, ignore_class=0):
    """
    Calculate mean Intersection over Union.
    
    For batch input, calculates per-image mIoU and averages across the batch.
    
    Args:
        pred_mask: Predicted masks (logits)
        mask: Ground truth masks
        n_classes: Number of classes
        ignore_class: Class to ignore in calculation
        
    Returns:
        Mean IoU score
    """
    with torch.no_grad():
        probs = F.softmax(pred_mask, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = preds.shape[0]
        batch_miou_scores = []

        for b in range(batch_size):
            pred_img = preds[b]
            mask_img = mask[b]

            iou_list = []
            
            for cls in range(n_classes):
                pred_c = (pred_img == cls)
                label_c = (mask_img == cls)

                intersection = (pred_c & label_c).sum().float()
                union = (pred_c | label_c).sum().float()

                if union == 0:
                    continue

                iou = intersection / union

                if cls != ignore_class:
                    iou_list.append(iou)

            if len(iou_list) > 0:
                img_miou = torch.stack(iou_list).mean()
                batch_miou_scores.append(img_miou)

        if len(batch_miou_scores) > 0:
            return float(torch.stack(batch_miou_scores).mean().item())
        else:
            return 0.0
