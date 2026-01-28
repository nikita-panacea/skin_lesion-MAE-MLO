# utils/losses.py
"""
Loss functions for MAE-MLO training

CRITICAL: mask_prob must flow gradients to masking module!
Based on MLO-MAE paper pattern where loss is weighted by mask_prob
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred, target, mask, mask_prob, norm_pix_loss=True):
    """
    Compute MAE reconstruction loss weighted by masking probabilities
    
    Following MLO-MAE paper (Eq. 1):
    Loss = Σ_j σ(M_j) * L_rec(M_j)
    where σ(M_j) is the masking probability (mask_prob)
    
    Args:
        pred: [B, N, patch_size^2 * 3] - predicted patches
        target: [B, N, patch_size^2 * 3] - target patches
        mask: [B, N] - binary mask (0=keep, 1=remove)
        mask_prob: [B, N] - masking probabilities (MUST have gradients!)
        norm_pix_loss: bool - normalize pixel loss
    
    Returns:
        loss: scalar - weighted reconstruction loss
    """
    if norm_pix_loss:
        # Normalize targets (per patch)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
    
    # Compute per-patch loss
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [B, N] - mean loss per patch
    
    # CRITICAL: Weight by mask_prob (this is what allows gradients to flow to masking module)
    # This is the key difference from standard MAE!
    loss = loss * mask_prob
    
    # Average over masked patches only
    loss = (loss * mask).sum() / mask.sum()
    
    return loss


def classification_loss(logits, labels, class_weights=None):
    """
    Compute classification loss with optional class weights
    
    Args:
        logits: [B, num_classes] - predicted logits
        labels: [B] - ground truth labels
        class_weights: [num_classes] - class weights for imbalanced data
    
    Returns:
        loss: scalar - classification loss
    """
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion(logits, labels)


def mixup_classification_loss(logits, labels_a, labels_b, lam, class_weights=None):
    """
    Compute mixup classification loss
    
    Args:
        logits: [B, num_classes] - predicted logits
        labels_a: [B] - first set of labels
        labels_b: [B] - second set of labels
        lam: float - mixup lambda
        class_weights: [num_classes] - class weights
    
    Returns:
        loss: scalar - mixup loss
    """
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
    return loss