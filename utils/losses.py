"""
Loss functions for MAE-MLO training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred, target, mask, mask_prob=None, norm_pix_loss=True):
    """
    Compute MAE reconstruction loss
    Args:
        pred: [B, N, patch_size^2 * 3] - predicted patches
        target: [B, N, patch_size^2 * 3] - target patches
        mask: [B, N] - binary mask (0=keep, 1=remove)
        mask_prob: [B, N] - masking probabilities (optional)
        norm_pix_loss: bool - normalize pixels
    Returns:
        loss: scalar
    """
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
    
    # MSE loss
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [B, N]
    
    # Weight by mask probability if provided
    if mask_prob is not None:
        loss = loss * mask_prob
    
    # Mean loss on masked patches only
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    
    return loss


def classification_loss(logits, labels, class_weights=None):
    """
    Compute classification loss with optional class weighting
    Args:
        logits: [B, num_classes] - predicted logits
        labels: [B] - ground truth labels
        class_weights: [num_classes] - class weights (optional)
    Returns:
        loss: scalar
    """
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    loss = criterion(logits, labels)
    return loss


def mixup_classification_loss(logits, labels_a, labels_b, lam, class_weights=None):
    """
    Compute classification loss with mixup
    Args:
        logits: [B, num_classes]
        labels_a: [B] - first set of labels
        labels_b: [B] - second set of labels
        lam: float - mixup coefficient
        class_weights: [num_classes] - class weights
    Returns:
        loss: scalar
    """
    loss_a = classification_loss(logits, labels_a, class_weights)
    loss_b = classification_loss(logits, labels_b, class_weights)
    
    return lam * loss_a + (1 - lam) * loss_b