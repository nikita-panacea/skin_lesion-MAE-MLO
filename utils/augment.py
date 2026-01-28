# utils/augment.py
"""
Data augmentation aligned with paper methodology.
Paper mentions: scaling, smoothing, mix-up, color jitter, and flipping
"""
from typing import Optional
from torchvision import transforms
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(train: bool = True, input_size: int = 224, deterministic: bool = False):
    """
    Build transforms matching paper augmentation strategy.
    
    Paper mentions:
    - Scaling (RandomResizedCrop)
    - Smoothing (via interpolation in resize)
    - Color jitter
    - Flipping
    - Mix-up (applied in training loop, not here)
    
    Args:
        train: If True, apply training augmentations
        input_size: Target image size (paper uses 224)
        deterministic: If True, disable random augmentations (for debugging)
    """
    if deterministic:
        # Deterministic transforms for overfit debugging
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if train:
        # Training augmentations as per paper
        return transforms.Compose([
            # Scaling: Random resized crop with scale range
            transforms.RandomResizedCrop(
                input_size, 
                scale=(0.8, 1.0),  # 80-100% of original
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            
            # Flipping
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Color jitter for robustness
            transforms.ColorJitter(
                brightness=0.2,  # ±20%
                contrast=0.2,
                saturation=0.2,
                hue=0.02  # Small hue variation
            ),
            
            # Optional: Add slight rotation for skin lesion invariance
            transforms.RandomRotation(degrees=15),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Validation/test transforms: simple resize and normalize
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])