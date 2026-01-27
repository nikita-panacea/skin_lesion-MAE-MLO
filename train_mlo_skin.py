"""
Main training script for MAE-MLO with Hybrid Encoder
End-to-end training for skin lesion classification

FIXES (based on MLO-MAE paper implementation):
1. Fixed module references: Use self.problem.module for Betty framework
2. Fixed dependencies to match paper: l2u: {mae: [cls, mask], cls: [mask]}, u2l: {mask: [mae]}
3. Fixed masking module access pattern
4. Removed double .module references (not needed in our implementation)
"""
import os
import sys
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

from config import get_args
from datasets.isic_dataset import ISICDataset, ISIC_CLASSES, get_transforms
from models.mae_hybrid import MAEHybrid
from models.hybrid_classifier import HybridClassifier
from models.masking_module import UNetMaskingModule
from utils.losses import reconstruction_loss, classification_loss, mixup_classification_loss
from utils.metrics import compute_metrics, print_metrics


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation"""
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class MAEProblem(ImplicitProblem):
    """
    MAE reconstruction problem (Level 1 - Lowest)
    
    This is the lowest level that:
    1. Takes images as input
    2. Uses masking module from upper level to generate masks
    3. Reconstructs masked patches
    """
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        
        # Embed patches
        patches = self.module.patch_embed(images)
        
        # Generate masks using masking module from upper level
        # Access pattern: self.mask.module (Betty's dependency system)
        x_masked, mask, ids_restore, mask_prob = self.mask.module(
            images, 
            patches,
            mask_ratio=self.mask_ratio,
            random=self.random_mask
        )
        
        # Forward through MAE encoder and decoder
        pred = self.module(x_masked, mask, ids_restore, mask_prob)
        
        # Compute reconstruction loss
        target = self.module.patchify(images)
        loss = reconstruction_loss(
            pred, target, mask, mask_prob,
            norm_pix_loss=self.module.norm_pix_loss
        )
        
        if self.is_rank_zero():
            wandb.log({'mae/loss': loss.item()})
        
        return loss


class ClassifierProblem(ImplicitProblem):
    """
    Classification problem (Level 2 - Middle)
    
    This middle level:
    1. Trains classifier on real images
    2. Also trains on reconstructed (fake) images from MAE
    3. Uses MAE from lower level and masking from upper level
    """
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Real images - forward pass
        logits_real = self.module(images)
        loss_real = classification_loss(logits_real, labels, self.class_weights)
        
        # Fake (reconstructed) images
        # NO torch.no_grad() to maintain gradient flow for MLO
        
        # Get patches from MAE
        patches = self.mae.module.patch_embed(images)
        
        # Mask using masking module from upper level
        x_masked, mask, ids_restore, mask_prob = self.mask.module(
            images, patches,
            mask_ratio=self.mask_ratio,
            random=self.random_mask
        )
        
        # Reconstruct using MAE from lower level
        pred = self.mae.module(x_masked, mask, ids_restore, mask_prob)
        
        # Unpatchify to get reconstructed images
        fake_images = self.mae.module.unpatchify(pred)
        
        # Normalize fake images to match input distribution
        fake_images = torch.clamp(fake_images, 0, 1)
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        fake_images = (fake_images - mean) / std
        
        # Detach to save memory (gradients for MAE come from mae_problem)
        fake_images = fake_images.detach()
        
        # Classify fake images
        logits_fake = self.module(fake_images)
        loss_fake = classification_loss(logits_fake, labels, self.class_weights)
        
        # Combined loss
        loss = loss_real + self.loss_lambda * loss_fake
        
        if self.is_rank_zero():
            wandb.log({
                'classifier/loss': loss.item(),
                'classifier/loss_real': loss_real.item(),
                'classifier/loss_fake': loss_fake.item()
            })
        
        return loss


class MaskingProblem(ImplicitProblem):
    """
    Masking module optimization (Level 3 - Highest)
    
    This highest level:
    1. Uses validation data
    2. Optimizes masking strategy based on classifier validation performance
    3. Provides feedback to lower levels (MAE and Classifier)
    """
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward through classifier from lower level
        logits = self.classifier.module(images)
        
        # Validation loss - guides masking optimization
        loss = classification_loss(logits, labels, self.class_weights)
        
        if self.is_rank_zero():
            wandb.log({'masking/loss': loss.item()})
        
        return loss


class MLOEngine(Engine):
    """Custom engine with validation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_loader = None
        self.num_classes = None
        self.best_f1 = 0.0
        self.save_checkpoint_fn = None
    
    @torch.no_grad()
    def validation(self):
        """Validate on validation set"""
        if self.val_loader is None:
            return
        
        # Find classifier problem
        classifier_problem = None
        for problem in self.problems:
            if problem.name == 'classifier':
                classifier_problem = problem
                break
        
        if classifier_problem is None:
            return
        
        classifier_problem.module.eval()
        
        all_preds = []
        all_targets = []
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = classifier_problem.module(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, num_classes=self.num_classes)
        
        # Log metrics
        if self.is_rank_zero():
            wandb.log({
                'val/accuracy': metrics['accuracy'],
                'val/precision': metrics['precision_macro'],
                'val/recall': metrics['recall_macro'],
                'val/f1': metrics['f1_macro'],
                'step': self.global_step
            })
            
            print(f"\n[Validation @ Step {self.global_step}]")
            print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"  Precision: {metrics['precision_macro']*100:.2f}%")
            print(f"  Recall:    {metrics['recall_macro']*100:.2f}%")
            print(f"  F1-score:  {metrics['f1_macro']*100:.2f}%")
        
        # Save best model
        if metrics['f1_macro'] > self.best_f1:
            self.best_f1 = metrics['f1_macro']
            if self.is_rank_zero() and self.save_checkpoint_fn is not None:
                self.save_checkpoint_fn('best_model.pt', metrics)
        
        classifier_problem.module.train()


def main():
    args = get_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.wandb_mode != 'disabled':
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
            mode=args.wandb_mode
        )
    
    print(f"Using device: {device}")
    print(f"Experiment: {args.exp_name}")
    
    # Datasets
    train_transform = get_transforms(train=True, img_size=args.img_size)
    val_transform = get_transforms(train=False, img_size=args.img_size)
    
    train_dataset = ISICDataset(args.train_csv, args.img_dir, train_transform)
    val_dataset = ISICDataset(args.val_csv, args.img_dir, val_transform)
    
    print(f"\nDataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Models
    print("\nBuilding models...")
    
    # MAE
    mae_model = MAEHybrid(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        norm_pix_loss=True
    ).to(device)
    
    # Masking module
    masking_module = UNetMaskingModule(
        pretrained_unet_path=args.pretrained_unet,
        num_patches=(args.img_size // args.patch_size) ** 2,
        embed_dim=args.encoder_dim,
        learnable=not args.baseline
    ).to(device)
    
    # Classifier
    classifier = HybridClassifier(
        num_classes=args.num_classes,
        pretrained_mae=None,
        freeze_encoder=False
    ).to(device)
    
    print(f"\nModels:")
    print(f"  MAE parameters: {sum(p.numel() for p in mae_model.parameters())/1e6:.2f}M")
    print(f"  Masking parameters: {sum(p.numel() for p in masking_module.parameters())/1e6:.2f}M")
    print(f"  Classifier parameters: {sum(p.numel() for p in classifier.parameters())/1e6:.2f}M")
    
    # Optimizers
    optimizer_mae = torch.optim.AdamW(
        mae_model.parameters(),
        lr=args.mae_lr,
        betas=(0.9, 0.95),
        weight_decay=args.mae_weight_decay
    )
    
    optimizer_cls = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.cls_lr,
        betas=(0.9, 0.95),
        weight_decay=args.cls_weight_decay
    )
    
    optimizer_mask = torch.optim.AdamW(
        masking_module.parameters(),
        lr=args.mask_lr,
        betas=(0.9, 0.95),
        weight_decay=args.mask_weight_decay
    )
    
    # Compute class weights
    from collections import Counter
    label_counts = Counter(train_dataset.labels)
    total = len(train_dataset)
    class_weights = torch.tensor([
        total / (len(ISIC_CLASSES) * label_counts[i])
        for i in range(len(ISIC_CLASSES))
    ], dtype=torch.float32).to(device)
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    # Calculate training iterations
    iters_per_epoch = len(train_loader)
    train_iters = args.epochs * iters_per_epoch
    
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Iterations per epoch: {iters_per_epoch}")
    print(f"  Total iterations: {train_iters}")
    print(f"  Validation every: {args.valid_step} iterations")
    
    # Betty MLO setup
    mae_config = Config(
        type='darts',
        retain_graph=True,
        unroll_steps=args.unroll_steps_mae,
        log_step=args.log_freq,
        allow_unused=True
    )
    
    cls_config = Config(
        type='darts',
        retain_graph=True,
        unroll_steps=args.unroll_steps_cls,
        log_step=args.log_freq,
        allow_unused=True
    )
    
    mask_config = Config(
        type='darts',
        retain_graph=True,
        unroll_steps=args.unroll_steps_mask,
        log_step=args.log_freq,
        allow_unused=True
    )
    
    engine_config = EngineConfig(
        strategy='default',
        train_iters=train_iters,
        valid_step=args.valid_step,
        logger_type='tensorboard'
    )
    
    # Create problems
    mae_problem = MAEProblem(
        name='mae',
        module=mae_model,
        optimizer=optimizer_mae,
        train_data_loader=train_loader,
        config=mae_config,
        # device=device
    )
    mae_problem.mask_ratio = args.mask_ratio
    mae_problem.random_mask = args.random_mask or args.baseline
    
    cls_problem = ClassifierProblem(
        name='classifier',
        module=classifier,
        optimizer=optimizer_cls,
        train_data_loader=train_loader,
        config=cls_config,
        # device=device
    )
    cls_problem.class_weights = class_weights
    cls_problem.mask_ratio = args.mask_ratio
    cls_problem.random_mask = args.random_mask or args.baseline
    cls_problem.loss_lambda = args.loss_lambda
    
    mask_problem = MaskingProblem(
        name='masking',
        module=masking_module,
        optimizer=optimizer_mask,
        train_data_loader=val_loader,  # Uses validation data!
        config=mask_config,
        # device=device
    )
    mask_problem.class_weights = class_weights
    
    # Define dependencies (following MLO-MAE paper pattern)
    if args.baseline:
        # Baseline: no MLO, just MAE and classifier
        problems = [mae_problem, cls_problem]
        dependencies = {'l2u': {}, 'u2l': {}}
    else:
        # Full MLO following paper:
        # l2u: {mae: [cls, mask], cls: [mask]}  - lower to upper
        # u2l: {mask: [mae]}  - upper to lower
        problems = [mae_problem, cls_problem, mask_problem]
        dependencies = {
            'l2u': {
                mae_problem: [cls_problem, mask_problem],  # MAE feeds both classifier and masking
                cls_problem: [mask_problem]  # Classifier feeds masking
            },
            'u2l': {
                mask_problem: [mae_problem]  # Masking provides feedback to MAE
            }
        }
    
    # Create engine
    engine = MLOEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies
    )
    
    # Add attributes
    engine.val_loader = val_loader
    engine.num_classes = args.num_classes
    
    def save_checkpoint(filename, metrics=None):
        """Save checkpoint"""
        save_path = os.path.join(args.output_dir, filename)
        checkpoint = {
            'mae': mae_model.state_dict(),
            'classifier': classifier.state_dict(),
            'masking': masking_module.state_dict(),
            'optimizer_mae': optimizer_mae.state_dict(),
            'optimizer_cls': optimizer_cls.state_dict(),
            'optimizer_mask': optimizer_mask.state_dict(),
            'args': vars(args),
            'metrics': metrics
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    engine.save_checkpoint_fn = save_checkpoint
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    engine.run()
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Test set evaluation
    if args.test_csv:
        print("\nEvaluating on test set...")
        test_dataset = ISICDataset(args.test_csv, args.img_dir, val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        classifier.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = classifier(images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())
        
        metrics = compute_metrics(all_preds, all_targets, num_classes=args.num_classes)
        print_metrics(metrics, class_names=ISIC_CLASSES)
        
        if args.wandb_mode != 'disabled':
            wandb.log({
                'test/accuracy': metrics['accuracy'],
                'test/precision': metrics['precision_macro'],
                'test/recall': metrics['recall_macro'],
                'test/f1': metrics['f1_macro']
            })
    
    if args.wandb_mode != 'disabled':
        wandb.finish()


if __name__ == '__main__':
    main()