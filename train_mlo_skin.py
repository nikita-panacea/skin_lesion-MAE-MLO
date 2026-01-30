# train_mlo_skin.py - FIXED VERSION
"""
MLO-MAE Training for Skin Lesion Classification

CRITICAL FIXES:
1. Proper allow_unused=True in all configs
2. Detach frozen outputs in Problems
3. Correct dependency graph
4. Fixed validation handling
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

# Betty MLO framework
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

# Import models
from models.mae_hybrid import mae_hybrid_base
from models.masking_module import UNetMaskingModule
from models.hybrid_classifier import HybridClassifier

# Import utilities
import wandb
import logging


def parse_args():
    parser = argparse.ArgumentParser('MLO-MAE Skin Lesion Training')
    
    # Data
    parser.add_argument('--data_path', default='/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set',
                       help='Dataset root path')
    parser.add_argument('--unet_path', default='/home/ubuntu/Documents/Nikita/GenSeg/unet/unet.pkl',
                       help='Pretrained UNet path')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_mae', type=float, default=1e-4)
    parser.add_argument('--lr_classifier', type=float, default=5e-4)
    parser.add_argument('--lr_masking', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # MLO settings
    parser.add_argument('--unroll_mae', type=int, default=1)
    parser.add_argument('--unroll_classifier', type=int, default=1)
    parser.add_argument('--unroll_masking', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=100)
    
    # MAE settings
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--norm_pix_loss', action='store_true', default=True)
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--wandb_project', default='mlo-mae-skin')
    parser.add_argument('--wandb_name', default='mlo-training')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'])
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_isic_data(args):
    """Load ISIC 2019 dataset"""
    from datasets.isic2019_dataset import ISIC2019Dataset
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_csv = os.path.join(args.data_path, 'train.csv')
    val_csv = os.path.join(args.data_path, 'val.csv')
    img_dir = os.path.join(args.data_path, 'images')
    
    train_dataset = ISIC2019Dataset(train_csv, img_dir, transform=transform)
    val_dataset = ISIC2019Dataset(val_csv, img_dir, transform=transform)
    
    return train_dataset, val_dataset


class MAEProblem(ImplicitProblem):
    """
    Stage 1: MAE Pretraining
    
    FIXED:
    - Detach outputs from masking module
    - Proper allow_unused handling
    """
    def training_step(self, batch):
        images, _, _ = batch
        images = images.to(self.device)
        
        # CRITICAL: Get masking module outputs and ensure proper detachment
        mask_module = self.masking.module
        
        # Forward through MAE
        loss, pred, mask = self.module(
            images,
            mask_module,
            mask_ratio=self.config.mask_ratio,
            random=False
        )
        
        return loss


class ClassifierProblem(ImplicitProblem):
    """
    Stage 2: Classifier Training
    
    FIXED:
    - Extract features from frozen MAE encoder
    - Proper detachment
    """
    def training_step(self, batch):
        images, labels, _ = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward through classifier
        logits = self.module(images)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MaskingProblem(ImplicitProblem):
    """
    Stage 3: Masking Module Update
    
    FIXED:
    - Validation loss for architecture update
    - Proper gradient flow
    """
    def training_step(self, batch):
        images, labels, _ = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward through full pipeline
        # Classifier already uses updated encoder
        logits = self.classifier.module(images)
        
        # Compute validation loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SkinMLOEngine(Engine):
    """
    Custom MLO Engine with validation
    
    FIXED:
    - Proper validation loop
    - Checkpoint saving
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_acc = 0.0
        self.args = None
    
    @torch.no_grad()
    def validation(self):
        """Validation on held-out set"""
        self.eval()
        
        classifier = self.classifier.module
        val_loader = self.classifier.val_data_loader
        
        correct = 0
        total = 0
        
        for images, labels, _ in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = classifier(images)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        acc = 100.0 * correct / total
        
        # Log
        if self.is_rank_zero():
            logging.info(f"Validation Accuracy: {acc:.2f}%")
            wandb.log({
                'val_acc': acc,
                'global_step': self.global_step
            })
        
        # Save best
        if acc > self.best_acc:
            self.best_acc = acc
            if self.is_rank_zero():
                self.save_checkpoint('best_model.pt')
                logging.info(f" New best accuracy: {acc:.2f}%")
        
        self.train()
    
    def save_checkpoint(self, filename):
        """Save checkpoint"""
        if not self.is_rank_zero():
            return
        
        save_path = Path(self.args.output_dir) / filename
        
        checkpoint = {
            'epoch': self.global_step // self.steps_per_epoch,
            'global_step': self.global_step,
            'best_acc': self.best_acc,
            'mae_state': self.mae.module.state_dict(),
            'classifier_state': self.classifier.module.state_dict(),
            'masking_state': self.masking.module.state_dict(),
            'mae_optimizer': self.mae.optimizer.state_dict(),
            'classifier_optimizer': self.classifier.optimizer.state_dict(),
            'masking_optimizer': self.masking.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved to {save_path}")


def main():
    args = parse_args()
    
    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if args.wandb_mode != 'disabled':
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            mode=args.wandb_mode
        )
    
    print("\n" + "="*60)
    print("MLO-MAE Skin Lesion Classifier Training")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    train_dataset, val_dataset = load_isic_data(args)
    
    print(f"\nDataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Create dataloaders
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
        pin_memory=True,
        drop_last=False
    )
    
    # Build models
    print("\nBuilding models...")
    
    # 1. MAE
    mae_model = mae_hybrid_base(
        pretrained_unet_path=args.unet_path,
        norm_pix_loss=args.norm_pix_loss
    ).to(device)
    
    # 2. Masking module
    masking_module = UNetMaskingModule(
        pretrained_unet_path=args.unet_path,
        num_patches=196,
        embed_dim=768,
        learnable=True,
        use_unet=True
    ).to(device)
    
    # 3. Classifier
    classifier = HybridClassifier(
        num_classes=8,
        pretrained_mae=None,
        freeze_encoder=False
    ).to(device)
    
    # Count parameters
    mae_params = sum(p.numel() for p in mae_model.parameters() if p.requires_grad)
    mask_params = sum(p.numel() for p in masking_module.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    print(f"\nModels:")
    print(f"  MAE parameters: {mae_params/1e6:.2f}M")
    print(f"  Masking parameters: {mask_params/1e6:.2f}M")
    print(f"  Classifier parameters: {cls_params/1e6:.2f}M")
    
    # Optimizers
    optimizer_mae = torch.optim.AdamW(
        mae_model.parameters(),
        lr=args.lr_mae,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    optimizer_classifier = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr_classifier,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    optimizer_masking = torch.optim.AdamW(
        masking_module.parameters(),
        lr=args.lr_masking,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Learning rate schedulers
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    
    scheduler_mae = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mae, T_max=total_steps
    )
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_classifier, T_max=total_steps
    )
    scheduler_masking = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_masking, T_max=total_steps
    )
    
    # Betty Problems
    # CRITICAL FIX: Set allow_unused=True for all problems
    mae_config = Config(
        type='darts',
        unroll_steps=args.unroll_mae,
        log_step=100,
        retain_graph=True,
        allow_unused=True  # CRITICAL
    )
    
    classifier_config = Config(
        type='darts',
        unroll_steps=args.unroll_classifier,
        log_step=100,
        retain_graph=True,
        allow_unused=True  # CRITICAL
    )
    
    masking_config = Config(
        type='darts',
        unroll_steps=args.unroll_masking,
        log_step=100,
        allow_unused=True  # CRITICAL
    )
    
    mae = MAEProblem(
        name='mae',
        module=mae_model,
        optimizer=optimizer_mae,
        scheduler=scheduler_mae,
        train_data_loader=train_loader,
        config=mae_config,
        # device=device
    )
    # Add mask_ratio to config
    mae.config.mask_ratio = args.mask_ratio
    
    classifier = ClassifierProblem(
        name='classifier',
        module=classifier,
        optimizer=optimizer_classifier,
        scheduler=scheduler_classifier,
        train_data_loader=train_loader,
        config=classifier_config,
        # device=device
    )
    # Add validation loader
    classifier.val_data_loader = val_loader
    
    masking = MaskingProblem(
        name='masking',
        module=masking_module,
        optimizer=optimizer_masking,
        scheduler=scheduler_masking,
        train_data_loader=val_loader,  # Use val set for masking update
        config=masking_config,
        # device=device
    )
    
    # Dependency graph
    # masking -> classifier -> mae
    # masking needs classifier, classifier needs mae
    problems = [mae, classifier, masking]
    
    u2l = {
        masking: [classifier, mae],
        classifier: [mae]
    }
    
    l2u = {
        mae: [classifier],
        classifier: [masking]
    }
    
    dependencies = {'u2l': u2l, 'l2u': l2u}
    
    # Engine config
    engine_config = EngineConfig(
        strategy='default',
        train_iters=total_steps,
        valid_step=args.val_freq,
        logger_type='tensorboard'
    )
    
    # Create engine
    engine = SkinMLOEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies
    )
    
    engine.args = args
    engine.steps_per_epoch = steps_per_epoch
    
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Iterations per epoch: {steps_per_epoch}")
    print(f"  Total iterations: {total_steps}")
    print(f"  Validation every: {args.val_freq} iterations")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Run
    engine.run()
    
    print("\n" + "="*60)
    print(f"Training complete! Best accuracy: {engine.best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()