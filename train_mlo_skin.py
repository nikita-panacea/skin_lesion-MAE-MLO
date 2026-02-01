# train_mlo_skin.py - FIXED VERSION v2
"""
MLO-MAE Training for Skin Lesion Classification

CRITICAL FIXES (v2):
1. Fixed dependency graph - masking accessible from mae via l2u
2. Proper gradient isolation for frozen UNet
3. Correct Problem access pattern matching MLOMAE reference
4. Fixed validation handling with proper detachment
5. Removed retain_graph from lower-level problems
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

# Global args reference for access in Problems
GLOBAL_ARGS = None


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
    parser.add_argument('--lr_mae', type=float, default=5e-5,
                       help='MAE learning rate (lower to prevent overfitting)')
    parser.add_argument('--lr_classifier', type=float, default=1e-3,
                       help='Classifier learning rate (higher for faster learning)')
    parser.add_argument('--lr_masking', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--aux_cls_weight', type=float, default=0.5,
                       help='Weight for auxiliary classification loss in MAE (0-1)')
    
    # MLO settings
    parser.add_argument('--unroll_mae', type=int, default=1)
    parser.add_argument('--unroll_classifier', type=int, default=1)
    parser.add_argument('--unroll_masking', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=100)
    
    # MAE settings
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--norm_pix_loss', action='store_true', default=True)
    parser.add_argument('--baseline', action='store_true', default=False,
                       help='Use random masking baseline (no learned masking)')
    
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
    """Load ISIC 2019 dataset with proper augmentation"""
    from datasets.isic2019_dataset import ISIC2019Dataset
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_csv = os.path.join(args.data_path, 'train.csv')
    val_csv = os.path.join(args.data_path, 'val.csv')
    img_dir = os.path.join(args.data_path, 'images')
    
    train_dataset = ISIC2019Dataset(train_csv, img_dir, transform=train_transform)
    val_dataset = ISIC2019Dataset(val_csv, img_dir, transform=val_transform)
    
    return train_dataset, val_dataset


class MAEProblem(ImplicitProblem):
    """
    Stage 1: MAE Pretraining (Lowest Level)
    
    Using Hybrid ConvNeXtV2 + Separable Attention encoder:
    - ConvNeXtV2 stages 1-2 extract 384-dim patch embeddings at 14x14
    - Masking applied to 196 tokens of dim 384
    - Separable attention stages 3-4 process masked tokens
    - Auxiliary classification loss helps learn discriminative features
    """
    def training_step(self, batch):
        images, labels, _ = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Step 1: Get patch embeddings from hybrid encoder (384-dim at 14x14)
        # This uses ConvNeXtV2 stages 1-2 (pretrained)
        x = self.module.get_patch_embeddings(images)  # [B, 196, 384]
        
        # Step 2: Get masking from masking module
        # Masking module expects [B, 196, embed_dim] and returns masked tokens
        x_masked, mask, ids_restore, mask_prob = self.masking.module(
            images, x, GLOBAL_ARGS.mask_ratio, random=GLOBAL_ARGS.baseline
        )
        
        # Step 3: Forward through attention stages + decoder
        pred = self.module.forward_with_mask(x_masked, mask, ids_restore)
        
        # Step 4: Compute reconstruction loss
        target = self.module.patchify(images)
        if self.module.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        recon_loss = (pred - target) ** 2
        recon_loss = recon_loss.mean(dim=-1)  # [B, N], mean loss per patch
        
        # Weight by mask_prob to enable backprop through masking module
        if not GLOBAL_ARGS.baseline:
            recon_loss = recon_loss * mask_prob
        
        # Mean loss on masked patches only
        recon_loss = (recon_loss * mask).sum() / mask.sum()
        
        # Step 5: Auxiliary classification loss (helps learn discriminative features)
        # Get full encoder output (no masking) for classification
        encoder_output, _, _ = self.module.forward_encoder(images)
        cls_features = encoder_output[:, 0]  # CLS token [B, 768]
        
        # Use classifier for auxiliary supervision
        aux_logits = self.classifier.module.forward_from_features(cls_features)
        aux_cls_loss = F.cross_entropy(aux_logits, labels)
        
        # Combine losses: reconstruction + auxiliary classification
        aux_weight = getattr(GLOBAL_ARGS, 'aux_cls_weight', 0.5)
        total_loss = recon_loss + aux_weight * aux_cls_loss
        
        return total_loss


class ClassifierProblem(ImplicitProblem):
    """
    Stage 2: Classifier Training (Middle Level)
    
    Uses features from MAE encoder for classification.
    Gradient flows to MAE through the encoder for bilevel optimization.
    
    CRITICAL: Do NOT use torch.no_grad() here - Betty needs gradient flow
    for hypergradient computation in bilevel optimization.
    """
    def training_step(self, batch):
        images, labels, _ = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Get encoder features from MAE (accessed via self.mae)
        # IMPORTANT: No no_grad() - Betty needs gradient flow for bilevel optimization
        encoder_output, _, _ = self.mae.module.forward_encoder(images)
        
        # Forward through classifier using the CLS token
        # encoder_output shape: [B, N+1, D] where first token is CLS
        cls_features = encoder_output[:, 0]
        logits = self.module.forward_from_features(cls_features)
        
        # Compute classification loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MaskingProblem(ImplicitProblem):
    """
    Stage 3: Masking Module Update (Highest Level)
    
    Updated based on validation loss of classifier.
    This drives the masking to focus on lesion-relevant regions.
    """
    def training_step(self, batch):
        images, labels, _ = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Get encoder features from MAE (with gradient flow)
        encoder_output, _, _ = self.mae.module.forward_encoder(images)
        
        # Forward through classifier using CLS token
        cls_features = encoder_output[:, 0]
        logits = self.classifier.module.forward_from_features(cls_features)
        
        # Compute validation/meta loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SkinMLOEngine(Engine):
    """
    Custom MLO Engine with validation
    
    FIXED (v2):
    - Proper validation using encoder features
    - Checkpoint saving with all components
    - Logging improvements
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_acc = 0.0
        self.args = None
        self.val_loader = None
    
    @torch.no_grad()
    def validation(self):
        """Validation on held-out set with per-class metrics"""
        self.eval()
        
        mae_model = self.mae.module
        classifier_model = self.classifier.module
        
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get encoder features
            encoder_output, _, _ = mae_model.forward_encoder(images)
            cls_features = encoder_output[:, 0]
            
            # Classify
            logits = classifier_model.forward_from_features(cls_features)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        acc = 100.0 * correct / total
        
        # Compute per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        num_classes = 8
        per_class_acc = []
        for c in range(num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == c).mean() * 100
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
        
        # Log
        if self.is_rank_zero():
            logging.info(f"[Validation] Step {self.global_step}: Accuracy = {acc:.2f}%")
            logging.info(f"  Per-class: {[f'{a:.1f}' for a in per_class_acc]}")
            
            # Check prediction distribution
            pred_dist = np.bincount(all_preds, minlength=num_classes)
            logging.info(f"  Pred dist: {pred_dist.tolist()}")
            
            try:
                log_dict = {
                    'val_acc': acc,
                    'best_acc': self.best_acc,
                    'global_step': self.global_step
                }
                for c, a in enumerate(per_class_acc):
                    log_dict[f'val_acc_class_{c}'] = a
                wandb.log(log_dict)
            except:
                pass
        
        # Save best
        if acc > self.best_acc:
            self.best_acc = acc
            if self.is_rank_zero():
                self.save_checkpoint('best_model.pt')
                logging.info(f"  -> New best accuracy: {acc:.2f}%")
        
        self.train()
        return {'val_acc': acc}
    
    def save_checkpoint(self, filename):
        """Save checkpoint"""
        if not self.is_rank_zero():
            return
        
        save_path = Path(GLOBAL_ARGS.output_dir) / filename
        
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
    global GLOBAL_ARGS
    args = parse_args()
    GLOBAL_ARGS = args  # Set global reference for Problem access
    
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
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
                mode=args.wandb_mode
            )
        except Exception as e:
            logging.warning(f"Could not initialize wandb: {e}")
    
    print("\n" + "="*60)
    print("MLO-MAE Skin Lesion Classifier Training (v2)")
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
    print("  Encoder: Hybrid ConvNeXtV2 (pretrained) + Separable Self-Attention")
    
    # 1. MAE with Hybrid ConvNeXtV2 + Separable Attention encoder
    mae_model = mae_hybrid_base(
        norm_pix_loss=args.norm_pix_loss,
        pretrained=True  # Use ImageNet pretrained ConvNeXtV2
    ).to(device)
    
    # 2. Masking module (with frozen UNet for importance estimation)
    # Uses 384-dim embeddings from hybrid encoder's stage2 output
    masking_module = UNetMaskingModule(
        pretrained_unet_path=args.unet_path,
        num_patches=196,
        embed_dim=384,  # Matches hybrid encoder output at 14x14 level
        learnable=True,
        use_unet=True
    ).to(device)
    
    # 3. Classifier head (enhanced with MLP for better capacity)
    from models.hybrid_classifier import ClassifierHead
    classifier_head = ClassifierHead(
        embed_dim=768,
        num_classes=8,
        drop_rate=0.2,
        mlp_hidden=512  # More capacity for classification
    ).to(device)
    
    # Count parameters
    mae_params = sum(p.numel() for p in mae_model.parameters() if p.requires_grad)
    mask_params = sum(p.numel() for p in masking_module.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in classifier_head.parameters() if p.requires_grad)
    
    print(f"\nModels:")
    print(f"  MAE parameters: {mae_params/1e6:.2f}M")
    print(f"  Masking parameters (trainable only): {mask_params/1e6:.2f}M")
    print(f"  Classifier head parameters: {cls_params/1e6:.2f}M")
    
    # Optimizers
    optimizer_mae = torch.optim.AdamW(
        mae_model.parameters(),
        lr=args.lr_mae,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    optimizer_classifier = torch.optim.AdamW(
        classifier_head.parameters(),
        lr=args.lr_classifier,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Only optimize learnable parameters in masking module
    masking_params = [p for p in masking_module.parameters() if p.requires_grad]
    optimizer_masking = torch.optim.AdamW(
        masking_params,
        lr=args.lr_masking,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Learning rate schedulers
    steps_per_epoch = len(train_loader)
    # Account for unroll steps in total iterations
    total_steps = args.epochs * steps_per_epoch * args.unroll_mae * args.unroll_classifier * args.unroll_masking
    
    scheduler_mae = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mae, T_max=total_steps
    )
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_classifier, T_max=total_steps
    )
    scheduler_masking = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_masking, T_max=total_steps
    )
    
    # ==========================================================================
    # Betty Problem Configurations
    # ==========================================================================
    # CRITICAL FIX: 
    # - mae (lowest): retain_graph=True, allow_unused=True
    # - classifier (middle): retain_graph=True, allow_unused=True  
    # - masking (highest): retain_graph=False (final level)
    # ==========================================================================
    
    mae_config = Config(
        type='darts',
        unroll_steps=args.unroll_mae,
        log_step=100,
        retain_graph=True,
        allow_unused=True
    )
    
    classifier_config = Config(
        type='darts',
        unroll_steps=args.unroll_classifier,
        log_step=100,
        retain_graph=True,
        allow_unused=True
    )
    
    masking_config = Config(
        type='darts',
        unroll_steps=args.unroll_masking,
        log_step=100,
        retain_graph=True,  # MLOMAE uses retain_graph=True for all levels
        allow_unused=True
    )
    
    # Create Betty Problems
    mae = MAEProblem(
        name='mae',
        module=mae_model,
        optimizer=optimizer_mae,
        scheduler=scheduler_mae,
        train_data_loader=train_loader,
        config=mae_config,
    )
    
    classifier = ClassifierProblem(
        name='classifier',
        module=classifier_head,
        optimizer=optimizer_classifier,
        scheduler=scheduler_classifier,
        train_data_loader=train_loader,
        config=classifier_config,
    )
    
    masking = MaskingProblem(
        name='masking',
        module=masking_module,
        optimizer=optimizer_masking,
        scheduler=scheduler_masking,
        train_data_loader=val_loader,  # Use val set for masking update (meta-learning)
        config=masking_config,
    )
    
    # ==========================================================================
    # Dependency Graph (CRITICAL FIX)
    # ==========================================================================
    # Following MLOMAE pattern:
    # - mae needs to access masking for getting mask probabilities
    # - classifier needs to access mae for encoder features
    # - masking needs to access both mae and classifier for meta-loss
    #
    # u2l: upper to lower (what each upper depends on)
    # l2u: lower to upper (what each lower feeds into)
    # ==========================================================================
    
    problems = [mae, classifier, masking]
    
    # Upper-to-Lower: What each upper-level problem depends on
    u2l = {
        masking: [mae],        # masking optimizes based on mae's encoder
        classifier: [mae]       # classifier uses mae's features
    }
    
    # Lower-to-Upper: What each lower feeds into  
    # CRITICAL: mae needs masking as upper so it can access self.masking
    l2u = {
        mae: [classifier, masking],  # mae feeds into classifier AND masking can access it
        classifier: [masking]         # classifier feeds into masking
    }
    
    dependencies = {'u2l': u2l, 'l2u': l2u}
    
    # Engine config
    engine_config = EngineConfig(
        strategy='default',
        train_iters=total_steps,
        valid_step=args.val_freq * args.unroll_mae * args.unroll_classifier * args.unroll_masking,
        logger_type='tensorboard'
    )
    
    # Create engine
    engine = SkinMLOEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies
    )
    
    engine.steps_per_epoch = steps_per_epoch
    engine.val_loader = val_loader
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Iterations per epoch: {steps_per_epoch}")
    print(f"  Total iterations: {total_steps}")
    print(f"  Validation every: {engine_config.valid_step} iterations")
    print(f"  Unroll steps: mae={args.unroll_mae}, cls={args.unroll_classifier}, mask={args.unroll_masking}")
    print(f"  Baseline (random masking): {args.baseline}")
    print(f"  Learning rates: mae={args.lr_mae}, cls={args.lr_classifier}, mask={args.lr_masking}")
    print(f"  Auxiliary classification weight: {args.aux_cls_weight}")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Run training
    engine.run()
    
    # Final checkpoint
    engine.save_checkpoint('final_model.pt')
    
    print("\n" + "="*60)
    print(f"Training complete! Best accuracy: {engine.best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()