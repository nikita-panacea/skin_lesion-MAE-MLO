# train_mlo_skin.py
"""
MLO-MAE Training for Skin Lesion Classification

CRITICAL FIXES:

2. Module access: self.module.X() for methods
3. MAE forward signature: forward(x_masked, mask, ids_restore) - NO mask_prob
4. mask_prob used ONLY in loss computation (for gradient flow)
5. Proper ISIC dataset loading from CSV
"""
import os
import argparse
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from betty.engine import Engine
from betty.problems import ImplicitProblem  
from betty.configs import Config, EngineConfig

# Import dataset
from datasets.isic_dataset import ISICDataset, ISIC_CLASSES

# Import models
import sys
sys.path.append('.')
from models.mae_hybrid import MAEHybrid
from models.masking_module import UNetMaskingModule
from models.hybrid_model import HybridConvNeXtV2


def build_transforms(train=True, input_size=224, deterministic=False):
    """Build transforms"""
    import torchvision.transforms as transforms
    if train and not deterministic:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def compute_class_weights_from_csv(csv_path, class_names, max_weight=10.0):
    """Compute class weights from CSV"""
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(csv_path)
    
    # Get labels
    if 'label' in df.columns:
        labels = df['label'].astype(int).values
    else:
        label_cols = [col for col in class_names if col in df.columns]
        onehot = df[label_cols].fillna(0).astype(int).values
        labels = np.argmax(onehot, axis=1)
        label_mapping = {i: class_names.index(col) for i, col in enumerate(label_cols)}
        labels = np.array([label_mapping[int(lbl)] for lbl in labels])
    
    # Compute inverse frequency
    counts = np.bincount(labels, minlength=len(class_names))
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(class_names)
    weights = np.minimum(weights, max_weight)
    
    return torch.tensor(weights, dtype=torch.float32)


class MAEProblem(ImplicitProblem):
    """
    MAE reconstruction problem (Level 1 - Lowest)
    - Methods: self.module.X() (single .module)
    - mask_prob used ONLY in loss, NOT in MAE forward
    """
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        
        # Embed patches (following reference line 306-307)
        x = self.module.patch_embed(images)
        x = x + self.module.pos_embed[:, 1:, :].detach()  # Add positional embeddings
        
        # Generate masks (following reference line 309)
        x_masked, mask, ids_restore, mask_prob = self.masking.module.forward(
            images,  # For UNet importance
            x,       # For learned refinement
            mask_ratio=self.mask_ratio,
            random=self.random_mask
        )
        
        # Forward through MAE (following reference line 311)
        # CRITICAL: Do NOT pass mask_prob to MAE forward! Only 3 arguments
        pred = self.module.forward(x_masked, mask, ids_restore)  # ← Only 3 args
        
        # Compute target (following reference line 312-316)
        # CRITICAL: Use .module.module for patchify attribute
        target = self.module.patchify(images)
        if self.module.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        
        # Compute loss (following reference line 318-325)
        # CRITICAL: Inline loss computation, mask_prob used HERE
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        
        # Weight loss by masking probability (following reference line 322-323)
        # CRITICAL: Gradients flow through mask_prob HERE, not in MAE forward
        if not self.random_mask:
            loss = loss * mask_prob  # ← THIS is where gradients flow
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        if self.is_rank_zero():
            wandb.log({'mae/loss': loss.item()})
        
        return loss


class ClassifierProblem(ImplicitProblem):
    """
    Classification problem (Level 2 - Middle)
    
    Trains classifier on real images
    """
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Classify using hybrid classifier (takes raw images)
        logits = self.module.forward(images)
        
        # Classification loss
        loss = F.cross_entropy(logits, labels)
        
        if self.is_rank_zero():
            wandb.log({'classifier/loss': loss.item()})
        
        return loss


class MaskingProblem(ImplicitProblem):
    """
    Masking problem (Level 3 - Highest)
    
    Optimizes masking strategy based on validation classification performance
    """
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Classify using hybrid classifier
        logits = self.classifier.module.forward(images)
        
        # Validation loss
        loss = F.cross_entropy(logits, labels)
        
        if self.is_rank_zero():
            wandb.log({'masking/val_loss': loss.item()})
        
        return loss


class MLOEngine(Engine):
    """Custom engine with validation"""
    
    @torch.no_grad()
    def validation(self):
        """Validate on validation set"""
        if not hasattr(self, 'val_loader'):
            return
        
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward through classifier
            logits = self.classifier.module.forward(images)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        acc = 100.0 * correct / total
        
        if self.is_rank_zero():
            print(f"Validation accuracy: {acc:.2f}%")
            wandb.log({'val/accuracy': acc})
        
        # Save checkpoint if best
        if hasattr(self, 'best_acc'):
            if acc > self.best_acc:
                self.best_acc = acc
                if hasattr(self, 'save_checkpoint_fn'):
                    self.save_checkpoint_fn('best_model.pt', {'accuracy': acc})
        else:
            self.best_acc = acc


def main():
    parser = argparse.ArgumentParser(description='MLO-MAE Skin Lesion Classifier')
    
    # Data
    parser.add_argument('--train_csv', required=True, help='Training CSV path')
    parser.add_argument('--val_csv', required=True, help='Validation CSV path')
    parser.add_argument('--test_csv', default='', help='Test CSV path')
    parser.add_argument('--img_dir', required=True, help='Image directory')
    parser.add_argument('--unet_checkpoint', required=True, help='Pretrained UNet path')
    
    # Model
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--norm_pix_loss', action='store_true', default=True)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Optimization
    parser.add_argument('--lr_mae', type=float, default=1.5e-4)
    parser.add_argument('--lr_cls', type=float, default=5e-4)
    parser.add_argument('--lr_mask', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # MLO settings (following reference)
    parser.add_argument('--unroll_steps_mae', type=int, default=2)
    parser.add_argument('--unroll_steps_cls', type=int, default=1)
    parser.add_argument('--unroll_steps_mask', type=int, default=1)
    parser.add_argument('--valid_step', type=int, default=100)
    
    # Options
    parser.add_argument('--baseline', action='store_true', help='Random masking baseline')
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # Wandb
    parser.add_argument('--wandb_project', default='mlo-mae-skin')
    parser.add_argument('--wandb_name', default='mlo-training')
    parser.add_argument('--wandb_mode', default='online')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args)
    )
    
    print("\n" + "="*60)
    print("MLO-MAE Skin Lesion Classifier Training")
    print("="*60)
    
    # Datasets
    print("\nLoading datasets...")
    train_transform = build_transforms(train=True, input_size=args.input_size)
    val_transform = build_transforms(train=False, input_size=args.input_size)
    
    train_dataset = ISICDataset(args.train_csv, args.img_dir, train_transform)
    val_dataset = ISICDataset(args.val_csv, args.img_dir, val_transform)
    
    print(f"\nDataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty! Check CSV path: {args.train_csv}")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty! Check CSV path: {args.val_csv}")
    
    # Dataloaders
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
        shuffle=True,  # Shuffle for MLO training
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Models
    print("\nBuilding models...")
    
    # MAE model with hybrid encoder
    mae_model = MAEHybrid(
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=3,
        encoder_dim=768,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_pix_loss=args.norm_pix_loss
    ).to(device)
    
    # Masking module
    num_patches = (args.input_size // args.patch_size) ** 2
    masking_module = UNetMaskingModule(
        pretrained_unet_path=args.unet_checkpoint,
        num_patches=num_patches,
        embed_dim=768,
        learnable=not args.baseline
    ).to(device)
    
    # Classifier (Hybrid ConvNeXtV2)
    classifier = HybridConvNeXtV2(
        num_classes=len(ISIC_CLASSES),
        pretrained=True
    ).to(device)
    
    print(f"\nModels:")
    print(f"  MAE parameters: {sum(p.numel() for p in mae_model.parameters())/1e6:.2f}M")
    print(f"  Masking parameters: {sum(p.numel() for p in masking_module.parameters())/1e6:.2f}M")
    print(f"  Classifier parameters: {sum(p.numel() for p in classifier.parameters())/1e6:.2f}M")
    
    # Class weights
    class_weights = compute_class_weights_from_csv(
        args.train_csv,
        class_names=ISIC_CLASSES,
        max_weight=10.0
    ).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    # Optimizers
    optimizer_mae = torch.optim.AdamW(
        mae_model.parameters(),
        lr=args.lr_mae,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    optimizer_cls = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr_cls,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    optimizer_mask = torch.optim.AdamW(
        masking_module.parameters(),
        lr=args.lr_mask,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Calculate total iterations
    num_train_batches = len(train_loader)
    total_iters = args.epochs * num_train_batches * args.unroll_steps_mae * \
                  args.unroll_steps_cls * args.unroll_steps_mask
    
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Iterations per epoch: {num_train_batches}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Validation every: {args.valid_step} iterations")
    
    # Schedulers
    scheduler_mae = CosineAnnealingLR(optimizer_mae, T_max=total_iters)
    scheduler_cls = CosineAnnealingLR(optimizer_cls, T_max=total_iters)
    scheduler_mask = CosineAnnealingLR(optimizer_mask, T_max=total_iters)
    
    # Betty configs
    mae_config = Config(
        type="darts",
        # retain_graph=True,
        log_step=10,
        unroll_steps=args.unroll_steps_mae,
        allow_unused=True
    )
    
    cls_config = Config(
        type="darts",
        # retain_graph=True,
        log_step=10,
        unroll_steps=args.unroll_steps_cls,
        allow_unused=True
    )
    
    mask_config = Config(
        type="darts",
        # retain_graph=True,
        log_step=10,
        unroll_steps=args.unroll_steps_mask,
        allow_unused=True
    )
    
    engine_config = EngineConfig(
        valid_step=args.valid_step,
        train_iters=total_iters,
        roll_back=True
    )
    
    # Create problems
    mae_problem = MAEProblem(
        name='mae',
        module=mae_model,
        optimizer=optimizer_mae,
        scheduler=scheduler_mae,
        train_data_loader=train_loader,
        config=mae_config,
    )
    mae_problem.mask_ratio = args.mask_ratio
    mae_problem.random_mask = args.baseline
    mae_problem.device = device
    
    cls_problem = ClassifierProblem(
        name='classifier',
        module=classifier,
        optimizer=optimizer_cls,
        scheduler=scheduler_cls,
        train_data_loader=train_loader,
        config=cls_config,
    )
    cls_problem.device = device
    
    mask_problem = MaskingProblem(
        name='masking',
        module=masking_module,
        optimizer=optimizer_mask,
        scheduler=scheduler_mask,
        train_data_loader=val_loader,
        config=mask_config,
    )
    mask_problem.device = device
    
    # Dependencies
    if args.baseline:
        problems = [mae_problem]
        dependencies = {'l2u': {}, 'u2l': {}}
    else:
        problems = [mae_problem, cls_problem, mask_problem]
        dependencies = {
            'l2u': {
                mae_problem: [cls_problem, mask_problem],
                cls_problem: [mask_problem]
            },
            'u2l': {
                mask_problem: [mae_problem]
            }
        }
    
    # Create engine
    engine = MLOEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies
    )
    
    # Add attributes
    engine.val_loader = val_eval_loader
    engine.mae = mae_problem
    engine.classifier = cls_problem
    engine.device = device
    
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
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()