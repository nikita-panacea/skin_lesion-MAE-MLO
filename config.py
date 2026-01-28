# config.py
"""
Configuration for MAE-MLO training on skin lesions
"""
import argparse


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MAE-MLO with Hybrid Encoder for Skin Lesion Classification'
    )
    
    # Data
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, required=True,
                       help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV (optional)')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--pretrained_unet', type=str, default=None,
                       help='Path to pretrained UNet for mask initialization')
    
    # Model
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for MAE')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Masking ratio for MAE')
    parser.add_argument('--encoder_dim', type=int, default=768,
                       help='Encoder dimension')
    parser.add_argument('--decoder_dim', type=int, default=512,
                       help='Decoder dimension')
    parser.add_argument('--decoder_depth', type=int, default=4,
                       help='Number of decoder blocks')
    parser.add_argument('--num_classes', type=int, default=8,
                       help='Number of classes')
    
    # Training - MAE
    parser.add_argument('--mae_lr', type=float, default=1.5e-4,
                       help='Learning rate for MAE')
    parser.add_argument('--mae_weight_decay', type=float, default=0.05,
                       help='Weight decay for MAE')
    
    # Training - Classifier
    parser.add_argument('--cls_lr', type=float, default=5e-4,
                       help='Learning rate for classifier')
    parser.add_argument('--cls_weight_decay', type=float, default=0.05,
                       help='Weight decay for classifier')
    
    # Training - Masking
    parser.add_argument('--mask_lr', type=float, default=5e-5,
                       help='Learning rate for masking module')
    parser.add_argument('--mask_weight_decay', type=float, default=0.05,
                       help='Weight decay for masking module')
    
    # Training - General
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # MLO
    parser.add_argument('--unroll_steps_mae', type=int, default=2,
                       help='Unrolling steps for MAE')
    parser.add_argument('--unroll_steps_cls', type=int, default=1,
                       help='Unrolling steps for classifier')
    parser.add_argument('--unroll_steps_mask', type=int, default=1,
                       help='Unrolling steps for masking')
    parser.add_argument('--valid_step', type=int, default=100,
                       help='Validation frequency (in iterations)')
    parser.add_argument('--loss_lambda', type=float, default=1.0,
                       help='Weight for reconstruction loss from fake images')
    
    # Augmentation
    parser.add_argument('--use_mixup', action='store_true',
                       help='Use mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                       help='Mixup alpha parameter')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='Log frequency (in iterations)')
    
    # Experiment
    parser.add_argument('--exp_name', type=str, default='mlo_skin',
                       help='Experiment name')
    parser.add_argument('--wandb_project', type=str, default='MAE-MLO-Skin',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_mode', type=str, default='disabled',
                       choices=['online', 'offline', 'disabled'],
                       help='Weights & Biases mode')
    
    # Baseline
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline (random masking, no MLO)')
    parser.add_argument('--random_mask', action='store_true',
                       help='Use random masking instead of UNet-guided')
    
    # Resume
    parser.add_argument('--resume', type=str, default='',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    return args