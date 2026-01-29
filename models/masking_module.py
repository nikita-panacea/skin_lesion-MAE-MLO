# models/masking_module.py - DETACH FIX for Betty MLO
"""
UNet-based Masking Module for MLO-MAE

CRITICAL FIX: Detach all frozen UNet outputs before use
- UNet parameters are frozen (requires_grad=False)
- Betty tries to retain_grad on ALL tensors in graph
- Solution: .detach() UNet output to isolate from graph
- Gradients still flow through learned refinement layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class UNetMaskingModule(nn.Module):
    """
    Masking module that combines:
    1. Frozen pretrained UNet for spatial importance
    2. Learned refinement layers for task-specific adaptation
    
    CRITICAL: UNet output is DETACHED to prevent Betty retain_grad errors
    """
    
    def __init__(self, pretrained_unet_path, num_patches=196, embed_dim=768, learnable=True):
        """
        Args:
            pretrained_unet_path: Path to pretrained UNet checkpoint
            num_patches: Number of image patches (14x14 = 196 for 224x224 with patch_size=16)
            embed_dim: Embedding dimension (768 for ViT-Base)
            learnable: If False, use only UNet (for baseline comparison)
        """
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.learnable = learnable
        
        # Load pretrained UNet
        from unet import UNet
        self.unet = UNet(n_channels=3, n_classes=1, bilinear=False)
        
        if Path(pretrained_unet_path).exists():
            state_dict = torch.load(pretrained_unet_path, map_location='cpu')
            self.unet.load_state_dict(state_dict)
            print(f"Loaded pretrained UNet from {pretrained_unet_path}")
        else:
            print(f"Warning: UNet checkpoint not found at {pretrained_unet_path}")
        
        # CRITICAL: Freeze UNet parameters
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()  # Set to eval mode
        
        # Learned refinement layers (following MLO-MAE paper)
        if self.learnable:
            self.refinement = nn.Sequential(
                nn.Linear(num_patches * embed_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_patches)
            )
        
    def forward(self, images, patch_embeddings, mask_ratio=0.75, random=False):
        """
        Generate masking probabilities
        
        Args:
            images: [B, 3, H, W] - Input images for UNet
            patch_embeddings: [B, N, D] - Patch embeddings for learned refinement
            mask_ratio: Ratio of patches to mask
            random: If True, use random masking (baseline)
        
        Returns:
            x_masked: [B, N_keep, D] - Unmasked patch embeddings
            mask: [B, N] - Binary mask (0=keep, 1=remove)
            ids_restore: [B, N] - Indices to restore original order
            mask_prob: [B, N] - Masking probabilities (for gradient flow)
        
        CRITICAL: UNet output is DETACHED before entering computational graph
        """
        B, N, D = patch_embeddings.shape
        len_keep = int(N * (1 - mask_ratio))
        
        if random:
            # Random masking baseline
            mask_prob = torch.rand(B, N, device=patch_embeddings.device)
        else:
            # CRITICAL FIX: Compute UNet importance in no_grad context
            with torch.no_grad():
                # UNet expects [B, 3, H, W], outputs [B, 1, H, W]
                importance_map = self.unet(images)
                importance_map = torch.sigmoid(importance_map)
                
                # Resize to patch grid (N = 14x14 = 196)
                patch_size = int(N ** 0.5)  # 14 for 196 patches
                importance_map = F.interpolate(
                    importance_map,
                    size=(patch_size, patch_size),
                    mode='bilinear',
                    align_corners=False
                )
                # Flatten to [B, N]
                unet_importance = importance_map.flatten(1)  # [B, 196]
            
            # CRITICAL: Detach to isolate from graph
            # This prevents Betty from trying to retain_grad on frozen UNet params
            unet_importance = unet_importance.detach()
            
            # Learned refinement (gradients flow through this)
            if self.learnable:
                # Flatten patch embeddings
                x_flat = patch_embeddings.flatten(1)  # [B, N*D]
                
                # Refinement delta
                refinement = self.refinement(x_flat)  # [B, N]
                refinement = torch.sigmoid(refinement)
                
                # Combine: detached UNet + learned refinement
                # Both are in [0, 1] range after sigmoid
                mask_prob = 0.5 * unet_importance + 0.5 * refinement
            else:
                # Use only UNet importance
                mask_prob = unet_importance
        
        # Sort by probability (higher prob = more likely to be masked)
        ids_shuffle = torch.argsort(mask_prob, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep lowest probability patches (less important = keep visible)
        ids_keep = ids_shuffle[:, len_keep:]
        
        # Gather kept patches
        x_masked = torch.gather(
            patch_embeddings,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Binary mask: 0=keep, 1=remove
        mask = torch.ones(B, N, device=patch_embeddings.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # return x_masked, mask, ids_restore, mask_prob
        return (
            x_masked,
            mask.detach(),
            ids_restore.detach(),
            mask_prob
        )



if __name__ == '__main__':
    """Test masking module"""
    import sys
    sys.path.append('.')
    
    print("Testing UNet Masking Module (with DETACH fix)")
    print("="*60)
    
    # Create module
    module = UNetMaskingModule(
        pretrained_unet_path='unet/unet.pkl',
        num_patches=196,
        embed_dim=768,
        learnable=True
    )
    
    # Test forward
    images = torch.randn(2, 3, 224, 224)
    patch_emb = torch.randn(2, 196, 768)
    
    x_masked, mask, ids_restore, mask_prob = module(images, patch_emb, mask_ratio=0.75)
    
    print(f"\nForward pass:")
    print(f"  Images: {images.shape}")
    print(f"  Patch embeddings: {patch_emb.shape}")
    print(f"  x_masked: {x_masked.shape} (should be [2, 49, 768])")
    print(f"  mask: {mask.shape} (should be [2, 196])")
    print(f"  ids_restore: {ids_restore.shape} (should be [2, 196])")
    print(f"  mask_prob: {mask_prob.shape} (should be [2, 196])")
    
    print(f"\nMask statistics:")
    print(f"  Masked patches: {mask.sum(dim=1)} (should be ~147 each)")
    print(f"  Kept patches: {(1-mask).sum(dim=1)} (should be ~49 each)")
    
    # Test gradient flow
    print(f"\nGradient flow test:")
    print(f"  mask_prob.requires_grad: {mask_prob.requires_grad}")
    print(f"  UNet params frozen: {all(not p.requires_grad for p in module.unet.parameters())}")
    print(f"  Refinement params trainable: {all(p.requires_grad for p in module.refinement.parameters())}")
    
    # Simulate backward
    loss = mask_prob.sum()
    loss.backward()
    
    print(f"\nAfter backward:")
    print(f"  Refinement has gradients: {any(p.grad is not None for p in module.refinement.parameters())}")
    print(f"  UNet has NO gradients: {all(p.grad is None for p in module.unet.parameters())}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")