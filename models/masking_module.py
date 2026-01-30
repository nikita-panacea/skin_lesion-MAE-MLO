# models/masking_module.py - COMPLETE FIX
"""
UNet-based Masking Module for MLO-MAE

CRITICAL FIXES:
1. All frozen UNet computations in no_grad context
2. Detach ALL outputs (mask, ids_restore, mask_prob for frozen parts)
3. Only learnable refinement delta remains in graph
4. Explicit gradient isolation for Betty framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class UNetMaskingModule(nn.Module):
    """
    Masking module combining frozen UNet + learned refinement
    
    Design:
    - UNet (frozen): Provides spatial importance baseline
    - Refinement (learnable): Task-specific adaptation
    - Output: Masking probabilities with proper gradient flow
    
    Betty MLO compatibility:
    - UNet outputs completely isolated from graph (no_grad + detach)
    - Only refinement delta participates in hypergradient computation
    """
    
    def __init__(self, pretrained_unet_path, num_patches=196, embed_dim=768, 
                 learnable=True, use_unet=True):
        """
        Args:
            pretrained_unet_path: Path to UNet checkpoint
            num_patches: Number of patches (196 for 224x224/16)
            embed_dim: Embedding dimension (768 for ViT-Base)
            learnable: Enable learned refinement
            use_unet: Use UNet importance (if False, only learned)
        """
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.learnable = learnable
        self.use_unet = use_unet
        
        # Load frozen UNet
        if use_unet:
            from unet import UNet
            self.unet = UNet(n_channels=3, n_classes=1, bilinear=False)
            
            if Path(pretrained_unet_path).exists():
                state_dict = torch.load(pretrained_unet_path, map_location='cpu')
                self.unet.load_state_dict(state_dict)
                print(f" Loaded pretrained UNet from {pretrained_unet_path}")
            else:
                print(f"  Warning: UNet checkpoint not found at {pretrained_unet_path}")
            
            # CRITICAL: Freeze and set to eval
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()
        
        # Learned refinement (always trainable)
        if learnable:
            self.refinement = nn.Sequential(
                nn.Linear(num_patches * embed_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_patches),
                nn.Sigmoid()  # Output in [0, 1]
            )
            # Initialize
            for m in self.refinement.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _compute_unet_importance(self, images):
        """
        Compute importance map from frozen UNet
        
        CRITICAL: Entire computation in no_grad + detach result
        """
        with torch.no_grad():
            # Forward through frozen UNet
            importance_map = self.unet(images)  # [B, 1, H, W]
            importance_map = torch.sigmoid(importance_map)
            
            # Resize to patch grid
            patch_size = int(self.num_patches ** 0.5)
            importance_map = F.interpolate(
                importance_map,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            )
            
            # Flatten: [B, N]
            importance = importance_map.flatten(1)
        
        # CRITICAL: Detach to completely isolate from graph
        return importance.detach()
    
    def forward(self, images, patch_embeddings, mask_ratio=0.75, random=False):
        """
        Generate masking probabilities and apply masking
        
        Args:
            images: [B, 3, H, W] - For UNet
            patch_embeddings: [B, N, D] - For refinement
            mask_ratio: Masking ratio
            random: Use random baseline
        
        Returns:
            x_masked: [B, N_keep, D] - Unmasked patches
            mask: [B, N] - Binary mask (0=keep, 1=remove) [DETACHED]
            ids_restore: [B, N] - Restore indices [DETACHED]
            mask_prob: [B, N] - Masking probabilities [keeps gradients for learnable part]
        """
        B, N, D = patch_embeddings.shape
        len_keep = int(N * (1 - mask_ratio))
        
        if random:
            # Random baseline
            mask_prob = torch.rand(B, N, device=patch_embeddings.device)
        else:
            # Start with zeros (will accumulate contributions)
            mask_prob = torch.zeros(B, N, device=patch_embeddings.device)
            
            # Add UNet importance (frozen, detached)
            if self.use_unet:
                unet_importance = self._compute_unet_importance(images)
                mask_prob = mask_prob + 0.5 * unet_importance
            
            # Add learned refinement (trainable, keeps gradients)
            if self.learnable:
                x_flat = patch_embeddings.flatten(1)  # [B, N*D]
                refinement_delta = self.refinement(x_flat)  # [B, N]
                
                # CRITICAL: This is the ONLY part that should have gradients
                mask_prob = mask_prob + 0.5 * refinement_delta
            
            # Normalize to [0, 1] if needed
            if not self.learnable and not self.use_unet:
                mask_prob = torch.rand(B, N, device=patch_embeddings.device)
        
        # Sort by probability (higher = more likely to mask)
        ids_shuffle = torch.argsort(mask_prob, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep lowest probability patches (least important = keep visible)
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
        
        # CRITICAL FIX: Return detached mask and ids_restore
        # These should NOT participate in hypergradient computation
        # Only mask_prob (which contains learnable refinement) keeps gradients
        return (
            x_masked,
            mask.detach(),
            ids_restore.detach(),
            mask_prob  # Keep gradients for learnable part
        )


if __name__ == '__main__':
    """Test masking module with gradient flow verification"""
    import sys
    sys.path.append('.')
    
    print("Testing UNet Masking Module (Complete Fix)")
    print("="*60)
    
    # Create module
    module = UNetMaskingModule(
        pretrained_unet_path='unet/unet.pkl',
        num_patches=196,
        embed_dim=768,
        learnable=True,
        use_unet=True
    )
    
    # Count parameters
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total - trainable:,}")
    
    # Test forward
    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    patch_emb = torch.randn(2, 196, 768, requires_grad=True)
    
    x_masked, mask, ids_restore, mask_prob = module(images, patch_emb, mask_ratio=0.75)
    
    print(f"\nForward pass:")
    print(f"  x_masked: {x_masked.shape}")
    print(f"  mask: {mask.shape}, requires_grad={mask.requires_grad}")
    print(f"  ids_restore: {ids_restore.shape}, requires_grad={ids_restore.requires_grad}")
    print(f"  mask_prob: {mask_prob.shape}, requires_grad={mask_prob.requires_grad}")
    
    # Test gradient flow
    print(f"\nGradient flow test:")
    loss = mask_prob.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in module.refinement.parameters())
    no_grad = all(p.grad is None for p in module.unet.parameters())
    
    print(f"  Refinement has gradients: {has_grad}")
    print(f"  UNet has NO gradients: {no_grad}")
    print(f"  Input images.grad: {images.grad is not None}")
    print(f"  Input patch_emb.grad: {patch_emb.grad is not None}")
    
    if has_grad and no_grad:
        print("\n All tests passed!")
    else:
        print("\n Gradient flow incorrect!")