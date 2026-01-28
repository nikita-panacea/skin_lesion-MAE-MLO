# models/masking_module.py

"""
UNet-initialized Masking Module
Uses pretrained UNet segmentation masks to guide MAE masking

CRITICAL FIXES FOR GRADIENT FLOW:
1. Detach unet_importance before combining (it's from frozen UNet)
2. Detach mask_prob BEFORE sorting (torch.argsort breaks gradients)
3. Return ORIGINAL mask_prob (with gradients) for loss weighting
4. Gradients flow through mask_prob weighting in reconstruction loss, NOT through sorting

This follows MLO-MAE paper pattern (Eq. 1):
Loss = Σ_j σ(M_j) * L_rec(M_j)
Gradients flow through σ(M_j) = mask_prob, which contains learned_importance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetMaskingModule(nn.Module):
    """
    Masking module initialized with UNet segmentation predictions
    Learns to refine masks for optimal reconstruction
    """
    def __init__(self, pretrained_unet_path=None, num_patches=196, 
                 embed_dim=768, hidden_dim=512, learnable=True):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.learnable = learnable
        
        # Load pretrained UNet for mask initialization
        self.unet = None
        if pretrained_unet_path is not None:
            try:
                from unet.unet_model import UNet
                self.unet = UNet(n_channels=3, n_classes=1, bilinear=False)
                state_dict = torch.load(pretrained_unet_path, map_location='cpu')
                self.unet.load_state_dict(state_dict)
                self.unet.eval()
                # Freeze UNet
                for param in self.unet.parameters():
                    param.requires_grad = False
                print(f"Loaded pretrained UNet from {pretrained_unet_path}")
            except Exception as e:
                print(f"Warning: Could not load UNet: {e}")
                self.unet = None
        
        # Learnable refinement network
        if learnable:
            self.mask_refine = nn.Sequential(
                nn.Linear(num_patches * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_patches),
                nn.Sigmoid()
            )
        else:
            self.mask_refine = None
    
    def get_unet_mask(self, images):
        """
        Get segmentation mask from UNet
        Args:
            images: [B, 3, H, W] - input images
        Returns:
            masks: [B, 1, H, W] - segmentation masks
        """
        if self.unet is None:
            # Return uniform random mask if no UNet
            return torch.rand_like(images[:, :1, :, :])
        
        # Predict mask (frozen UNet, no gradients needed here)
        with torch.no_grad():
            masks = self.unet(images)
            masks = torch.sigmoid(masks)
        
        return masks
    
    def mask_to_patch_importance(self, seg_masks, patch_size=16):
        """
        Convert pixel-level segmentation masks to patch-level importance
        Args:
            seg_masks: [B, 1, H, W] - segmentation masks
            patch_size: int - patch size
        Returns:
            importance: [B, num_patches] - patch importance scores
        """
        B, _, H, W = seg_masks.shape
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Reshape to patches
        patches = seg_masks.reshape(
            B, 1, 
            num_patches_h, patch_size,
            num_patches_w, patch_size
        )
        
        # Average over patch pixels to get importance
        # Higher mask value = more important (lesion region)
        importance = patches.mean(dim=(1, 3, 5))  # [B, num_patches_h, num_patches_w]
        importance = importance.reshape(B, -1)  # [B, num_patches]
        
        # Invert: we want to MASK (remove) lesion regions to force reconstruction
        # This makes the model learn lesion features better
        importance = 1.0 - importance
        
        return importance
    
    def forward(self, images, patches, mask_ratio=0.75, random=False):
        """
        Generate masking indices based on UNet predictions
        Args:
            images: [B, 3, H, W] - original images
            patches: [B, N, D] - patch embeddings
            mask_ratio: float - ratio of patches to mask
            random: bool - use random masking (baseline)
        Returns:
            x_masked: [B, N*(1-mask_ratio), D] - unmasked patches
            mask: [B, N] - binary mask (0=keep, 1=remove)
            ids_restore: [B, N] - indices to restore order
            mask_prob: [B, N] - masking probabilities (WITH gradients for MLO!)
        """
        B, N, D = patches.shape
        len_keep = int(N * (1 - mask_ratio))
        
        if random or not self.learnable or self.mask_refine is None:
            # Random masking (baseline) or non-learnable mode
            # Use random noise - no gradients needed
            noise = torch.rand(B, N, device=patches.device)
            mask_prob = noise
        else:
            # Learnable masking mode
            # CRITICAL FIX: Proper gradient handling for MLO
            
            # Get UNet importance (no gradient - frozen UNet)
            seg_masks = self.get_unet_mask(images)
            unet_importance = self.mask_to_patch_importance(seg_masks)
            
            # Get learned importance (HAS gradients from learnable network)
            patches_flat = patches.flatten(1)  # [B, N*D]
            learned_importance = self.mask_refine(patches_flat)  # [B, N]
            
            # FIX 1: Explicitly detach unet_importance (it's from frozen UNet anyway)
            # This makes the combination cleaner - we only want gradients from learned part
            mask_prob = 0.7 * unet_importance.detach() + 0.3 * learned_importance
            
            # Now mask_prob has gradients ONLY from learned_importance (which is correct!)
        
        # FIX 2: Detach mask_prob BEFORE sorting
        # torch.argsort is non-differentiable, but we don't need gradients through sorting
        # Gradients flow through mask_prob weighting in the loss, not through index selection
        mask_prob_for_sort = mask_prob.detach() if mask_prob.requires_grad else mask_prob
        
        # Sort by importance (lower = more likely to mask)
        # Use detached version for sorting to avoid gradient issues
        ids_shuffle = torch.argsort(mask_prob_for_sort, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep least important patches visible (most important are masked)
        # This forces model to reconstruct lesion regions
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Gather visible patches
        x_masked = torch.gather(
            patches, 
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        
        # Generate binary mask
        mask = torch.ones(B, N, device=patches.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # FIX 3: Return ORIGINAL mask_prob (with gradients)
        # This is used in reconstruction loss weighting where gradients DO flow
        # Following MLO-MAE pattern: loss = loss * mask_prob (Eq. 1 in paper)
        return x_masked, mask, ids_restore, mask_prob