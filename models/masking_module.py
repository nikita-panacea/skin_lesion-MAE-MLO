# models/masking_module.py - COMPLETE FIX v3
"""
UNet-based Masking Module for MLO-MAE

CRITICAL FIXES (v3):
1. UNet stored in a container dict to prevent nn.Module registration
2. All frozen UNet computations completely isolated (no_grad + detach + eval)
3. Only learnable refinement MLP participates in hypergradient computation
4. Proper gradient flow for Betty MLO framework
5. Matches MLOMAE MLOPatchMasking pattern
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class UNetMaskingModule(nn.Module):
    """
    Masking module combining frozen UNet + learned refinement
    
    Design (matching MLOMAE MLOPatchMasking):
    - UNet (frozen): Provides spatial importance baseline
    - Refinement (learnable): Task-specific MLP adaptation
    - Output: Masking probabilities with proper gradient flow
    
    Betty MLO compatibility:
    - UNet is stored in a dict container to prevent PyTorch submodule registration
    - Only refinement MLP is registered and tracked by Betty
    - parameters() only returns MLP parameters
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
        
        # CRITICAL: Store UNet in a dict to prevent nn.Module registration
        # PyTorch's __setattr__ registers any nn.Module assigned to self.xxx
        # But it doesn't look inside dict/list containers
        object.__setattr__(self, '_unet_container', {'model': None, 'device': None})
        
        # Load frozen UNet
        if use_unet:
            self._load_unet(pretrained_unet_path)
        
        # Learned refinement MLP (this IS registered as submodule)
        # Matches MLOMAE's MLOPatchMasking structure
        if learnable:
            hidden_size = 512
            self.first_hidden_layer = nn.Sequential(
                nn.Linear(num_patches * embed_dim, hidden_size),
                nn.ReLU(inplace=True)
            )
            self.output_layer = nn.Linear(hidden_size, num_patches)
            
            # Initialize weights
            for m in [self.first_hidden_layer, self.output_layer]:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _load_unet(self, pretrained_unet_path):
        """Load UNet as a frozen, non-tracked model"""
        from unet import UNet
        unet = UNet(n_channels=3, n_classes=1, bilinear=False)
        
        if Path(pretrained_unet_path).exists():
            state_dict = torch.load(pretrained_unet_path, map_location='cpu', 
                                   weights_only=True)
            unet.load_state_dict(state_dict)
            print(f" Loaded pretrained UNet from {pretrained_unet_path}")
        else:
            print(f"  Warning: UNet checkpoint not found at {pretrained_unet_path}")
        
        # CRITICAL: Freeze ALL parameters
        for param in unet.parameters():
            param.requires_grad = False
        
        # Set to eval mode permanently
        unet.eval()
        
        # Store in container dict (bypasses nn.Module registration)
        self._unet_container['model'] = unet
    
    def _ensure_unet_on_device(self, device):
        """Move UNet to correct device if needed"""
        unet = self._unet_container['model']
        current_device = self._unet_container['device']
        
        if unet is not None and current_device != device:
            unet = unet.to(device)
            self._unet_container['model'] = unet
            self._unet_container['device'] = device
            # Re-freeze after moving
            for param in unet.parameters():
                param.requires_grad = False
            unet.eval()
    
    @torch.no_grad()
    def _compute_unet_importance(self, images):
        """
        Compute importance map from frozen UNet
        
        CRITICAL: Entire computation in no_grad context
        Returns a completely detached tensor
        """
        self._ensure_unet_on_device(images.device)
        
        unet = self._unet_container['model']
        if unet is None:
            # Fallback to uniform importance if UNet not available
            B = images.shape[0]
            return torch.ones(B, self.num_patches, device=images.device) * 0.5
        
        # Forward through frozen UNet
        unet.eval()  # Ensure eval mode
        importance_map = unet(images)  # [B, 1, H, W]
        importance_map = torch.sigmoid(importance_map)
        
        # Resize to patch grid
        patch_size = int(self.num_patches ** 0.5)
        importance_map = F.interpolate(
            importance_map,
            size=(patch_size, patch_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Flatten and detach: [B, N]
        importance = importance_map.flatten(1).detach()
        
        return importance
    
    def forward(self, images, patch_embeddings, mask_ratio=0.75, random=False):
        """
        Generate masking probabilities and apply masking
        
        Following MLOMAE MLOPatchMasking pattern exactly.
        
        Args:
            images: [B, 3, H, W] - For UNet importance (ignored if use_unet=False)
            patch_embeddings: [B, N, D] - Input to refinement MLP
            mask_ratio: Masking ratio (default 0.75)
            random: Use random masking baseline (for ablation)
        
        Returns:
            x_masked: [B, N_keep, D] - Kept (unmasked) patches
            mask: [B, N] - Binary mask (0=keep, 1=remove) 
            ids_restore: [B, N] - Indices to restore original order
            mask_prob: [B, N] - Masking probabilities (gradients flow through this)
        """
        B, N, D = patch_embeddings.shape
        len_keep = int(N * (1 - mask_ratio))
        
        if random:
            # Random baseline - no learning
            mask_prob = torch.rand(B, N, device=patch_embeddings.device)
        else:
            # Compute mask_prob from learnable refinement
            # This matches MLOMAE's MLOPatchMasking.forward()
            
            if self.learnable:
                # Flatten patch embeddings
                x_flat = patch_embeddings.flatten(1)  # [B, N*D]
                
                # Forward through MLP (this has gradients)
                x2 = self.first_hidden_layer(x_flat)
                x2 = self.output_layer(x2)
                mask_prob = torch.sigmoid(x2)  # [B, N]
            else:
                mask_prob = torch.rand(B, N, device=patch_embeddings.device)
            
            # Optionally blend with UNet importance (detached, no gradients)
            if self.use_unet and self._unet_container['model'] is not None:
                unet_importance = self._compute_unet_importance(images)
                # Blend: learned refinement + UNet prior
                # UNet is detached so gradients only flow through mask_prob
                mask_prob = 0.5 * mask_prob + 0.5 * unet_importance
        
        # Sort by probability (higher = more likely to mask)
        ids_shuffle = torch.argsort(mask_prob, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep (lowest probability = least important to mask)
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
        
        # Return: x_masked and mask_prob keep gradients, mask/ids_restore are indices
        return x_masked, mask, ids_restore, mask_prob


if __name__ == '__main__':
    """Test masking module with gradient flow verification"""
    import sys
    sys.path.append('.')
    
    print("Testing UNet Masking Module (Complete Fix v3)")
    print("="*60)
    
    # Create module (without UNet for quick test)
    module = UNetMaskingModule(
        pretrained_unet_path='unet/unet.pkl',
        num_patches=196,
        embed_dim=768,
        learnable=True,
        use_unet=False  # Skip UNet for this test
    )
    
    # Count parameters - only learnable MLP should be counted
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"\nParameters (MLP only, UNet not counted):")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    # Verify UNet is NOT in parameters
    param_names = [n for n, _ in module.named_parameters()]
    has_unet_params = any('unet' in n.lower() for n in param_names)
    print(f"  UNet in parameters: {has_unet_params} (should be False)")
    
    # Test forward
    images = torch.randn(2, 3, 224, 224)
    patch_emb = torch.randn(2, 196, 768, requires_grad=True)
    
    x_masked, mask, ids_restore, mask_prob = module(images, patch_emb, mask_ratio=0.75)
    
    print(f"\nForward pass:")
    print(f"  x_masked: {x_masked.shape}")
    print(f"  mask: {mask.shape}, requires_grad={mask.requires_grad}")
    print(f"  ids_restore: {ids_restore.shape}")
    print(f"  mask_prob: {mask_prob.shape}, requires_grad={mask_prob.requires_grad}")
    
    # Test gradient flow
    print(f"\nGradient flow test:")
    loss = mask_prob.sum()
    loss.backward()
    
    # Check MLP gradients
    mlp_has_grad = any(p.grad is not None for p in module.first_hidden_layer.parameters())
    mlp_has_grad = mlp_has_grad or (module.output_layer.weight.grad is not None)
    
    print(f"  MLP has gradients: {mlp_has_grad}")
    print(f"  Input patch_emb.grad exists: {patch_emb.grad is not None}")
    
    # Verify all parameters require grad
    all_require_grad = all(p.requires_grad for p in module.parameters())
    print(f"  All parameters require grad: {all_require_grad}")
    
    if mlp_has_grad and not has_unet_params and all_require_grad:
        print("\n All tests passed!")
    else:
        print("\n Some tests failed!")
    
    # Test with random baseline
    print("\nTesting random baseline:")
    module.zero_grad()
    x_masked_r, mask_r, ids_r, prob_r = module(images, patch_emb.detach(), 
                                                mask_ratio=0.75, random=True)
    print(f"  mask_prob requires_grad (should be False): {prob_r.requires_grad}")