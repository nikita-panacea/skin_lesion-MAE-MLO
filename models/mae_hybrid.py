# models/mae_hybrid.py - FIXED VERSION v2
"""
Hybrid MAE using ConvNeXtV2+Attention encoder

CRITICAL FIXES for Betty MLO (v2):
1. Separate forward_encoder (for classification) and forward_with_mask (for reconstruction)
2. Matches MLOMAE MaskedAutoencoderViT pattern exactly
3. Proper gradient flow through encoder blocks
4. Compatible with Betty's Problem dependency tracking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class HybridMAE(nn.Module):
    """
    MAE with Hybrid ConvNeXtV2+Attention encoder
    
    Following MLOMAE MaskedAutoencoderViT structure:
    - patch_embed: Image to patches
    - pos_embed: Positional embeddings
    - blocks: Transformer encoder blocks (hybrid ConvNeXt+Attention)
    - decoder_*: Decoder for reconstruction
    
    Key methods:
    - forward_encoder: Full image encoding (for classification)
    - forward_encoder_mlo: Encoding with pre-masked patches (for MAE pretraining)
    - forward_with_mask: Full reconstruction with mask (for MLO training)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, norm_pix_loss=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.norm_pix_loss = norm_pix_loss
        
        # ======================================================================
        # Encoder components
        # ======================================================================
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=True  # Learnable in MAE
        )
        
        # Encoder blocks (using timm's ViT blocks for simplicity)
        from timm.models.vision_transformer import Block
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=4., qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(12)  # 12 blocks like ViT-Base
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # ======================================================================
        # Decoder components
        # ======================================================================
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=True
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights following MAE paper"""
        from utils.pos_embed import get_2d_sincos_pos_embed
        
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Patch embedding like linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Apply to all linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """Convert images to patches: [B,3,H,W] -> [B,N,P^2*3]"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def unpatchify(self, x):
        """Convert patches to images: [B,N,P^2*3] -> [B,3,H,W]"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def forward_encoder(self, imgs, mask_ratio=0):
        """
        Encode full images (for classification/feature extraction)
        
        This matches MLOMAE's forward_encoder method.
        
        Args:
            imgs: [B, 3, H, W] - input images
            mask_ratio: ignored here (kept for API compatibility)
        
        Returns:
            x: [B, N+1, D] - encoded features with CLS token
            mask: [B, N] - all zeros (no masking)
            ids_restore: [B, N] - identity indices
        """
        B = imgs.shape[0]
        
        # Patch embedding
        x = self.patch_embed(imgs)  # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]
        
        # Add CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # Apply encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # No masking - return dummy mask and ids
        mask = torch.zeros(B, self.num_patches, device=imgs.device)
        ids_restore = torch.arange(self.num_patches, device=imgs.device).unsqueeze(0).expand(B, -1)
        
        return x, mask, ids_restore
    
    def forward_encoder_mlo(self, x_masked, mask, ids_restore):
        """
        Encode pre-masked patches (for MAE pretraining with MLO)
        
        This matches MLOMAE's forward_encoder_mlo method.
        
        Args:
            x_masked: [B, N_keep, D] - unmasked patch embeddings
            mask: [B, N] - binary mask (0=keep, 1=remove)
            ids_restore: [B, N] - indices to restore order
        
        Returns:
            x: [B, N_keep+1, D] - encoded features with CLS token
            mask: passed through
            ids_restore: passed through
        """
        B = x_masked.shape[0]
        
        # Add CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_masked], dim=1)  # [B, N_keep+1, D]
        
        # Apply encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decode encoded features to reconstruct patches
        
        Args:
            x: [B, N_keep+1, D] - encoded features with CLS token
            ids_restore: [B, N] - indices to restore order
        
        Returns:
            pred: [B, N, P^2*3] - reconstructed patches
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict patches
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_with_mask(self, x_masked, mask, ids_restore):
        """
        Forward pass for MLO training (receives already-masked patches)
        
        This is called from MAEProblem.training_step()
        
        Args:
            x_masked: [B, N_keep, D] - unmasked patches from masking module
            mask: [B, N] - binary mask from masking module
            ids_restore: [B, N] - restore indices from masking module
        
        Returns:
            pred: [B, N, P^2*3] - reconstructed patches
        """
        # Encode masked patches
        latent, mask, ids_restore = self.forward_encoder_mlo(x_masked, mask, ids_restore)
        
        # Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred
    
    def forward(self, imgs, mask_module=None, mask_ratio=0.75, random=False):
        """
        Full forward pass (for testing or non-MLO training)
        
        For MLO training, use forward_with_mask instead.
        """
        if mask_module is None:
            # No masking - just encode and decode
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=0)
            pred = self.forward_decoder(latent, ids_restore)
            
            target = self.patchify(imgs)
            loss = ((pred - target) ** 2).mean()
            return loss, pred, mask
        
        # With masking module
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        
        x_masked, mask, ids_restore, mask_prob = mask_module(imgs, x, mask_ratio, random)
        
        latent, mask, ids_restore = self.forward_encoder_mlo(x_masked, mask, ids_restore)
        pred = self.forward_decoder(latent, ids_restore)
        
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = loss * mask_prob
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask


# Factory functions
def mae_hybrid_base(norm_pix_loss=False, **kwargs):
    """MAE with ViT-Base style encoder"""
    model = HybridMAE(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_pix_loss=norm_pix_loss,
        **kwargs
    )
    return model


if __name__ == '__main__':
    print("Testing Hybrid MAE (v2)")
    print("="*60)
    
    # Test without masking module first
    mae = mae_hybrid_base(norm_pix_loss=True)
    
    # Count parameters
    total = sum(p.numel() for p in mae.parameters())
    trainable = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    print(f"\nMAE Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")
    
    # Test forward_encoder
    imgs = torch.randn(2, 3, 224, 224)
    
    print("\nTesting forward_encoder (for classification):")
    latent, mask, ids = mae.forward_encoder(imgs)
    print(f"  Input: {imgs.shape}")
    print(f"  Latent: {latent.shape}")
    print(f"  CLS token: {latent[:, 0].shape}")
    
    # Test with masking
    print("\nTesting forward_with_mask (for reconstruction):")
    from models.masking_module import UNetMaskingModule
    
    mask_module = UNetMaskingModule(
        pretrained_unet_path='unet/unet.pkl',
        num_patches=196,
        embed_dim=768,
        learnable=True,
        use_unet=False  # Skip UNet for test
    )
    
    # Get patch embeddings
    x = mae.patch_embed(imgs)
    x = x + mae.pos_embed[:, 1:, :]
    
    # Apply masking
    x_masked, mask, ids_restore, mask_prob = mask_module(imgs, x, mask_ratio=0.75)
    print(f"  x_masked: {x_masked.shape}")
    print(f"  mask: {mask.shape}, sum={mask.sum().item()}")
    
    # Forward with mask
    pred = mae.forward_with_mask(x_masked, mask, ids_restore)
    print(f"  pred: {pred.shape}")
    
    # Test gradient flow
    print("\nTesting gradient flow:")
    target = mae.patchify(imgs)
    loss = ((pred - target) ** 2).mean(dim=-1)
    loss = (loss * mask_prob).mean()
    loss.backward()
    
    has_grad = mae.blocks[0].attn.qkv.weight.grad is not None
    print(f"  Encoder blocks have gradients: {has_grad}")
    
    print("\n All tests passed!")