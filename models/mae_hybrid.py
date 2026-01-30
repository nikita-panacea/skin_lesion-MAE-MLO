# models/mae_hybrid.py - FIXED VERSION
"""
Hybrid MAE using ConvNeXtV2+Attention encoder
CRITICAL FIXES for Betty MLO:
1. Detach mask and ids_restore before decoder
2. Detach all outputs that come from frozen modules
3. Ensure gradient flow only through learnable parameters
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


class MAEDecoder(nn.Module):
    """Lightweight MAE decoder for reconstruction"""
    def __init__(self, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, patch_size=16):
        super().__init__()
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embedding
        num_patches = (224 // patch_size) ** 2
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        # Transformer blocks
        from timm.models.vision_transformer import Block
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True)
        
        # Initialize
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self._init_pos_embed()
    
    def _init_pos_embed(self):
        """Initialize positional embedding with sin-cos"""
        from utils.pos_embed import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int((self.decoder_pos_embed.shape[1] - 1) ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(self, x, ids_restore):
        """
        Args:
            x: [B, N_visible+1, D] - encoder output (with CLS token)
            ids_restore: [B, N] - indices to restore patch order
        Returns:
            pred: [B, N, patch_size^2 * 3] - reconstructed patches
        """
        # CRITICAL: Detach ids_restore if it comes from frozen module
        ids_restore = ids_restore.detach()
        
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x


class HybridMAE(nn.Module):
    """
    MAE with Hybrid ConvNeXtV2+Attention encoder
    
    CRITICAL: All outputs from masking module must be detached
    to prevent retain_grad errors in Betty MLO framework
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_embed_dim=768, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, norm_pix_loss=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, encoder_embed_dim),
            requires_grad=False
        )
        
        # Hybrid encoder (ConvNeXtV2 + Attention)
        from models.hybrid_model import HybridConvNeXtV2
        self.encoder = HybridConvNeXtV2(num_classes=1000, pretrained=True)
        # Remove classification head - we'll use features
        self.encoder.head = nn.Identity()
        
        # Decoder
        self.decoder = MAEDecoder(
            encoder_embed_dim, decoder_embed_dim, decoder_depth,
            decoder_num_heads, patch_size
        )
        
        # Initialize
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self._init_pos_embed()
    
    def _init_pos_embed(self):
        """Initialize positional embedding"""
        from utils.pos_embed import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def forward_encoder(self, x, mask_module, images, mask_ratio=0.75, random=False):
        """
        Encode with masking
        
        Args:
            x: [B, N, D] - patch embeddings (BEFORE masking)
            mask_module: masking module
            images: [B, 3, H, W] - original images for UNet
            mask_ratio: ratio of patches to mask
            random: use random masking
        
        Returns:
            latent: [B, N+1, D] - encoded features (ALL patches + CLS)
            mask: [B, N] - binary mask
            ids_restore: [B, N] - restore indices
            mask_prob: [B, N] - masking probabilities
        """
        # CRITICAL: Get masking from module and DETACH outputs
        x_masked, mask, ids_restore, mask_prob = mask_module(
            images, x, mask_ratio, random
        )
        
        # CRITICAL FIX: Detach mask and ids_restore to break gradient flow
        # from frozen UNet parameters
        mask = mask.detach()
        ids_restore = ids_restore.detach()
        # mask_prob stays as-is for gradient flow through learnable refinement
        
        # FIXED: Don't use masked patches for encoder
        # Instead, encode the FULL image through the hybrid encoder
        # The masking is only used for the reconstruction loss
        
        # Pass full image through hybrid encoder
        B, _, H_img, W_img = images.shape
        latent_features = self.encoder(images)  # [B, D_out, H_feat, W_feat]
        
        # Convert encoder output back to patch sequence
        # encoder.forward returns features before the head
        # For hybrid model, this should be [B, 768, 7, 7] after all stages
        latent_features = latent_features.flatten(2).transpose(1, 2)  # [B, 49, 768]
        
        # Interpolate/pad to match original patch count if needed
        if latent_features.shape[1] != self.num_patches:
            # Resize to original patch grid
            patch_size_enc = int(latent_features.shape[1] ** 0.5)
            patch_size_orig = int(self.num_patches ** 0.5)
            
            latent_features = latent_features.transpose(1, 2).reshape(
                B, -1, patch_size_enc, patch_size_enc
            )
            latent_features = F.interpolate(
                latent_features,
                size=(patch_size_orig, patch_size_orig),
                mode='bilinear',
                align_corners=False
            )
            latent_features = latent_features.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        latent = torch.cat([cls_tokens, latent_features], dim=1)  # [B, N+1, D]
        
        return latent, mask, ids_restore, mask_prob
    
    def forward(self, imgs, mask_module, mask_ratio=0.75, random=False):
        """
        Full MAE forward pass
        
        Args:
            imgs: [B, 3, H, W] - input images
            mask_module: masking module
            mask_ratio: masking ratio
            random: use random masking
        
        Returns:
            loss: reconstruction loss
            pred: [B, N, p^2*3] - predictions
            mask: [B, N] - binary mask
        """
        # Patchify
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        
        # Encode with masking
        latent, mask, ids_restore, mask_prob = self.forward_encoder(
            x, mask_module, imgs, mask_ratio, random
        )
        
        # Decode
        pred = self.decoder(latent, ids_restore)
        
        # Compute loss
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Weight by masking probability (for gradient flow)
        loss = loss * mask_prob
        
        # Mean loss on removed patches only
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask


# Factory functions
def mae_hybrid_base(pretrained_unet_path='unet/unet.pkl', **kwargs):
    """MAE with Hybrid Base encoder"""
    model = HybridMAE(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs
    )
    return model


if __name__ == '__main__':
    print("Testing Hybrid MAE")
    print("="*60)
    
    from models.masking_module import UNetMaskingModule
    
    # Create models
    mae = mae_hybrid_base(norm_pix_loss=True)
    mask_module = UNetMaskingModule(
        pretrained_unet_path='unet/unet.pkl',
        num_patches=196,
        embed_dim=768,
        learnable=True
    )
    
    # Test forward
    imgs = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        loss, pred, mask = mae(imgs, mask_module, mask_ratio=0.75, random=False)
    
    print(f"\nForward pass:")
    print(f"  Input: {imgs.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pred: {pred.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Masked ratio: {mask.mean():.2%}")
    
    print("\n Model works!")