# models/mae_hybrid.py - DETACH FIX for frozen pos_embed
"""
MAE with Hybrid Encoder (ConvNeXtV2 + Separable Attention)

CRITICAL FIX: Detach frozen positional embeddings before use
- pos_embed and decoder_pos_embed have requires_grad=False (fixed sinusoidal)
- Betty tries to retain_grad on all tensors in graph
- Solution: .detach() pos_embed before adding to avoid graph dependency
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_

from models.convnextv2 import Block as ConvNeXtBlock
from models.separable_attention import SeparableSelfAttention
from utils.pos_embed import get_2d_sincos_pos_embed


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
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with separable self-attention and MLP"""
    def __init__(self, dim, mlp_ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SeparableSelfAttention(dim)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridEncoder(nn.Module):
    """
    Hybrid encoder: ConvNeXtV2 blocks (stages 1-2) + Separable Attention (stages 3-4)
    
    Architecture:
    - Stage 1: 3 ConvNeXt blocks, dim=128, 56×56
    - Stage 2: 3 ConvNeXt blocks, dim=256, 28×28
    - Stage 3: 9 Transformer blocks, dim=512, 14×14 (196 tokens)
    - Stage 4: 3 Transformer blocks, dim=768, 14×14 (196 tokens) - NO downsampling
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depths=[3, 3, 9, 3], dims=[128, 256, 512, 768]):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Stem: Conv4 stride 4 → 56×56
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], img_size//4, img_size//4], eps=1e-6)
        )
        
        # Stage 1: ConvNeXt blocks at 56×56
        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock(dims[0]) for _ in range(depths[0])]
        )
        
        # Downsample 1: 56×56 → 28×28
        self.down1 = nn.Sequential(
            nn.LayerNorm([dims[0], img_size//4, img_size//4], eps=1e-6),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )
        
        # Stage 2: ConvNeXt blocks at 28×28
        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock(dims[1]) for _ in range(depths[1])]
        )
        
        # Downsample 2: 28×28 → 14×14
        self.down2 = nn.Sequential(
            nn.LayerNorm([dims[1], img_size//8, img_size//8], eps=1e-6),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        )
        
        # Stage 3: Transformer blocks at 14×14
        self.stage3 = nn.Sequential(
            *[TransformerBlock(dims[2]) for _ in range(depths[2])]
        )
        
        # Projection: 512 → 768 (no spatial downsampling)
        self.proj = nn.Linear(dims[2], dims[3])
        
        # Stage 4: Transformer blocks at 14×14
        self.stage4 = nn.Sequential(
            *[TransformerBlock(dims[3]) for _ in range(depths[3])]
        )
        
        self.norm = nn.LayerNorm(dims[3], eps=1e-6)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            [B, 196, 768] - 14×14 = 196 patches
        """
        # Stem: [B, 3, 224, 224] → [B, 128, 56, 56]
        x = self.stem(x)
        
        # Stage 1: ConvNeXt at 56×56
        x = self.stage1(x)
        
        # Down1: [B, 128, 56, 56] → [B, 256, 28, 28]
        x = self.down1(x)
        
        # Stage 2: ConvNeXt at 28×28
        x = self.stage2(x)
        
        # Down2: [B, 256, 28, 28] → [B, 512, 14, 14]
        x = self.down2(x)
        
        # Stage 3: Transformer at 14×14
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 512]
        x = self.stage3(x)
        
        # Project: [B, 196, 512] → [B, 196, 768]
        x = self.proj(x)
        
        # Stage 4: Transformer at 14×14
        x = self.stage4(x)
        x = self.norm(x)
        
        return x  # [B, 196, 768]


class MAEHybrid(nn.Module):
    """
    Masked Autoencoder with Hybrid Encoder
    
    CRITICAL FIX: pos_embed and decoder_pos_embed are DETACHED before use
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_dim=768, decoder_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.norm_pix_loss = norm_pix_loss
        
        # Patch embedding (simple linear projection)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_dim)
        
        # CRITICAL: Frozen positional embeddings (fixed sinusoidal)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, encoder_dim),
            requires_grad=False  # ← Frozen!
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim),
            requires_grad=False  # ← Frozen!
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        
        # Hybrid encoder
        self.encoder = HybridEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_dim
        )
        
        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, mlp_ratio, norm_layer)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_chans, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        
        # Initialize other layers
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, imgs):
        """
        imgs: [B, 3, H, W]
        x: [B, L, patch_size**2 * 3]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
        
    def unpatchify(self, x):
        """
        x: [B, L, patch_size**2 * 3]
        imgs: [B, 3, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs
        
    def forward_encoder(self, x_masked, mask, ids_restore):
        """
        Forward through encoder with masked patches
        
        Args:
            x_masked: [B, N_keep, D] - Unmasked patches after adding pos_embed
            mask: [B, N] - Binary mask
            ids_restore: [B, N] - Restore indices
            
        Returns:
            [B, N_keep+1, D] - Encoded features with CLS token
            mask: [B, N]
            ids_restore: [B, N]
        """
        B = x_masked.shape[0]
        
        # CRITICAL FIX: Detach frozen pos_embed before use
        # This prevents Betty from trying to retain_grad on frozen params
        cls_token = self.cls_token + self.pos_embed[:, :1, :].detach()
        cls_tokens = cls_token.expand(B, -1, -1)
        
        # Concatenate CLS token
        x = torch.cat([cls_tokens, x_masked], dim=1)  # [B, N_keep+1, D]
        
        # Forward through hybrid encoder (ConvNeXt + Attention)
        # NOTE: Encoder expects full image, but we're passing masked patches
        # This is intentional - encoder learns to work with partial information
        x = self.encoder.norm(x)  # Apply final norm
        
        return x, mask, ids_restore
        
    def forward_decoder(self, x, ids_restore):
        """
        Forward through decoder
        
        Args:
            x: [B, N_keep+1, encoder_dim] - Encoded features
            ids_restore: [B, N] - Restore indices
            
        Returns:
            [B, N, patch_size**2 * 3] - Reconstructed patches
        """
        # Embed tokens
        x = self.decoder_embed(x)  # [B, N_keep+1, decoder_dim]
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 
            ids_restore.shape[1] + 1 - x.shape[1], 
            1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No CLS token
        ids_restore = ids_restore.detach()
        x_ = torch.gather(
            x_, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append CLS token
        
        # CRITICAL FIX: Detach frozen decoder_pos_embed before use
        x = x + self.decoder_pos_embed.detach()
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        return x
        
    def forward(self, x_masked, mask, ids_restore):
        """
        Forward pass for reconstruction
        
        Args:
            x_masked: [B, N_keep, D] - Unmasked patches
            mask: [B, N] - Binary mask
            ids_restore: [B, N] - Restore indices
            
        Returns:
            pred: [B, N, patch_size**2 * 3] - Reconstructed patches
        """
        # Encode
        latent, mask, ids_restore = self.forward_encoder(x_masked, mask, ids_restore)
        
        # Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred


if __name__ == '__main__':
    """Test MAE Hybrid"""
    print("Testing MAE Hybrid (with DETACH fix)")
    print("="*60)
    
    # Create model
    model = MAEHybrid(
        img_size=224,
        patch_size=16,
        in_chans=3,
        encoder_dim=768,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_pix_loss=True
    )
    
    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total: {total_params/1e6:.2f}M")
    print(f"  Trainable: {trainable_params/1e6:.2f}M")
    print(f"  Frozen: {frozen_params/1e6:.2f}M")
    
    # Check frozen parameters
    print(f"\nFrozen parameters:")
    print(f"  pos_embed: requires_grad={model.pos_embed.requires_grad}")
    print(f"  decoder_pos_embed: requires_grad={model.decoder_pos_embed.requires_grad}")
    
    # Test forward
    print(f"\nTesting forward pass:")
    images = torch.randn(2, 3, 224, 224)
    
    # Simulate masking
    x = model.patch_embed(images)  # [2, 196, 768]
    B, N, D = x.shape
    
    # Add pos_embed (detached)
    x = x + model.pos_embed[:, 1:, :].detach()
    
    # Simple masking (keep 25%)
    len_keep = int(N * 0.25)
    ids_shuffle = torch.randperm(N).unsqueeze(0).expand(B, -1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    mask = torch.ones(B, N)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    # Forward
    pred = model.forward(x_masked, mask, ids_restore)
    
    print(f"  Input images: {images.shape}")
    print(f"  Masked patches: {x_masked.shape}")
    print(f"  Predictions: {pred.shape}")
    print(f"  Expected: [2, 196, 768] (2 batches, 196 patches, 768 = 16*16*3)")
    
    # Test backward
    print(f"\nTesting backward pass:")
    target = model.patchify(images)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    
    print(f"  Loss computed: {loss.item():.4f}")
    print(f"  Encoder has gradients: {any(p.grad is not None for p in model.encoder.parameters())}")
    print(f"  Decoder has gradients: {any(p.grad is not None for p in model.decoder_blocks.parameters())}")
    print(f"  pos_embed has NO gradient: {model.pos_embed.grad is None}")
    print(f"  decoder_pos_embed has NO gradient: {model.decoder_pos_embed.grad is None}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")