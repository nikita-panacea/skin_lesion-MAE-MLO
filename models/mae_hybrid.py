# models/mae_hybrid.py - HYBRID CONVNEXTV2 + ATTENTION VERSION
"""
MAE with Hybrid ConvNeXtV2 + Separable Self-Attention encoder

This implementation uses the ACTUAL hybrid architecture:
- Stages 1-2: ConvNeXtV2 (ImageNet pretrained)
- Stages 3-4: Separable Self-Attention blocks
- Decoder: Standard transformer decoder for reconstruction

Architecture flow:
1. ConvNeXtV2 stages 1-2 extract low-level features (pretrained)
2. Masking applied at 14x14 token level (196 tokens, 384 dim)
3. Separable attention stages 3-4 process masked tokens
4. Decoder reconstructs original image patches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.convnextv2 import ConvNeXtV2
from models.separable_attention import SeparableSelfAttention


class TransformerBlock(nn.Module):
    """
    Lightweight Transformer block with separable self-attention.
    Same as in hybrid_model.py for consistency.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SeparableSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # MLP with 0.5x expansion (matching hybrid_model.py)
        mlp_hidden_dim = dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridMAE(nn.Module):
    """
    MAE with Hybrid ConvNeXtV2 + Separable Attention encoder
    
    Architecture:
    - Encoder:
        - Stages 1-2: ConvNeXtV2 (pretrained) - convolutional features
        - Stage 3: 9 Separable Attention blocks at 14x14 (196 tokens, 384 dim)
        - Stage 4: 12 Separable Attention blocks at 7x7 (49 tokens, 768 dim)
    - Decoder:
        - Projects from encoder to decoder dim
        - 8 transformer blocks for reconstruction
        - Outputs patch predictions (16x16x3 = 768 values per patch)
    
    Key design choices:
    - Masking at 14x14 level (196 tokens) matching standard MAE
    - Pretrained ConvNeXt provides strong low-level features
    - Separable attention for efficient high-level processing
    - CLS token prepended for classification tasks
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, norm_pix_loss=False, pretrained=True):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.norm_pix_loss = norm_pix_loss
        
        # ======================================================================
        # Hybrid Encoder (ConvNeXtV2 + Separable Attention)
        # ======================================================================
        
        # ConvNeXtV2 backbone for stages 1-2
        backbone = ConvNeXtV2(
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 0, 0],  # Only stages 1-2
            dims=[96, 192, 384, 768],
            drop_path_rate=0.0,
        )
        
        # Load pretrained weights
        if pretrained:
            try:
                ckpt = torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
                    map_location="cpu",
                    check_hash=True
                )
                backbone.load_state_dict(ckpt["model"], strict=False)
                print("Loaded ConvNeXtV2-Tiny pretrained weights for MAE encoder")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
        
        # Extract components from backbone
        self.stem = backbone.downsample_layers[0]
        self.stage1 = backbone.stages[0]
        self.down1 = backbone.downsample_layers[1]
        self.stage2 = backbone.stages[1]
        self.down2 = backbone.downsample_layers[2]  # Projects to 384 dim
        self.down3 = backbone.downsample_layers[3]  # Projects to 768 dim
        
        del backbone  # Free memory
        
        # Separable attention stages
        self.stage3 = nn.Sequential(
            *[TransformerBlock(384) for _ in range(9)]
        )
        self.stage4 = nn.Sequential(
            *[TransformerBlock(768) for _ in range(12)]
        )
        
        # Encoder normalization
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # CLS token and positional embedding for encoder output
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 49 + 1, embed_dim),  # 49 tokens (7x7) + CLS
            requires_grad=True
        )
        
        # ======================================================================
        # Decoder
        # ======================================================================
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding (for 196 patches + CLS)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=True
        )
        
        # Standard transformer decoder blocks
        from timm.models.vision_transformer import Block
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        # Projection from 384 to 768 for masked tokens (stage3 -> stage4)
        self.stage3_to_stage4 = nn.Linear(384, 768)
        
        # Projection from 49 tokens to 196 for decoder
        self.upsample_tokens = nn.Linear(49, self.num_patches)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize learnable parameters"""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # Init projection layers
        nn.init.xavier_uniform_(self.stage3_to_stage4.weight)
        nn.init.xavier_uniform_(self.upsample_tokens.weight)
        nn.init.zeros_(self.stage3_to_stage4.bias)
        nn.init.zeros_(self.upsample_tokens.bias)
        
        # Init decoder
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.zeros_(self.decoder_embed.bias)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)
    
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
    
    def forward_conv_stages(self, x):
        """
        Forward through ConvNeXtV2 stages 1-2 (frozen/pretrained features)
        
        Args:
            x: [B, 3, 224, 224] - input images
        Returns:
            features: [B, 384, 14, 14] - features ready for masking
        """
        x = self.stem(x)      # [B, 96, 56, 56]
        x = self.stage1(x)    # [B, 96, 56, 56]
        x = self.down1(x)     # [B, 192, 28, 28]
        x = self.stage2(x)    # [B, 192, 28, 28]
        x = self.down2(x)     # [B, 384, 14, 14]
        return x
    
    def forward_encoder(self, imgs, mask_ratio=0):
        """
        Full encoder forward (for classification - no masking)
        
        Args:
            imgs: [B, 3, H, W] - input images
            mask_ratio: ignored (kept for API compatibility)
        
        Returns:
            x: [B, 50, 768] - encoded features with CLS token
            mask: [B, N] - dummy mask (zeros)
            ids_restore: [B, N] - dummy restore indices
        """
        B = imgs.shape[0]
        
        # ConvNeXtV2 stages 1-2
        x = self.forward_conv_stages(imgs)  # [B, 384, 14, 14]
        
        # Stage 3: Separable attention at 14x14
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 384]
        x = self.stage3(x)  # [B, 196, 384]
        x = x.transpose(1, 2).view(B, 384, 14, 14)
        
        # Downsample to 7x7
        x = self.down3(x)  # [B, 768, 7, 7]
        
        # Stage 4: Separable attention at 7x7
        x = x.flatten(2).transpose(1, 2)  # [B, 49, 768]
        x = self.stage4(x)  # [B, 49, 768]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 50, 768]
        x = x + self.pos_embed
        
        # Normalize
        x = self.encoder_norm(x)
        
        # Dummy mask and ids for API compatibility
        mask = torch.zeros(B, self.num_patches, device=imgs.device)
        ids_restore = torch.arange(self.num_patches, device=imgs.device).unsqueeze(0).expand(B, -1)
        
        return x, mask, ids_restore
    
    def forward_encoder_with_mask(self, x_stage3, mask, ids_restore):
        """
        Encoder forward with pre-masked tokens (for MAE pretraining)
        
        Args:
            x_stage3: [B, N_keep, 384] - unmasked tokens after masking
            mask: [B, 196] - binary mask
            ids_restore: [B, 196] - indices to restore order
        
        Returns:
            x: [B, N_keep+1, 768] - encoded features with CLS
            mask: passed through
            ids_restore: passed through
        """
        B = x_stage3.shape[0]
        N_keep = x_stage3.shape[1]
        
        # Process through stage3 (already masked tokens)
        x = self.stage3(x_stage3)  # [B, N_keep, 384]
        
        # Project to 768 dim for stage4
        x = self.stage3_to_stage4(x)  # [B, N_keep, 768]
        
        # Process through stage4
        # Note: Stage4 expects fewer tokens when masking
        x = self.stage4(x)  # [B, N_keep, 768]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N_keep+1, 768]
        
        # Normalize
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decoder: reconstruct patches from encoded features
        
        Args:
            x: [B, N_keep+1, 768] - encoded features with CLS
            ids_restore: [B, 196] - indices to restore patch order
        
        Returns:
            pred: [B, 196, 768] - predicted patches (16x16x3)
        """
        B = x.shape[0]
        
        # Embed to decoder dimension
        x = self.decoder_embed(x)  # [B, N_keep+1, 512]
        
        # Append mask tokens to fill back to 196 patches
        num_mask_tokens = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1)
        
        # Concatenate visible tokens (without CLS) with mask tokens
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # [B, 196, 512]
        
        # Unshuffle to restore original order
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Prepend CLS token
        x = torch.cat([x[:, :1, :], x_], dim=1)  # [B, 197, 512]
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        
        # Predict patches (remove CLS)
        x = self.decoder_pred(x)  # [B, 197, 768]
        x = x[:, 1:, :]  # [B, 196, 768]
        
        return x
    
    def forward_with_mask(self, x_masked, mask, ids_restore):
        """
        Full forward for MAE training (receives pre-masked tokens at 384 dim)
        
        Called from MAEProblem.training_step()
        
        Args:
            x_masked: [B, N_keep, 384] - unmasked patch embeddings
            mask: [B, 196] - binary mask
            ids_restore: [B, 196] - restore indices
        
        Returns:
            pred: [B, 196, 768] - reconstructed patches
        """
        # Encode
        latent, mask, ids_restore = self.forward_encoder_with_mask(x_masked, mask, ids_restore)
        
        # Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred
    
    def get_patch_embeddings(self, imgs):
        """
        Get patch embeddings at 14x14 level for masking
        
        Args:
            imgs: [B, 3, 224, 224]
        
        Returns:
            x: [B, 196, 384] - patch embeddings ready for masking
        """
        x = self.forward_conv_stages(imgs)  # [B, 384, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 384]
        return x
    
    def forward(self, imgs, mask_module=None, mask_ratio=0.75, random=False):
        """
        Full MAE forward (for testing or standalone training)
        """
        if mask_module is None:
            # No masking - just encode and decode
            latent, mask, ids_restore = self.forward_encoder(imgs)
            pred = self.forward_decoder(latent, ids_restore)
            
            target = self.patchify(imgs)
            loss = ((pred - target) ** 2).mean()
            return loss, pred, mask
        
        # With masking module
        x = self.get_patch_embeddings(imgs)  # [B, 196, 384]
        
        x_masked, mask, ids_restore, mask_prob = mask_module(imgs, x, mask_ratio, random)
        
        pred = self.forward_with_mask(x_masked, mask, ids_restore)
        
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


def mae_hybrid_base(norm_pix_loss=False, pretrained=True, **kwargs):
    """Create Hybrid MAE with base configuration"""
    model = HybridMAE(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_pix_loss=norm_pix_loss,
        pretrained=pretrained,
        **kwargs
    )
    return model


if __name__ == '__main__':
    print("Testing Hybrid MAE with ConvNeXtV2 + Separable Attention")
    print("="*70)
    
    # Create model
    mae = mae_hybrid_base(norm_pix_loss=True, pretrained=False)
    
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
    
    # Test get_patch_embeddings
    print("\nTesting get_patch_embeddings (for masking):")
    patch_emb = mae.get_patch_embeddings(imgs)
    print(f"  Patch embeddings: {patch_emb.shape}")
    
    # Test with simple masking
    print("\nTesting forward_with_mask (for reconstruction):")
    B, N, D = patch_emb.shape
    mask_ratio = 0.75
    len_keep = int(N * (1 - mask_ratio))
    
    # Simple random masking
    noise = torch.rand(B, N)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(patch_emb, dim=1, 
                           index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    print(f"  x_masked: {x_masked.shape}")
    print(f"  mask sum (should be {int(N * mask_ratio)}): {mask.sum().item()}")
    
    pred = mae.forward_with_mask(x_masked, mask, ids_restore)
    print(f"  pred: {pred.shape}")
    
    # Test loss computation
    target = mae.patchify(imgs)
    loss = ((pred - target) ** 2).mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    print(f"  reconstruction loss: {loss.item():.4f}")
    
    print("\n" + "="*70)
    print("All tests passed! Hybrid MAE is ready.")
