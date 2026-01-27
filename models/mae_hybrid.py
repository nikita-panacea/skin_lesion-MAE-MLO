# models/mae_hybrid.py
"""
Masked Autoencoder with Hybrid ConvNeXtV2 + Attention Encoder
Adapted for skin lesion image reconstruction

FIXES:
1. Proper forward pass without mixing input types
2. Clear separation of embedding, encoding, and decoding
3. Maintains gradient flow for MLO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for MAE"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        return x


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction"""
    def __init__(self, embed_dim=768, decoder_embed_dim=512, 
                 decoder_depth=4, num_patches=196, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        # Projection from encoder to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), 
            requires_grad=True
        )
        
        # Decoder blocks (simple transformer)
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=8,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            patch_size ** 2 * 3,  # RGB patches
            bias=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embedding with sine-cosine
        pos_embed = self._get_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5)
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def _get_sincos_pos_embed(embed_dim, grid_size):
        """Generate 2D sincos positional embedding"""
        import numpy as np
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        # Use half dims for each axis
        emb_h = MAEDecoder._get_1d_sincos_from_grid(embed_dim // 2, grid[0])
        emb_w = MAEDecoder._get_1d_sincos_from_grid(embed_dim // 2, grid[1])
        pos_embed = np.concatenate([emb_h, emb_w], axis=1)
        return pos_embed
    
    @staticmethod
    def _get_1d_sincos_from_grid(embed_dim, pos):
        import numpy as np
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega
        
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    def forward(self, x, ids_restore):
        """
        Args:
            x: [B, N_visible, embed_dim] - visible patches from encoder
            ids_restore: [B, N] - indices to restore original order
        Returns:
            pred: [B, N, patch_size^2 * 3] - reconstructed patches
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 
            ids_restore.shape[1] - x.shape[1], 
            1
        )
        x = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle
        x = torch.gather(
            x, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        return x


class MAEHybrid(nn.Module):
    """
    Masked Autoencoder with Hybrid Encoder
    Uses Transformer blocks for processing masked patches
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_dim=768, decoder_dim=512, decoder_depth=4,
                 norm_pix_loss=True):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoder_dim = encoder_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_dim
        )
        
        # Encoder with separable attention
        self.encoder = self._build_encoder(encoder_dim)
        
        # Decoder
        self.decoder = MAEDecoder(
            embed_dim=encoder_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=decoder_depth,
            num_patches=self.num_patches,
            patch_size=patch_size
        )
    
    def _build_encoder(self, dim):
        """Build encoder with separable attention"""
        from models.separable_attention import SeparableSelfAttention
        
        class TransformerBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim, eps=1e-6)
                self.attn = SeparableSelfAttention(dim)
                self.norm2 = nn.LayerNorm(dim, eps=1e-6)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            
            def forward(self, x):
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x
        
        # Build encoder
        encoder = nn.ModuleDict()
        
        # Transformer blocks (12 blocks like ViT-Base)
        encoder['blocks'] = nn.Sequential(
            *[TransformerBlock(dim) for _ in range(12)]
        )
        
        # Final norm
        encoder['norm'] = nn.LayerNorm(dim, eps=1e-6)
        
        return encoder
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs
    
    def forward_encoder(self, x, mask, ids_restore):
        """
        Forward through encoder with masking
        Args:
            x: [B, N_visible, encoder_dim] - visible patches after masking
            mask: [B, N] - binary mask (0 is keep, 1 is remove)
            ids_restore: [B, N] - indices to restore order
        Returns:
            latent: [B, N_visible, encoder_dim] - encoded features
        """
        # Apply transformer blocks
        x = self.encoder['blocks'](x)
        
        # Final norm
        x = self.encoder['norm'](x)
        
        return x
    
    def forward_decoder(self, latent, ids_restore):
        """
        Forward through decoder
        Args:
            latent: [B, N_visible, encoder_dim]
            ids_restore: [B, N]
        Returns:
            pred: [B, N, patch_size^2 * 3]
        """
        pred = self.decoder(latent, ids_restore)
        return pred
    
    def forward_loss(self, imgs, pred, mask, mask_prob=None):
        """
        Compute reconstruction loss
        Args:
            imgs: [B, 3, H, W] - original images
            pred: [B, N, patch_size^2 * 3] - predicted patches
            mask: [B, N] - binary mask
            mask_prob: [B, N] - masking probabilities (optional)
        Returns:
            loss: scalar
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean per patch
        
        # Weight by mask probability if provided
        if mask_prob is not None:
            loss = loss * mask_prob
        
        # Mean loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, x_masked, mask, ids_restore, mask_prob=None):
        """
        Full forward pass
        Args:
            x_masked: [B, N_visible, encoder_dim] - masked patches (already embedded)
            mask: [B, N] - binary mask
            ids_restore: [B, N] - restore indices
            mask_prob: [B, N] - masking probabilities (not used in forward, only in loss)
        Returns:
            pred: [B, N, patch_size^2 * 3] - reconstructed patches
        """
        # x_masked are the visible patches after masking, already embedded
        latent = self.forward_encoder(x_masked, mask, ids_restore)
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred