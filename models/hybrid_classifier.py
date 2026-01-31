# models/hybrid_classifier.py - FIXED VERSION v2

"""
Classifier components for MLO-MAE training

This file contains two classifier types:
1. ClassifierHead: Lightweight classifier that takes encoded features (for MLO training)
2. HybridClassifier: Full encoder+classifier (for standalone training/inference)

The ClassifierHead matches MLOMAE's FinetuneVisionTransformer pattern.
"""
import torch
import torch.nn as nn
from functools import partial


class ClassifierHead(nn.Module):
    """
    Lightweight classifier head for MLO training
    
    Takes encoded features from MAE and produces class predictions.
    Matches MLOMAE's FinetuneVisionTransformer structure.
    
    This is used as the module for ClassifierProblem in Betty MLO.
    """
    def __init__(self, embed_dim=768, num_classes=8, drop_rate=0.1, global_pool=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        
        # Layer norm before classification
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Dropout
        self.head_drop = nn.Dropout(drop_rate)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
    
    def forward(self, x, mae_model=None, pre_logits=False):
        """
        Forward pass
        
        Matches MLOMAE FinetuneVisionTransformer.forward()
        
        Args:
            x: Features from MAE encoder
               - If x is [B, N+1, D] (full sequence with CLS): uses CLS token
               - If x is [B, D] (already pooled): uses directly
            mae_model: Unused (kept for API compatibility)
            pre_logits: If True, return features before classification head
        
        Returns:
            logits: [B, num_classes] or features: [B, embed_dim]
        """
        # Handle different input shapes
        if x.dim() == 3:
            if self.global_pool:
                # Average pool over all tokens except CLS
                x = x[:, 1:, :].mean(dim=1)
            else:
                # Use CLS token (first token)
                x = x[:, 0]
        
        # x is now [B, D]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        
        if pre_logits:
            return x
        
        return self.head(x)
    
    def forward_from_features(self, features):
        """
        Classify from pre-extracted features
        
        Args:
            features: [B, embed_dim] - CLS token or pooled features
        
        Returns:
            logits: [B, num_classes]
        """
        x = self.fc_norm(features)
        x = self.head_drop(x)
        return self.head(x)


class HybridClassifier(nn.Module):
    """
    Full classifier using hybrid ConvNeXtV2 + Attention encoder
    
    This is for standalone training/inference, NOT for MLO training.
    For MLO training, use ClassifierHead with MAE encoder features.
    """
    def __init__(self, num_classes=8, pretrained_mae=None, freeze_encoder=False):
        super().__init__()
        
        # Import here to avoid circular dependencies
        from models.hybrid_model import HybridConvNeXtV2
        
        self.num_classes = num_classes
        
        # Use the hybrid model directly
        self.encoder = HybridConvNeXtV2(
            num_classes=num_classes,
            pretrained=True
        )
        
        # If pretrained MAE provided, initialize encoder weights
        if pretrained_mae is not None:
            self._init_from_mae(pretrained_mae)
        
        # Optionally freeze encoder
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    
    def _init_from_mae(self, mae_model):
        """Initialize encoder from pretrained MAE"""
        print("Initializing from pretrained MAE...")
        
        try:
            # Copy compatible weights from MAE blocks
            mae_blocks = list(mae_model.blocks)
            
            # Initialize attention stages with MAE block weights where compatible
            for i, (mae_block, enc_block) in enumerate(zip(mae_blocks, 
                                                            list(self.encoder.stage3) + list(self.encoder.stage4))):
                # Try to copy attention weights
                for name, param in enc_block.named_parameters():
                    try:
                        mae_param = dict(mae_block.named_parameters()).get(name)
                        if mae_param is not None and param.shape == mae_param.shape:
                            param.data.copy_(mae_param.data)
                    except:
                        pass
            
            print("Encoder initialized from pretrained MAE")
        except Exception as e:
            print(f"Warning: Could not fully initialize from MAE: {e}")
    
    def forward(self, x):
        """Full forward pass with encoder"""
        return self.encoder(x)
    
    def forward_from_features(self, features):
        """
        Classify from pre-extracted features
        
        This provides API compatibility with ClassifierHead
        """
        # For HybridClassifier, we expect the encoder's norm and head
        x = self.encoder.norm(features)
        x = self.encoder.head(x)
        return x
    
    def extract_features(self, x):
        """Extract features without classification head"""
        return self.encoder.forward_features(x)