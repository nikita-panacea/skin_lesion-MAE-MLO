# models/hybrid_classifier.py

"""
Hybrid Classifier for fine-tuning stage
Wraps the hybrid encoder for classification

FIXES:
1. Simplified initialization
2. Proper forward pass
3. Compatible with MLO framework
"""
import torch
import torch.nn as nn


class HybridClassifier(nn.Module):
    """
    Classifier using hybrid ConvNeXtV2 + Attention encoder
    Can be initialized from pretrained MAE encoder
    """
    def __init__(self, num_classes=8, pretrained_mae=None, freeze_encoder=False):
        super().__init__()
        
        # Import here to avoid circular dependencies
        from models.hybrid_model import HybridConvNeXtV2
        
        self.num_classes = num_classes
        
        # Use the hybrid model directly
        self.encoder = HybridConvNeXtV2(
            num_classes=num_classes,
            pretrained=True  # Use ImageNet pretraining
        )
        
        # If pretrained MAE provided, initialize encoder weights
        if pretrained_mae is not None:
            self._init_from_mae(pretrained_mae)
        
        # Optionally freeze encoder
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if 'head' not in name:  # Keep head trainable
                    param.requires_grad = False
    
    def _init_from_mae(self, mae_model):
        """Initialize encoder from pretrained MAE"""
        # The MAE encoder uses transformer blocks
        # We can copy compatible weights to the attention stages
        
        mae_state = mae_model.encoder
        
        # Copy transformer block weights to stage3 and stage4
        # MAE has 12 blocks, we have 9+12=21 in hybrid model
        # We'll initialize the attention stages with MAE weights
        
        print("Initializing attention stages from MAE encoder...")
        
        try:
            # Get MAE encoder blocks
            mae_blocks = mae_state['blocks']
            
            # Initialize stage3 (9 blocks) with first 9 MAE blocks
            for i in range(min(9, len(mae_blocks))):
                src_block = mae_blocks[i]
                dst_block = self.encoder.stage3[i]
                
                # Copy compatible parameters
                for name, param in dst_block.named_parameters():
                    if hasattr(src_block, name.split('.')[0]):
                        src_param = src_block
                        for attr in name.split('.'):
                            src_param = getattr(src_param, attr, None)
                            if src_param is None:
                                break
                        
                        if src_param is not None and param.shape == src_param.shape:
                            param.data.copy_(src_param.data)
            
            # Initialize stage4 (12 blocks) with remaining MAE blocks
            for i in range(min(12, len(mae_blocks))):
                mae_idx = min(i, len(mae_blocks) - 1)
                src_block = mae_blocks[mae_idx]
                dst_block = self.encoder.stage4[i]
                
                # Copy compatible parameters
                for name, param in dst_block.named_parameters():
                    if hasattr(src_block, name.split('.')[0]):
                        src_param = src_block
                        for attr in name.split('.'):
                            src_param = getattr(src_param, attr, None)
                            if src_param is None:
                                break
                        
                        if src_param is not None and param.shape == src_param.shape:
                            param.data.copy_(src_param.data)
            
            print("Encoder initialized from pretrained MAE")
        except Exception as e:
            print(f"Warning: Could not fully initialize from MAE: {e}")
            print("Using default initialization")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] - input images
        Returns:
            logits: [B, num_classes]
        """
        return self.encoder(x)
    
    def extract_features(self, x):
        """Extract features without classification"""
        # Forward through all stages except head
        x = self.encoder.stem(x)
        x = self.encoder.stage1(x)
        x = self.encoder.down1(x)
        x = self.encoder.stage2(x)
        x = self.encoder.down2(x)
        
        # Convert to sequence for attention stages
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder.stage3(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        x = self.encoder.down3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder.stage4(x)
        
        # Global pooling
        x = x.mean(dim=1)
        x = self.encoder.norm(x)
        
        return x