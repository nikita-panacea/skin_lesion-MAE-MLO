# models/separable_attention.py
"""
Separable Self-Attention - Optimized implementation
Based on MobileViT-v2, matching paper equations (5-10)
Complexity: O(k) instead of O(k²)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableSelfAttention(nn.Module):
    """
    Separable self-attention mechanism.
    
    Key innovation: Uses a single latent token instead of pairwise attention,
    reducing complexity from O(k²) to O(k).
    
    Equations from paper:
    (5) cs = softmax(x @ W_I)
    (6) cv = Σ cs(i) · (x @ W_K)(i)
    (7) x_V = ReLU(x @ W_V)
    (8) z = cv ⊙ x_V
    (9) y = z @ W_O
    """

    def __init__(self, dim):
        super().__init__()
        
        # Context score projection (W_I) - projects to scalar
        self.context_score = nn.Linear(dim, 1, bias=False)
        
        # Key, Value, Output projections
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] - batch, num_tokens, channels
        Returns:
            y: [B, N, C] - attended features
        """
        # Equation 5: Context scores
        scores = self.context_score(x)  # [B, N, 1]
        context_scores = F.softmax(scores, dim=1)  # [B, N, 1]
        
        # Equation 6: Context vector (weighted aggregation)
        keys = self.key_proj(x)  # [B, N, C]
        context_vector = torch.sum(context_scores * keys, dim=1)  # [B, C]
        
        # Equation 7: Value projection with ReLU
        values = F.relu(self.value_proj(x))  # [B, N, C]
        
        # Equation 8: Element-wise multiplication
        context_vector = context_vector.unsqueeze(1)  # [B, 1, C]
        z = context_vector * values  # [B, N, C]
        
        # Equation 9: Output projection
        output = self.out_proj(z)  # [B, N, C]
        
        return output


def count_parameters(dim):
    """Calculate parameter count for given dimension"""
    # W_I: dim × 1
    # W_K: dim × dim
    # W_V: dim × dim
    # W_O: dim × dim
    return dim + 3 * (dim * dim)


if __name__ == "__main__":
    print("Testing Separable Self-Attention")
    print("="*60)
    
    dims = [384, 768]
    
    for dim in dims:
        attn = SeparableSelfAttention(dim)
        params = sum(p.numel() for p in attn.parameters())
        expected = count_parameters(dim)
        
        print(f"\nDim={dim}:")
        print(f"  Parameters: {params:,}")
        print(f"  Expected: {expected:,}")
        print(f"  Match: {'✓' if params == expected else '✗'}")
        
        # Test forward
        x = torch.randn(2, 196, dim)
        y = attn(x)
        print(f"  Forward: {x.shape} -> {y.shape}")