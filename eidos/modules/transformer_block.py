import torch
import torch.nn as nn

from eidos.modules.swiglu import SwiGLU
from eidos.modules.xsa_attention import XSA_MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = XSA_MultiHeadAttention(dim, heads)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x