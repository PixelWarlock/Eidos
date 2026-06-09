import torch
import torch.nn as nn


class XSA_MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)  # [B,H,N,d]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        y = attn @ v  # [B,H,N,d]

        # -----------------------
        # XSA projection removal
        # -----------------------

        y = y.permute(0, 2, 1, 3)      # [B,N,H,d]
        v_self = v.permute(0, 2, 1, 3)  # [B,N,H,d]

        v_norm = v_self / (v_self.norm(dim=-1, keepdim=True) + self.eps)

        proj = (y * v_norm).sum(dim=-1, keepdim=True) * v_norm

        z = y - proj

        z = z.permute(0, 2, 1, 3)      # [B,H,N,d]
        z = z.transpose(1, 2).reshape(B, N, D)

        return self.proj(z)