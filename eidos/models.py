import torch
import torch.nn as nn

from eidos.modules.patch_to_token import PatchToToken
from eidos.modules.transformer_block import TransformerBlock


class ViT(nn.Module):
    def __init__(self, dim=128, depth=6, heads=8, patch_size=16, in_channels=3):
        super().__init__()

        self.patch_proj = PatchToToken(
            patch_size=patch_size,
            in_channels=in_channels,
            dim=dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B,N,D]
        x = self.patch_proj(x)  # [B,N,D]

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)