import torch
import torch.nn as nn

from eidos.modules.patch_embedding import PatchEmbedding
from eidos.modules.transformer_block import TransformerBlock


class SpectralViT(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        dim=256,
        depth=6,
        heads=8
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(
            in_channels, dim, patch_size
        )

        num_patches = (image_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, dim) * 0.02
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        x = self.patch_embed(x)     # [B,N,D]

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x  # [B,N,D]