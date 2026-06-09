import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B,C,H,W]

        x = self.proj(x)              # [B,D,H/P,W/P]
        B, D, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # [B,N,D]

        return x