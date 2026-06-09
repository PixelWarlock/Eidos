import torch
import torch.nn as nn


class PatchToToken(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, dim=256):
        super().__init__()

        self.proj = nn.Linear(in_channels * patch_size * patch_size, dim)

    def forward(self, x):
        # x: [B,N,C,P,P]

        B, N, C, P, _ = x.shape
        x = x.reshape(B, N, C * P * P)

        return self.proj(x)