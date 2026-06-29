from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def feature_gram(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape [B, N, D]")
    normalized = F.normalize(features, p=2, dim=-1, eps=eps)
    return normalized @ normalized.transpose(-1, -2)


class GramReg(nn.Module):
    """MSE between feature cosine similarities and the mask target."""

    def forward(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        predicted = feature_gram(features)
        if predicted.shape != target.shape:
            raise ValueError("feature and target Gram matrices must have the same shape")
        diagonal = torch.eye(target.shape[-1], device=target.device, dtype=torch.bool)
        return (predicted - target).square().masked_select(~diagonal.unsqueeze(0)).mean()
