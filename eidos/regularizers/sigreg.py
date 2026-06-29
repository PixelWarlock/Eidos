from __future__ import annotations

import torch
from torch import nn


def _flatten_embeddings(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("features must contain a sample and feature dimension")
    return features.reshape(-1, features.shape[-1]).float()


class SIGReg(nn.Module):
    """Strong SIGReg using empirical characteristic-function matching."""

    def __init__(
        self,
        sketch_dim: int = 64,
        num_points: int = 17,
        integration_limit: float = 5.0,
    ) -> None:
        super().__init__()
        if sketch_dim <= 0 or num_points < 2:
            raise ValueError("sketch_dim must be positive and num_points must be at least two")
        self.sketch_dim = sketch_dim
        self.num_points = num_points
        self.integration_limit = integration_limit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = _flatten_embeddings(features)
        num_samples, channels = embeddings.shape
        projection = torch.randn(
            channels,
            self.sketch_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-6)
        integration_points = torch.linspace(
            -self.integration_limit,
            self.integration_limit,
            self.num_points,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        gaussian_cf = torch.exp(-0.5 * integration_points.square())
        projected = embeddings @ projection
        arguments = projected.unsqueeze(-1) * integration_points
        empirical_cf = torch.exp(1j * arguments).mean(dim=0)
        error = (empirical_cf - gaussian_cf).abs().square() * gaussian_cf
        integrated = torch.trapz(error, integration_points, dim=-1)
        return (integrated * num_samples).mean()


class WeakSIGReg(nn.Module):
    """Weak SIGReg matching a randomly sketched covariance to identity."""

    def __init__(self, sketch_dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        if sketch_dim <= 0:
            raise ValueError("sketch_dim must be positive")
        self.sketch_dim = sketch_dim
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = _flatten_embeddings(features)
        num_samples, channels = embeddings.shape
        if channels > self.sketch_dim:
            sketch = torch.randn(
                self.sketch_dim,
                channels,
                device=embeddings.device,
                dtype=embeddings.dtype,
            ) / (channels**0.5)
            embeddings = embeddings @ sketch.transpose(0, 1)

        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        covariance = embeddings.transpose(0, 1) @ embeddings / (num_samples - 1 + self.eps)
        target = torch.eye(
            covariance.shape[0], device=covariance.device, dtype=covariance.dtype
        )
        return torch.linalg.matrix_norm(covariance - target, ord="fro")

