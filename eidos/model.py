from __future__ import annotations

import torch
import lightning as L
import torch.nn.functional as F

from torch import nn
from torch.optim import AdamW
from eidos.regularizers import REGULARIZERS


class RoPE(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        frequencies = 1.0 / (
            10_000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tensor.shape[-2], device=tensor.device)
        angles = torch.outer(positions.float(), self.frequencies)
        cos = angles.cos().to(tensor.dtype)[None, None]
        sin = angles.sin().to(tensor.dtype)[None, None]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)


class ExclusiveAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.heads = max(1, dim // 64)
        if dim % self.heads or (dim // self.heads) % 2:
            raise ValueError("dim must produce an even attention head size")
        self.head_dim = dim // self.heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.rope = RoPE(self.head_dim)

    @staticmethod
    def exclude_value_direction(
        attended: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        direction = F.normalize(value, dim=-1)
        return attended - (attended * direction).sum(-1, keepdim=True) * direction

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = tensor.shape
        qkv = self.qkv(tensor).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(self.rope(query), self.rope(key), value)

        attended = self.exclude_value_direction(attended, value)
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, dim * 8)
        self.output = nn.Linear(dim * 4, dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(tensor).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = ExclusiveAttention(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor + self.attention(self.attention_norm(tensor))
        return tensor + self.mlp(self.mlp_norm(tensor))


class Eidos(nn.Module):
    """A small CLS-free ViT returning one feature per 16x16 patch."""

    def __init__(self, dim: int = 384, depth: int = 6, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, dim, patch_size, patch_size)
        self.blocks = nn.Sequential(*(Block(dim) for _ in range(depth)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if any(side % self.patch_size for side in images.shape[-2:]):
            raise ValueError("image dimensions must be divisible by patch_size")
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        return self.norm(self.blocks(patches))


class EidosModule(L.LightningModule):
    def __init__(
        self,
        model: Eidos,
        regularizers: dict[str, float],
        learning_rate: float,
    ) -> None:
        super().__init__()
        unknown = regularizers.keys() - REGULARIZERS.keys()
        if unknown or not regularizers:
            raise ValueError(f"invalid regularizers: {sorted(unknown)}")
        self.model = model
        self.losses = nn.ModuleDict({name: REGULARIZERS[name]() for name in regularizers})
        self.weights = regularizers
        self.learning_rate = learning_rate

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        features = self(batch["image"])
        total = features.sum() * 0.0
        for name, regularizer in self.losses.items():
            value = (
                regularizer(features, batch["target_gram"])
                if name == "gram"
                else regularizer(features)
            )
            total = total + self.weights[name] * value
            self.log(f"{stage}/{name}", value, on_epoch=True, batch_size=len(features))
        self.log(f"{stage}/loss", total, prog_bar=True, on_epoch=True, batch_size=len(features))
        return total

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate)