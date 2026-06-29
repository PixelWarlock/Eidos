from __future__ import annotations

import numpy as np
import torch

class BaseTransform:
    """Convert an RGB uint8 image to CHW float32 in [0, 1]."""

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape [H, W, 3]")
        array = np.ascontiguousarray(image)
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
