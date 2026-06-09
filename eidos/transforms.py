import math
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class LoadImage:
    def __call__(self, path):
        image = Image.open(path).convert("RGB")
        return image

class ToNumpy:
    def __call__(self, x):
        return np.array(x)
    
class Patchify:
    def __init__(self, patch_size, image_size=(256,256), radius=1):
        self.patch_size = patch_size
        self.radius = radius

        h, w = image_size
        gh, gw = h // patch_size, w // patch_size

        self.grid_shape = (gh, gw)

        self.neighbors = self.build_neighbors(gh, gw)

        self.coords = [
            (i // gw, i % gw)
            for i in range(gh * gw)
        ]

    def build_neighbors(self, gh, gw):

        neighbors = []

        for r in range(gh):
            for c in range(gw):

                idx = r * gw + c
                nbs = []

                for dr in range(-self.radius, self.radius + 1):
                    for dc in range(-self.radius, self.radius + 1):

                        if dr == 0 and dc == 0:
                            continue

                        rr = r + dr
                        cc = c + dc

                        if 0 <= rr < gh and 0 <= cc < gw:
                            nbs.append(rr * gw + cc)

                neighbors.append(nbs)

        return neighbors

    def __call__(self, x):

        c, h, w = x.shape
        p = self.patch_size

        patches = F.unfold(
            x,
            kernel_size=p,
            stride=p
        ).T.reshape(-1, c, p, p)

        return {
            "patches": patches,
            "neighbors": self.neighbors,
            "coords": self.coords,
            "grid_shape": self.grid_shape,
        }
    
class Show:
    def __init__(self, figsize=(6, 6), cmap=None):
        self.figsize = figsize
        self.cmap = cmap

    def __call__(self, patches: torch.Tensor, grid_shape=None):
        """
        patches: [N, C, P, P]
        grid_shape: optional (gh, gw)
        """

        assert patches.dim() == 4, \
            f"Expected [N,C,P,P], got {patches.shape}"

        patches = patches.detach().cpu()

        n, c, p, _ = patches.shape

        # ----------------------------
        # infer grid shape if not given
        # ----------------------------
        if grid_shape is None:
            side = int(math.sqrt(n))
            assert side * side == n, \
                f"Cannot infer square grid from N={n}"
            gh, gw = side, side
        else:
            gh, gw = grid_shape
            assert gh * gw == n, \
                f"grid_shape {grid_shape} does not match N={n}"

        # ----------------------------
        # plotting
        # ----------------------------
        fig, axes = plt.subplots(
            gh, gw,
            figsize=self.figsize
        )

        # handle single row/col edge cases
        if gh == 1 and gw == 1:
            axes = [[axes]]
        elif gh == 1:
            axes = [axes]
        elif gw == 1:
            axes = [[ax] for ax in axes]

        idx = 0

        for i in range(gh):
            for j in range(gw):

                ax = axes[i][j]

                patch = patches[idx]

                # [C,P,P] → [P,P,C]
                img = patch.permute(1, 2, 0)

                # clamp for safety (ToTensor gives [0,1])
                img = img.clamp(0, 1)

                ax.imshow(img, cmap=self.cmap)
                ax.axis("off")

                idx += 1

        plt.tight_layout()
        plt.show()