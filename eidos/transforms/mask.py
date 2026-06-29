import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Sequence, Dict, Any


class ToMaskTensor:
    """
    Converts PIL / NumPy / Tensor mask into torch.LongTensor [H, W].

    This expects a single-channel class-id mask.
    """

    def __call__(self, mask) -> torch.Tensor:
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        # [H, W, 1] -> [H, W]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        # [1, H, W] -> [H, W]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        if mask.ndim != 2:
            raise ValueError(
                f"Expected single-channel mask [H, W], got shape {tuple(mask.shape)}"
            )

        return mask.long()


class MaskPatchify:
    """
    Splits a class-id mask into non-overlapping patches.

    Input:
        mask: [H, W]

    Output:
        patches: [N, patch_size, patch_size]

    Patch order is row-major:
        (1,1), (1,2), ..., (2,1), ...
    """

    def __init__(
        self,
        patch_size: int = 16,
        crop_if_needed: bool = True,
    ):
        self.patch_size = patch_size
        self.crop_if_needed = crop_if_needed

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [H, W], got {tuple(mask.shape)}")

        p = self.patch_size
        h, w = mask.shape

        h_crop = (h // p) * p
        w_crop = (w // p) * p

        if h_crop != h or w_crop != w:
            if not self.crop_if_needed:
                raise ValueError(
                    f"Mask shape {(h, w)} is not divisible by patch_size={p}."
                )

            mask = mask[:h_crop, :w_crop]

        h, w = mask.shape
        n_rows = h // p
        n_cols = w // p

        patches = mask.unfold(0, p, p).unfold(1, p, p)
        # [n_rows, n_cols, p, p]

        patches = patches.contiguous().view(n_rows * n_cols, p, p)
        # [N, p, p]

        return patches


class PatchClassDistribution:
    """
    Converts mask patches into per-patch class distributions.

    Example:
        pure class 5 patch:
            class 5 -> 1.0

        50% class 5 / 50% class 7 patch:
            class 5 -> 0.5
            class 7 -> 0.5

    Input:
        patches: [N, P, P]

    Output:
        dist: [N, C]
        classes: [C]
    """

    def __init__(
        self,
        class_ids: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Args:
            class_ids:
                Explicit class ids to use as columns in the distribution.
                Useful if your COCO ids are sparse or non-contiguous.

            num_classes:
                Use classes [0, 1, ..., num_classes - 1].
                Fast path for contiguous labels.

            validate:
                If True, checks that every patch distribution sums to 1.
        """
        if class_ids is not None and num_classes is not None:
            raise ValueError("Use either class_ids or num_classes, not both.")

        self.class_ids = class_ids
        self.num_classes = num_classes
        self.validate = validate

    def __call__(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        if patches.ndim != 3:
            raise ValueError(f"Expected patches [N, P, P], got {tuple(patches.shape)}")

        patches = patches.long()
        device = patches.device

        n, p1, p2 = patches.shape
        flat = patches.view(n, p1 * p2)

        if self.num_classes is not None:
            if flat.min() < 0 or flat.max() >= self.num_classes:
                raise ValueError(
                    f"Mask contains class ids outside [0, {self.num_classes - 1}]. "
                    f"Found min={int(flat.min())}, max={int(flat.max())}."
                )

            one_hot = F.one_hot(flat, num_classes=self.num_classes).float()
            dist = one_hot.mean(dim=1)
            classes = torch.arange(self.num_classes, device=device)

        else:
            if self.class_ids is None:
                classes = torch.unique(flat)
                classes = torch.sort(classes).values
            else:
                classes = torch.as_tensor(self.class_ids, device=device, dtype=torch.long)

            # [N, P*P, 1] == [1, 1, C] -> [N, P*P, C]
            matches = flat[..., None] == classes.view(1, 1, -1)
            dist = matches.float().mean(dim=1)

        if self.validate:
            sums = dist.sum(dim=-1)

            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
                bad = torch.nonzero(~torch.isclose(sums, torch.ones_like(sums), atol=1e-5))
                raise ValueError(
                    "Some patch distributions do not sum to 1. "
                    "This usually means the mask contains class ids not included in class_ids. "
                    f"Bad patch indices: {bad.flatten()[:10].tolist()}"
                )

        return {
            "dist": dist,
            "classes": classes,
        }


class CreateGram:
    """
    Creates ideal cosine-style Gram matrix from patch class distributions.

    Input:
        dist: [N, C]

    Output:
        gram: [N, N]

    Formula:
        overlap(i, j) = sum_c min(dist_i[c], dist_j[c])
        gram(i, j) = 2 * overlap(i, j) - 1

    Meaning:
        same semantic content        ->  1
        completely different content -> -1
        partial overlap              -> between -1 and 1
    """

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.ndim != 2:
            raise ValueError(f"Expected dist [N, C], got {tuple(dist.shape)}")

        overlap = torch.minimum(
            dist[:, None, :],
            dist[None, :, :],
        ).sum(dim=-1)

        gram = 2.0 * overlap - 1.0

        return gram


class MaskToGram:
    """
    Full mask-only transform:

        mask -> tensor -> patches -> class distributions -> ideal Gram

    This is the class you probably want to call inside your dataset.
    """

    def __init__(
        self,
        patch_size: int = 16,
        class_ids: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
        crop_if_needed: bool = True,
        return_patches: bool = False,
        return_distribution: bool = False,
    ):
        self.to_tensor = ToMaskTensor()
        self.patchify = MaskPatchify(
            patch_size=patch_size,
            crop_if_needed=crop_if_needed,
        )
        self.to_distribution = PatchClassDistribution(
            class_ids=class_ids,
            num_classes=num_classes,
            validate=True,
        )
        self.create_gram = CreateGram()

        self.return_patches = return_patches
        self.return_distribution = return_distribution

    def __call__(self, mask) -> Dict[str, Any]:
        mask = self.to_tensor(mask)
        patches = self.patchify(mask)

        dist_out = self.to_distribution(patches)
        dist = dist_out["dist"]
        classes = dist_out["classes"]

        gram = self.create_gram(dist)

        out = {
            "gram": gram.float(),
            "classes": classes,
        }

        if self.return_patches:
            out["patches"] = patches

        if self.return_distribution:
            out["dist"] = dist

        return out