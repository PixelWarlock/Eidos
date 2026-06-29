import os
import argparse
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def load_mask(mask_path: str) -> np.ndarray:
    mask = Image.open(mask_path)
    mask = np.array(mask)

    if mask.ndim == 3:
        raise ValueError(
            f"Expected single-channel class-id mask, got shape {mask.shape}. "
            "Do not pass the RGB visualization mask."
        )

    return mask.astype(np.int64)


def resize_mask_albumentations(
    mask: np.ndarray,
    height: int = 256,
    width: int = 256,
) -> np.ndarray:
    """
    Resizes segmentation mask with nearest-neighbor interpolation.
    Class ids are preserved.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected [H, W] mask, got {mask.shape}")

    dummy_image = np.zeros((*mask.shape, 3), dtype=np.uint8)

    try:
        transform = A.Compose([
            A.Resize(
                height=height,
                width=width,
                interpolation=cv2.INTER_NEAREST,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            )
        ])
    except TypeError:
        transform = A.Compose([
            A.Resize(
                height=height,
                width=width,
                interpolation=cv2.INTER_NEAREST,
                p=1.0,
            )
        ])

    out = transform(image=dummy_image, mask=mask)
    return out["mask"].astype(mask.dtype)


def patchify_mask(mask: np.ndarray, patch_size: int):
    h, w = mask.shape

    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size

    cropped_mask = mask[:h_crop, :w_crop]

    if h_crop != h or w_crop != w:
        print(
            f"[INFO] Cropped mask from {(h, w)} to {(h_crop, w_crop)} "
            f"so it is divisible by patch_size={patch_size}."
        )

    n_rows = h_crop // patch_size
    n_cols = w_crop // patch_size

    patches = cropped_mask.reshape(
        n_rows,
        patch_size,
        n_cols,
        patch_size,
    ).transpose(0, 2, 1, 3)

    return cropped_mask, patches, n_rows, n_cols


def patch_class_distributions(patches: np.ndarray):
    n_rows, n_cols, ph, pw = patches.shape

    flat_patches = patches.reshape(n_rows * n_cols, ph * pw)

    classes = np.unique(flat_patches)
    dist = np.zeros((flat_patches.shape[0], len(classes)), dtype=np.float32)

    for class_idx, cls in enumerate(classes):
        dist[:, class_idx] = (flat_patches == cls).mean(axis=1)

    return dist, classes


def create_ideal_gram(dist: np.ndarray) -> np.ndarray:
    """
    Ideal cosine-style Gram target in [-1, 1].

    overlap(i, j) = sum_c min(dist_i[c], dist_j[c])
    gram(i, j) = 2 * overlap(i, j) - 1
    """
    overlap = np.minimum(
        dist[:, None, :],
        dist[None, :, :],
    ).sum(axis=-1)

    gram = 2.0 * overlap - 1.0

    return gram


def make_spaced_patch_grid(
    mask: np.ndarray,
    patch_size: int,
    gap: int = 3,
    gap_value: float = np.nan,
):
    """
    Creates visual-only patch grid with spacing between patches.
    Gram computation still uses the real contiguous mask.
    """
    h, w = mask.shape

    n_rows = h // patch_size
    n_cols = w // patch_size

    spaced_h = n_rows * patch_size + (n_rows - 1) * gap
    spaced_w = n_cols * patch_size + (n_cols - 1) * gap

    spaced = np.full((spaced_h, spaced_w), gap_value, dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            src_y0 = r * patch_size
            src_y1 = src_y0 + patch_size
            src_x0 = c * patch_size
            src_x1 = src_x0 + patch_size

            dst_y0 = r * (patch_size + gap)
            dst_y1 = dst_y0 + patch_size
            dst_x0 = c * (patch_size + gap)
            dst_x1 = dst_x0 + patch_size

            spaced[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]

    return spaced


def patch_index_from_row_col(row_0based: int, col_0based: int, n_cols: int) -> int:
    return row_0based * n_cols + col_0based


def patch_row_col_from_xy_spaced(
    x,
    y,
    patch_size: int,
    gap: int,
    n_rows: int,
    n_cols: int,
):
    if x is None or y is None:
        return None

    cell = patch_size + gap

    col = int(x // cell)
    row = int(y // cell)

    if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
        return None

    local_x = x - col * cell
    local_y = y - row * cell

    if local_x >= patch_size or local_y >= patch_size:
        return None

    return row, col


def describe_patch_distribution(
    dist: np.ndarray,
    classes: np.ndarray,
    patch_idx: int,
) -> str:
    nonzero = dist[patch_idx] > 0

    parts = []
    for cls, frac in zip(classes[nonzero], dist[patch_idx][nonzero]):
        parts.append(f"class {int(cls)}: {float(frac):.3f}")

    return ", ".join(parts)


def visualize_interactive_gram_from_mask(
    mask: np.ndarray,
    patch_size: int = 16,
    cmap_mask: str = "gray",
    cmap_similarity: str = "coolwarm",
    show_text: bool = True,
    text_fontsize: int = 6,
    similarity_alpha: float = 0.65,
    grid_gap: int = 3,
):
    cropped_mask, patches, n_rows, n_cols = patchify_mask(mask, patch_size)
    spaced_mask = make_spaced_patch_grid(
        cropped_mask,
        patch_size=patch_size,
        gap=grid_gap,
    )

    dist, classes = patch_class_distributions(patches)
    gram = create_ideal_gram(dist)

    num_patches = n_rows * n_cols

    print("[INFO] Cropped mask shape:", cropped_mask.shape)
    print("[INFO] Patch grid:", n_rows, "x", n_cols)
    print("[INFO] Number of patches:", num_patches)
    print("[INFO] Classes in cropped mask:", classes.tolist())
    print("[INFO] Gram shape:", gram.shape)
    print("[INFO] Gram min/max:", float(gram.min()), float(gram.max()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax_original = axes[0]
    ax_patchified = axes[1]

    ax_original.imshow(cropped_mask, cmap=cmap_mask, interpolation="nearest")
    ax_original.set_title("Original single-channel mask")
    ax_original.axis("off")

    spaced_cmap = plt.get_cmap(cmap_mask).copy()
    spaced_cmap.set_bad(color="white")

    ax_patchified.imshow(spaced_mask, cmap=spaced_cmap, interpolation="nearest")
    ax_patchified.set_title("Patchified mask — hover over a patch")
    ax_patchified.axis("off")

    rects = []
    cell = patch_size + grid_gap

    for r in range(n_rows):
        for c in range(n_cols):
            rect = Rectangle(
                (c * cell, r * cell),
                patch_size,
                patch_size,
            )
            rects.append(rect)

    similarity_collection = PatchCollection(
        rects,
        cmap=cmap_similarity,
        alpha=similarity_alpha,
        edgecolor="white",
        linewidth=0.6,
    )
    similarity_collection.set_clim(-1.0, 1.0)
    similarity_collection.set_array(np.zeros(num_patches))
    similarity_collection.set_visible(False)

    ax_patchified.add_collection(similarity_collection)

    selected_rect = Rectangle(
        (0, 0),
        patch_size,
        patch_size,
        linewidth=3.0,
        edgecolor="yellow",
        facecolor="none",
        visible=False,
    )
    ax_patchified.add_patch(selected_rect)

    sim_texts = []

    for r in range(n_rows):
        for c in range(n_cols):
            x = c * cell
            y = r * cell
            x_center = x + patch_size / 2

            if show_text:
                sim_text = ax_patchified.text(
                    x_center,
                    y + patch_size * 0.28,
                    "",
                    color="white",
                    fontsize=text_fontsize,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1),
                )
                sim_texts.append(sim_text)

            ax_patchified.text(
                x_center,
                y + patch_size * 0.72,
                f"({r + 1},{c + 1})",
                color="white",
                fontsize=text_fontsize,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.45, pad=1),
            )

    info = fig.text(
        0.5,
        0.02,
        "Hover over a patch on the right.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    cbar = fig.colorbar(
        similarity_collection,
        ax=ax_patchified,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Similarity to hovered patch")

    def update_hover(event):
        if event.inaxes != ax_patchified:
            return

        rc = patch_row_col_from_xy_spaced(
            event.xdata,
            event.ydata,
            patch_size,
            grid_gap,
            n_rows,
            n_cols,
        )

        if rc is None:
            return

        row, col = rc
        selected_idx = patch_index_from_row_col(row, col, n_cols)

        similarities = gram[selected_idx]

        similarity_collection.set_array(similarities)
        similarity_collection.set_visible(True)

        selected_rect.set_xy((col * cell, row * cell))
        selected_rect.set_visible(True)

        if show_text:
            for idx, sim_text in enumerate(sim_texts):
                sim_text.set_text(f"{similarities[idx]:.2f}")

        selected_id = f"({row + 1},{col + 1})"
        selected_dist = describe_patch_distribution(dist, classes, selected_idx)

        info.set_text(
            f"Selected patch {selected_id} | "
            f"matrix index {selected_idx} | "
            f"{selected_dist}"
        )

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", update_hover)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

    return {
        "mask": cropped_mask,
        "patches": patches,
        "dist": dist,
        "classes": classes,
        "gram": gram,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


def save_npy(path: str, array: np.ndarray):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    np.save(path, array)


def main(mask_path):
    parser = argparse.ArgumentParser()

    parser.add_argument("--patch_size", type=int, default=16)

    parser.add_argument(
        "--resize_size",
        type=int,
        default=256,
        help="Resize mask to resize_size x resize_size. Use 0 to disable.",
    )

    parser.add_argument("--grid_gap", type=int, default=3)
    parser.add_argument("--cmap_mask", type=str, default="gray")
    parser.add_argument("--cmap_similarity", type=str, default="coolwarm")
    parser.add_argument("--similarity_alpha", type=float, default=0.65)
    parser.add_argument("--text_fontsize", type=int, default=6)

    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Disable similarity values. Patch ids are still shown.",
    )

    parser.add_argument("--save_gram", type=str, default=None)
    parser.add_argument("--save_dist", type=str, default=None)

    args = parser.parse_args()

    mask = load_mask(mask_path)

    if args.resize_size != 0:
        mask = resize_mask_albumentations(
            mask,
            height=args.resize_size,
            width=args.resize_size,
        )

        print(f"[INFO] Resized mask to: {mask.shape}")
        print("[INFO] Unique values after resize:", np.unique(mask).tolist())

    out = visualize_interactive_gram_from_mask(
        mask=mask,
        patch_size=args.patch_size,
        cmap_mask=args.cmap_mask,
        cmap_similarity=args.cmap_similarity,
        show_text=not args.no_text,
        text_fontsize=args.text_fontsize,
        similarity_alpha=args.similarity_alpha,
        grid_gap=args.grid_gap,
    )

    if args.save_gram is not None:
        save_npy(args.save_gram, out["gram"])
        print(f"[INFO] Saved Gram matrix to: {args.save_gram}")

    if args.save_dist is not None:
        save_npy(args.save_dist, out["dist"])
        print(f"[INFO] Saved patch distributions to: {args.save_dist}")


if __name__ == "__main__":
    FILENAME = '000000001000.png'
    IMG_PATH = f'/media/darnok/Nowy/X/datasets/coco-stuff/val2017/val2017/{FILENAME}'
    MASK_PATH = f'/media/darnok/Nowy/X/datasets/coco-stuff/annotations/val2017/{FILENAME}'
    main(mask_path=MASK_PATH)