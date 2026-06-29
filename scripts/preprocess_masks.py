import os
import json
import numpy as np
import colorsys

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "pycocotools is required for decoding COCO polygon/RLE masks. "
        "Install it with: pip install pycocotools"
    ) from e

UNKNOWN_VALUE = 255
BACKGROUND_VALUE = 0

def load_img(filepath: str) -> Image.Image:
    """
    Loads an image from disk.

    Returns:
        PIL image in RGB format.
    """
    return Image.open(filepath).convert("RGB")

def load_annotations(annotation_file: str):
    """
    Loads a COCO-style annotation JSON file and builds useful lookup maps.

    Expected COCO keys:
        - images
        - annotations
        - categories

    Returns:
        coco: full annotation dictionary
        filename_to_image: mapping from image filename to image metadata
        image_id_to_annotations: mapping from image_id to list of annotations
        valid_class_ids: set of valid category ids
    """
    with open(annotation_file, "r") as f:
        coco = json.load(f)

    filename_to_image = {
        img["file_name"]: img
        for img in coco["images"]
    }

    image_id_to_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        image_id_to_annotations[ann["image_id"]].append(ann)

    valid_class_ids = {
        cat["id"]
        for cat in coco.get("categories", [])
    }

    return coco, filename_to_image, image_id_to_annotations, valid_class_ids

def _decode_segmentation(annotation: dict, height: int, width: int) -> np.ndarray:
    """
    Decodes a single COCO segmentation annotation into a binary mask.

    Supports:
        - polygon segmentation
        - uncompressed RLE
        - compressed RLE
    """
    segmentation = annotation["segmentation"]

    if isinstance(segmentation, list):
        # Polygon format
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)

    elif isinstance(segmentation, dict):
        # RLE format, compressed or uncompressed
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation

        mask = mask_utils.decode(rle)

    else:
        raise ValueError(
            f"Unsupported segmentation format: {type(segmentation)}"
        )

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)

    return mask.astype(bool)

def get_mask(
    image_info: dict,
    image_annotations: list,
    unknown_value: int = UNKNOWN_VALUE,
) -> np.ndarray:
    """
    Builds a semantic segmentation mask for one image.

    Pixels start as UNKNOWN_VALUE.
    Annotated pixels receive their COCO category_id.
    Later fix_mask() converts UNKNOWN_VALUE and invalid pixels to background 0.

    Args:
        image_info: COCO image metadata.
        image_annotations: annotations belonging to that image.
        unknown_value: temporary value for unlabeled pixels.

    Returns:
        H x W uint16 semantic mask.
    """
    height = image_info["height"]
    width = image_info["width"]

    mask = np.full(
        shape=(height, width),
        fill_value=unknown_value,
        dtype=np.uint16,
    )

    # Optional: draw larger regions first, smaller objects later.
    # This helps smaller objects overwrite stuff/background regions.
    image_annotations = sorted(
        image_annotations,
        key=lambda ann: ann.get("area", 0),
        reverse=True,
    )

    for ann in image_annotations:
        if ann.get("iscrowd", 0) == 1:
            # You can remove this if you want crowd regions included.
            continue

        category_id = ann["category_id"]
        binary_mask = _decode_segmentation(ann, height, width)

        mask[binary_mask] = category_id

    return mask

'''
def check_mask(
    mask: np.ndarray,
    valid_class_ids: set,
    background_value: int = BACKGROUND_VALUE,
    unknown_value: int = UNKNOWN_VALUE,
) -> bool:
    """
    Checks whether all pixels belong to background or a valid class.

    Returns:
        True if mask is clean.
        False if there are unknown / invalid pixels.
    """
    valid_values = set(valid_class_ids)
    valid_values.add(background_value)

    # unknown_value is intentionally NOT valid here.
    # It marks pixels not covered by any COCO annotation.
    unique_values = set(np.unique(mask).tolist())

    return unique_values.issubset(valid_values)

def fix_mask(
    mask: np.ndarray,
    valid_class_ids: set,
    background_value: int = BACKGROUND_VALUE,
) -> np.ndarray:
    """
    Fixes a semantic segmentation mask.

    Any pixel whose value is not one of the known category ids
    is set to background_value.

    This catches:
        - unlabeled regions
        - boundary / void values
        - unknown values, e.g. 255
        - corrupt / unexpected class ids
    """
    valid_values = np.array(
        sorted(list(valid_class_ids) + [background_value]),
        dtype=mask.dtype,
    )

    valid_pixel_mask = np.isin(mask, valid_values)

    fixed = mask.copy()
    fixed[~valid_pixel_mask] = background_value

    return fixed
'''

def check_mask(
    mask: np.ndarray,
    valid_class_ids: set,
    background_value: int = 0,
    include_background: bool = False,
) -> bool:
    """
    Checks whether every pixel belongs to a valid class.

    By default, background 0 is NOT treated as valid, because you now want
    unlabeled/background-like pixels to be absorbed by the nearest mask.
    """
    valid_values = set(valid_class_ids)

    if include_background:
        valid_values.add(background_value)

    unique_values = set(np.unique(mask).tolist())

    return unique_values.issubset(valid_values)

def fix_mask(
    mask: np.ndarray,
    valid_class_ids: set,
    background_value: int = 0,
    include_background_as_source: bool = False,
) -> np.ndarray:
    """
    Fills unlabeled / invalid pixels with the nearest valid labeled pixel.

    This no longer converts unknown pixels to background.

    Example:
        255 pixels between two objects are assigned to whichever valid class
        is spatially closest.

    Args:
        mask:
            H x W class-id mask.
        valid_class_ids:
            Valid COCO category ids.
        background_value:
            Usually 0.
        include_background_as_source:
            If False, background 0 will also be replaced by the nearest object/stuff class.
            If True, existing background 0 can expand into unknown pixels too.

    Returns:
        fixed mask with invalid/unlabeled pixels filled from nearest valid regions.
    """
    source_values = set(valid_class_ids)

    if include_background_as_source:
        source_values.add(background_value)

    source_values = np.array(sorted(source_values), dtype=mask.dtype)

    # Pixels that are valid sources for nearest-neighbor filling.
    source_mask = np.isin(mask, source_values)

    # Nothing to fix.
    if source_mask.all():
        return mask

    # If there are no valid labels at all, fallback to background.
    # This should only happen for images with no usable annotations.
    if not source_mask.any():
        fixed = np.full_like(mask, fill_value=background_value)
        return fixed

    # distance_transform_edt computes nearest zero location.
    # So we pass ~source_mask:
    #   source pixels = False / 0
    #   invalid pixels = True / 1
    _, nearest_indices = distance_transform_edt(
        ~source_mask,
        return_indices=True,
    )

    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    fixed = mask.copy()

    invalid_mask = ~source_mask
    fixed[invalid_mask] = mask[
        nearest_y[invalid_mask],
        nearest_x[invalid_mask],
    ]

    return fixed

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb / 255.0
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Approximate RGB -> CIELAB conversion.
    rgb: [N, 3], values 0-255
    """
    rgb_linear = _srgb_to_linear(rgb.astype(np.float32))

    # sRGB D65 conversion
    xyz = rgb_linear @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T

    # D65 reference white
    xyz /= np.array([0.95047, 1.00000, 1.08883])

    eps = 216 / 24389
    kappa = 24389 / 27

    f = np.where(
        xyz > eps,
        np.cbrt(xyz),
        (kappa * xyz + 16) / 116,
    )

    lab = np.empty_like(f)
    lab[:, 0] = 116 * f[:, 1] - 16
    lab[:, 1] = 500 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200 * (f[:, 1] - f[:, 2])

    return lab


def _generate_distinct_colors(
    num_colors: int,
    seed: int = 42,
    candidate_hues: int = 720,
) -> np.ndarray:
    """
    Generates visually distinct RGB colors using greedy farthest-point sampling
    in CIELAB space.

    Returns:
        colors: [num_colors, 3] uint8
    """
    if num_colors <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    rng = np.random.default_rng(seed)

    # Candidate color pool.
    # Avoid very dark colors because background is black.
    # Avoid very low saturation because gray-ish colors are hard to distinguish.
    hue_offset = rng.random()
    hues = (np.linspace(0, 1, candidate_hues, endpoint=False) + hue_offset) % 1.0

    saturations = np.array([0.60, 0.75, 0.90, 1.00])
    values = np.array([0.70, 0.82, 0.94, 1.00])

    candidates = []

    for h in hues:
        for s in saturations:
            for v in values:
                r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
                candidates.append([r * 255, g * 255, b * 255])

    candidates = np.array(candidates, dtype=np.float32)
    candidates = np.unique(candidates.astype(np.uint8), axis=0)

    candidate_lab = _rgb_to_lab(candidates)

    # Treat black as already selected so no class color is close to background.
    black_lab = _rgb_to_lab(np.array([[0, 0, 0]], dtype=np.uint8))

    selected_colors = []
    selected_lab = black_lab.copy()

    # Current minimum distance from each candidate to any selected color.
    min_dist = np.linalg.norm(
        candidate_lab[:, None, :] - selected_lab[None, :, :],
        axis=-1,
    ).min(axis=1)

    for _ in range(num_colors):
        idx = int(np.argmax(min_dist))

        selected_colors.append(candidates[idx])

        new_lab = candidate_lab[idx:idx + 1]
        dist_to_new = np.linalg.norm(candidate_lab - new_lab, axis=1)

        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[idx] = -1.0  # do not select again

    return np.array(selected_colors, dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """
    Converts a single-channel class-id mask into an RGB visualization.

    Uses a deterministic, high-contrast, perceptually separated color palette.

    This is only for visualization. Do not use RGB masks for CrossEntropyLoss
    unless your dataset class converts them back to class ids.
    """
    mask = mask.astype(np.int64)

    if mask.ndim != 2:
        raise ValueError(f"Expected single-channel mask [H, W], got {mask.shape}")

    max_class_id = int(mask.max())

    # Class 0 stays black/background.
    palette = np.zeros((max_class_id + 1, 3), dtype=np.uint8)

    if max_class_id > 0:
        palette[1:] = _generate_distinct_colors(
            num_colors=max_class_id,
            seed=42,
        )

    color_mask = palette[mask]
    return Image.fromarray(color_mask, mode="RGB")

def save_color_mask(mask: np.ndarray, save_path: str):
    """
    Saves an RGB colorized version of the mask for debugging/visualization.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    color_mask = colorize_mask(mask)
    color_mask.save(save_path)

def save_mask(mask: np.ndarray, save_path: str):
    """
    Saves a segmentation mask as PNG.

    Uses uint16 PNG if category ids exceed uint8 range.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if mask.max() <= 255:
        mask = mask.astype(np.uint8)
    else:
        mask = mask.astype(np.uint16)

    Image.fromarray(mask).save(save_path)

def main(data_dir: str, annotation_file: str, dest_dir: str):
    coco, filename_to_image, image_id_to_annotations, valid_class_ids = load_annotations(
        annotation_file
    )

    filepaths = [
        os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    os.makedirs(dest_dir, exist_ok=True)

    fixed_count = 0
    skipped_count = 0

    for filepath in tqdm(filepaths, desc="Preparing masks"):
        filename = os.path.basename(filepath)

        if filename not in filename_to_image:
            skipped_count += 1
            continue

        image = load_img(filepath)
        image_info = filename_to_image[filename]
        #print(image_info['id'])

        # Safety check in case JSON dimensions differ from actual image size.
        img_width, img_height = image.size
        if image_info["width"] != img_width or image_info["height"] != img_height:
            image_info = dict(image_info)
            image_info["width"] = img_width
            image_info["height"] = img_height

        image_id = image_info["id"]
        anns = image_id_to_annotations.get(image_id, [])

        mask = get_mask(image_info, anns)

        if not check_mask(mask, valid_class_ids):
            mask = fix_mask(mask, valid_class_ids)
            fixed_count += 1

        output_name = os.path.splitext(filename)[0] + ".jpg"
        save_path = os.path.join(dest_dir, output_name)

        #unique_classes = np.unique(mask)

        #print(
        #    f"{filename}: {len(unique_classes)} distinct classes/pixel values -> "
        #    f"{unique_classes.tolist()}"
        #)
        #save_mask(mask, save_path)
        save_color_mask(mask=mask, save_path=save_path)
        

    print(f"Done.")
    print(f"Saved masks to: {dest_dir}")
    print(f"Fixed masks: {fixed_count}")
    print(f"Skipped images without annotations in JSON: {skipped_count}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='/media/darnok/Nowy/X/datasets/coco-stuff/val2017/val2017/')
    parser.add_argument('--annotation_file', default='/media/darnok/Nowy/X/datasets/coco-stuff/annotations/stuff_val2017.json')
    parser.add_argument('--dest_dir', default='/media/darnok/Nowy/X/datasets/coco-stuff/val2017_masks_colored')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        dest_dir=args.dest_dir,
    )