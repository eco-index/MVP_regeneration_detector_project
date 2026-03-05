import pathlib
from typing import List, Tuple, Dict, Generator
from PIL import Image
import numpy as np
import datasets
import argparse

def find_image_pairs(input_dir_str: str) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """
    Finds pairs of raw images and their corresponding mask images in a directory.

    Args:
        input_dir_str: The path to the input directory.

    Returns:
        A list of tuples, where each tuple contains the Path objects for a
        raw image and its corresponding mask image.
    """
    input_dir = pathlib.Path(input_dir_str)
    image_pairs: List[Tuple[pathlib.Path, pathlib.Path]] = []

    for raw_image_path in input_dir.rglob("*_raw.png"):
        mask_filename = raw_image_path.name.replace("_raw.png", "_mask.png")
        mask_image_path = raw_image_path.with_name(mask_filename)

        if mask_image_path.exists():
            image_pairs.append((raw_image_path, mask_image_path))

    return image_pairs

def generate_binary_mask(mask_image_path: pathlib.Path) -> Image.Image:
    """
    Generates a binary mask from a mask image.

    Pixels close to pure pink (R > 240, G < 15, B > 240) are set to white (255),
    others are set to black (0).

    Args:
        mask_image_path: Path to the _mask.png file.

    Returns:
        A PIL Image object representing the binary mask (mode 'L').
    """
    try:
        img = Image.open(mask_image_path)
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_image_path}")
        raise
    except Exception as e:
        print(f"Error loading image {mask_image_path}: {e}")
        raise

    if img.mode != "RGB":
        img = img.convert("RGB")

    binary_mask = Image.new("L", img.size)
    img_pixels = img.load()
    binary_mask_pixels = binary_mask.load()

    for x in range(img.width):
        for y in range(img.height):
            r, g, b = img_pixels[x, y]
            # Check for pinkish color (adjust thresholds as needed)
            if r > 240 and g < 15 and b > 240:
                binary_mask_pixels[x, y] = 255  # White
            else:
                binary_mask_pixels[x, y] = 0    # Black
    
    return binary_mask

def extract_patches_from_pair(
    raw_image: Image.Image,
    binary_mask: Image.Image,
    patch_size: int,
) -> List[Dict[str, object]]:
    """Split an image/mask pair into square patches.

    Each returned dictionary contains the image patch, mask patch and the number
    of mask pixels (0-255 scale) in that patch. Filtering based on
    ``min_mask_fraction`` is performed after patches from all images have been
    gathered so that the threshold is evaluated relative to the entire dataset.
    """

    patches: List[Dict[str, object]] = []
    width, height = raw_image.size
    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            box = (left, top, left + patch_size, top + patch_size)
            image_patch = raw_image.crop(box)
            mask_patch = binary_mask.crop(box)
            mask_array = np.array(mask_patch, dtype=np.uint8)
            mask_pixel_sum = int(mask_array.sum())
            patches.append({"image": image_patch, "mask": mask_patch, "mask_pixels": mask_pixel_sum})
    return patches

def select_patches_by_dataset_fraction(
    patches: List[Dict[str, object]],
    patch_size: int,
    min_mask_fraction: float,
) -> List[Dict[str, Image.Image]]:
    """Filter patches so the overall mask coverage meets ``min_mask_fraction``.

    Negative (all-zero) patches are randomly subsampled if needed so that the
    ratio of mask pixels across the entire dataset is at least
    ``min_mask_fraction``. Positive patches are always kept.
    """

    import random

    if not patches:
        return []

    total_pixels = len(patches) * patch_size * patch_size * 255
    total_mask_pixels = sum(p["mask_pixels"] for p in patches)

    dataset_fraction = total_mask_pixels / total_pixels
    print(f"Initial dataset mask fraction: {dataset_fraction:.4f}")

    if dataset_fraction >= min_mask_fraction:
        print("Dataset already meets minimum mask fraction; keeping all patches.")
        return [{"image": p["image"], "mask": p["mask"]} for p in patches]

    positive = [p for p in patches if p["mask_pixels"] > 0]
    negative = [p for p in patches if p["mask_pixels"] == 0]

    if not positive:
        print("Warning: no positive patches found; min_mask_fraction cannot be satisfied. Keeping all patches.")
        return [{"image": p["image"], "mask": p["mask"]} for p in patches]

    positive_pixels = sum(p["mask_pixels"] for p in positive)
    max_negatives = int((positive_pixels / (min_mask_fraction * patch_size * patch_size * 255)) - len(positive))
    max_negatives = max(0, min(max_negatives, len(negative)))

    if max_negatives < len(negative):
        print(f"Reducing negative patches from {len(negative)} to {max_negatives} to satisfy min_mask_fraction")
        random.shuffle(negative)
        negative = negative[:max_negatives]

    filtered = positive + negative
    random.shuffle(filtered)
    final_fraction = (sum(p["mask_pixels"] for p in filtered)) / (len(filtered) * patch_size * patch_size * 255)
    print(f"Final dataset mask fraction: {final_fraction:.4f}")
    return [{"image": p["image"], "mask": p["mask"]} for p in filtered]

def hf_dataset_generator(
    image_pairs: List[Tuple[pathlib.Path, pathlib.Path]],
    patch_size: int,
    min_mask_fraction: float,
) -> Generator[Dict[str, Image.Image], None, None]:
    """Yield image/mask patches for the Hugging Face dataset."""

    all_patches: List[Dict[str, object]] = []

    for raw_path, mask_path in image_pairs:
        try:
            raw_pil_image = Image.open(raw_path).convert("RGB")
            binary_mask_pil_image = generate_binary_mask(mask_path)
            patches = extract_patches_from_pair(
                raw_pil_image,
                binary_mask_pil_image,
                patch_size,
            )
            all_patches.extend(patches)
        except FileNotFoundError as e:
            print(f"Skipping pair due to missing file: {e}")
        except Exception as e:
            print(f"Skipping pair {raw_path}, {mask_path} due to error: {e}")

    selected = select_patches_by_dataset_fraction(all_patches, patch_size, min_mask_fraction)
    for item in selected:
        yield item


def main():
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for image segmentation.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/output/",
        help="Directory containing the raw and mask images (e.g., *_raw.png, *_mask.png)."
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default="data/hf_segmentation_dataset",
        help="Path where the Hugging Face dataset will be saved.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of square patches to extract from each image (default: 256).",
    )
    parser.add_argument(
        "--min_mask_fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum fraction of mask pixels required across the entire dataset "
            "(default: 0.01). Negative patches will be dropped if necessary "
            "to satisfy this ratio."
        ),
    )
    args = parser.parse_args()

    print(f"Looking for image pairs in: {args.input_dir}")
    found_pairs = find_image_pairs(args.input_dir)

    if not found_pairs:
        print(f"No image pairs found in '{args.input_dir}'. Exiting.")
        return

    print(f"Found {len(found_pairs)} image pairs.")
    for raw_path, mask_path in found_pairs:
        print(f"  Raw: {raw_path}, Mask: {mask_path}")

    # Define the features for the dataset
    # Masks are single channel (L mode)
    features = datasets.Features({
        'image': datasets.Image(),
        'mask': datasets.Image(mode='L') 
    })

    print("\nCreating Hugging Face dataset...")
    
    # Use a lambda to pass args to the generator
    dataset_generator_func = lambda: hf_dataset_generator(
        found_pairs,
        patch_size=args.patch_size,
        min_mask_fraction=args.min_mask_fraction,
    )
    
    hf_dataset = datasets.Dataset.from_generator(dataset_generator_func, features=features)

    print(f"Dataset created with {len(hf_dataset)} examples.")
    
    # Ensure output directory exists
    output_path = pathlib.Path(args.output_dataset_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to disk at: {output_path.resolve()}")
    hf_dataset.save_to_disk(str(output_path))
    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()
