import os
import shutil
import argparse
import re
from tqdm import tqdm
from PIL import Image
import numpy as np


def convert_and_save_mask(source_path, dest_path, target_color=(255, 0, 255)):
    """
    Opens an RGB image, identifies pixels of a specific color, and saves a
    new binary (black and white) mask.

    Args:
        source_path (str): Path to the source RGB mask image.
        dest_path (str): Path to save the new binary mask.
        target_color (tuple): The RGB tuple of the color to mask (e.g., magenta).
    """
    try:
        with Image.open(source_path) as img:
            # Ensure image is in RGB format to work with 3-channel color
            img_rgb = img.convert("RGB")

            # Convert image to numpy array for fast processing
            data = np.array(img_rgb)

            # Create a boolean mask where True corresponds to the target color
            # This works by comparing each pixel's [R, G, B] values to the target_color.
            # np.all(..., axis=2) ensures all 3 channels match.
            is_target_color = np.all(data == target_color, axis=2)

            # Create a new single-channel (L mode) array.
            # By default it's all zeros (black).
            height, width = data.shape[:2]
            binary_mask_array = np.zeros((height, width), dtype=np.uint8)

            # Where the color matched, set the pixel value to 255 (white).
            binary_mask_array[is_target_color] = 255

            # Convert the numpy array back to a PIL image and save it
            new_mask_img = Image.fromarray(binary_mask_array, mode="L")
            new_mask_img.save(dest_path)

    except Exception as e:
        print(f"\nError processing mask {source_path}: {e}")
        # As a fallback, you could copy the original if conversion fails
        # shutil.copy2(source_path, dest_path)


def flatten_structure(source_dir, dest_dir):
    """
    Finds all 'polygon_{n}_raw.png' and 'polygon_{n}_mask.png' pairs in a
    nested directory structure. Copies raw images directly and converts mask images
    from a color (e.g., magenta) to a binary format before saving.
    """
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    print("Masks will be converted from Magenta (255, 0, 255) to binary.")

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    # --- Phase 1: Discover all valid pairs ---
    print("\nScanning for image pairs...")
    pairs_to_process = []
    for root, _, files in os.walk(source_dir):
        polygon_files = {}
        pattern = re.compile(r"polygon_(\d+)_(raw|mask)\.png")

        for f in files:
            match = pattern.match(f)
            if match:
                poly_id, file_type = match.groups()
                if poly_id not in polygon_files:
                    polygon_files[poly_id] = {}
                polygon_files[poly_id][file_type] = os.path.join(root, f)

        for poly_id, paths in polygon_files.items():
            if "raw" in paths and "mask" in paths:
                pairs_to_process.append(
                    {"raw_src": paths["raw"], "mask_src": paths["mask"]}
                )

    if not pairs_to_process:
        print("No valid raw/mask pairs found. Exiting.")
        return

    print(f"Found {len(pairs_to_process)} valid pairs to process.")

    # --- Phase 2: Copy raw images and convert/save mask images ---
    print(f"Processing pairs and saving to '{dest_dir}'...")

    pairs_to_process.sort(key=lambda p: p["raw_src"])

    for i, pair in enumerate(tqdm(pairs_to_process, desc="Flattening & Converting")):
        # Define new destination paths with sequential numbering
        dest_raw_path = os.path.join(dest_dir, f"polygon_{i}_raw.png")
        dest_mask_path = os.path.join(dest_dir, f"polygon_{i}_mask.png")

        # Raw images can be copied directly
        shutil.copy2(pair["raw_src"], dest_raw_path)

        # ### THIS IS THE MODIFIED PART ###
        # For masks, call our conversion function instead of a simple copy
        convert_and_save_mask(pair["mask_src"], dest_mask_path)

    print(
        f"\nProcessing complete. Copied and converted {len(pairs_to_process)} pairs to '{dest_dir}'."
    )
    print(
        f"You can now run the create_dataset.py script on the '{dest_dir}' directory."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flattens a nested directory of image pairs into a single directory, converting color masks to binary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/output",
        help="The root directory of the nested data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="satellite_images",
        help="The flat output directory for the create_dataset.py script.",
    )

    args = parser.parse_args()
    flatten_structure(args.input_dir, args.output_dir)
