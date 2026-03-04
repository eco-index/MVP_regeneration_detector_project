import os
import random
import argparse
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm
import re


def sample_pairs(source_pairs, num_pairs, args, output_dir, split_name):
    """Generate cropped pairs for a specific split."""
    output_raw_dir = os.path.join(output_dir, "raw_images")
    output_mask_dir = os.path.join(output_dir, "mask_images")
    os.makedirs(output_raw_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    num_pos_needed = int(num_pairs * (args.pos_neg_ratio / (args.pos_neg_ratio + 1)))
    num_neg_needed = num_pairs - num_pos_needed

    print(
        f"\nGenerating {split_name} split: aiming for {num_pos_needed} positive and {num_neg_needed} negative samples from {len(source_pairs)} source images."
    )

    generated_pairs = []
    max_attempts = num_pairs * 20
    pbar = tqdm(total=num_pairs, desc=f"Generating {split_name}")

    attempts = 0
    while len(generated_pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        source = random.choice(source_pairs)

        try:
            with Image.open(source["raw"]) as raw_img, Image.open(source["mask"]) as mask_img:
                if mask_img.mode != "L":
                    mask_img = mask_img.convert("L")

                img_w, img_h = raw_img.size

                if args.size > img_w or args.size > img_h:
                    continue

                x = random.randint(0, img_w - args.size)
                y = random.randint(0, img_h - args.size)
                box = (x, y, x + args.size, y + args.size)

                raw_crop = raw_img.crop(box)
                mask_crop = mask_img.crop(box)

                mask_array = np.array(mask_crop)
                masked_pixels = np.sum(mask_array > 128)
                total_pixels = args.size * args.size
                masked_fraction = masked_pixels / total_pixels

                is_positive = masked_fraction > args.threshold

                if is_positive and num_pos_needed > 0:
                    num_pos_needed -= 1
                elif not is_positive and num_neg_needed > 0:
                    num_neg_needed -= 1
                else:
                    continue

                generated_pairs.append(
                    {
                        "raw_crop": raw_crop,
                        "mask_crop": mask_crop,
                        "masked_fraction": masked_fraction,
                    }
                )
                pbar.update(1)

        except Exception as e:
            print(f"Error processing {source['raw']} or {source['mask']}: {e}")
            continue

    pbar.close()

    if len(generated_pairs) < num_pairs:
        print(
            f"Warning: Could only generate {len(generated_pairs)} out of {num_pairs} requested pairs for {split_name}."
        )

    csv_path = os.path.join(output_dir, "pairs.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["raw_path", "mask_path", "masked_fraction"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pair in enumerate(tqdm(generated_pairs, desc=f"Saving {split_name}")):
            raw_filename = f"crop_{i}_raw.png"
            mask_filename = f"crop_{i}_mask.png"

            raw_save_path = os.path.join(output_raw_dir, raw_filename)
            mask_save_path = os.path.join(output_mask_dir, mask_filename)

            pair["raw_crop"].save(raw_save_path)
            pair["mask_crop"].save(mask_save_path)

            writer.writerow(
                {
                    "raw_path": os.path.relpath(raw_save_path, os.path.dirname(csv_path)),
                    "mask_path": os.path.relpath(mask_save_path, os.path.dirname(csv_path)),
                    "masked_fraction": f"{pair['masked_fraction']:.6f}",
                }
            )

    print(f"{split_name.capitalize()} split saved to '{output_dir}' with CSV manifest at '{csv_path}'.")


def create_dataset(args):
    """
    Samples random square crops from large satellite images and their corresponding
    masks to create a new dataset.
    """
    # --- 1. Setup and Validation ---
    print("--- Configuration ---")
    print(f"Source Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Crop Size: {args.size}x{args.size}")
    print(f"Total Pairs: {args.num_pairs}")
    print(f"Positive/Negative Ratio: {args.pos_neg_ratio}")
    print(f"Masked Fraction Threshold: {args.threshold}")
    print("-----------------------")

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Find all source image pairs ---
    all_files = os.listdir(args.input_dir)
    polygon_ids = sorted(
        list(
            set(
                [
                    re.match(r"polygon_(\d+)_", f).group(1)
                    for f in all_files
                    if re.match(r"polygon_(\d+)_", f)
                ]
            )
        )
    )

    source_pairs = []
    for pid in polygon_ids:
        raw_path = os.path.join(args.input_dir, f"polygon_{pid}_raw.png")
        mask_path = os.path.join(args.input_dir, f"polygon_{pid}_mask.png")
        if os.path.exists(raw_path) and os.path.exists(mask_path):
            source_pairs.append({"id": pid, "raw": raw_path, "mask": mask_path})

    if not source_pairs:
        print(
            f"Error: No valid 'polygon_{{n}}_raw.png' and 'polygon_{{n}}_mask.png' pairs found in '{args.input_dir}'."
        )
        return

    print(f"Found {len(source_pairs)} source image pairs.")

    # --- 3. Split source images into train and test sets ---
    random.shuffle(source_pairs)
    split_idx = int(len(source_pairs) * args.train_split)
    if split_idx == 0:
        split_idx = 1
    if split_idx == len(source_pairs):
        split_idx = len(source_pairs) - 1

    train_sources = source_pairs[:split_idx]
    test_sources = source_pairs[split_idx:]

    train_pairs = int(args.num_pairs * args.train_split)
    test_pairs = args.num_pairs - train_pairs

    print(
        f"Creating {train_pairs} training pairs from {len(train_sources)} source images and {test_pairs} testing pairs from {len(test_sources)} source images."
    )

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")

    sample_pairs(train_sources, train_pairs, args, train_dir, "train")
    sample_pairs(test_sources, test_pairs, args, test_dir, "test")

    print("\nDataset creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a dataset of image/mask crops from large satellite images."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="satellite_images",
        help="Directory containing the source PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Directory where the new dataset will be saved.",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="The size of the square crops (e.g., 512 for 512x512).",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        required=True,
        help="Total number of image/mask pairs to generate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help='Masked fraction threshold to define a "positive" sample (e.g., 0.01 for >1%% masked).',
    )
    parser.add_argument(
        "--pos_neg_ratio",
        type=float,
        required=True,
        help="Desired ratio of positive to negative samples (e.g., 1.0 for a 1:1 balance).",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of the dataset to use for training; remainder used for testing.",
    )

    args = parser.parse_args()
    create_dataset(args)
