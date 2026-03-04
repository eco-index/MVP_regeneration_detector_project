"""
Join Waiwhakaiho tiles with their probability maps side by side, while skipping
tiles that never loaded (e.g., the starfield Google Earth splash screen) using a
simple low-frequency brightness filter.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageStat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine raw tiles and probability tiles into a single image."
    )
    parser.add_argument(
        "--left-dir",
        default="data/Waiwhakaiho_grid",
        help="Directory with the raw tiles to place on the left.",
    )
    parser.add_argument(
        "--right-dir",
        default="data/waiwhakaiho_predictions/probabilities",
        help="Directory with probability tiles (expects *_prob.png naming).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/waiwhakaiho_joined",
        help="Directory to write combined images into.",
    )
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=0.12,
        help="Skip tiles whose mean brightness is below this value (0-1 range).",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=0.12,
        help="Skip tiles whose brightness stddev is below this value (0-1 range).",
    )
    parser.add_argument(
        "--dark-fraction",
        type=float,
        default=0.85,
        help=(
            "Skip tiles when the fraction of dark pixels is above this value; used "
            "alongside the mean/std thresholds."
        ),
    )
    parser.add_argument(
        "--dark-cutoff",
        type=int,
        default=40,
        help="Pixel value (0-255) considered 'dark' for the dark-fraction check.",
    )
    parser.add_argument(
        "--disable-filter",
        action="store_true",
        help="Process every tile without dropping low-information tiles.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tiles (after sorting). Useful for a quick dry run.",
    )
    return parser.parse_args()


def low_frequency_metrics(img: Image.Image, dark_cutoff: int) -> Tuple[float, float, float]:
    """
    Compute coarse mean, stddev, and fraction of dark pixels on a downscaled
    version of the image to cheaply capture low-frequency content.
    """
    dark_cutoff = max(0, min(255, dark_cutoff))
    reduced = img.convert("RGB").resize((64, 64))
    stat = ImageStat.Stat(reduced)
    mean = sum(stat.mean) / (len(stat.mean) * 255.0)
    stddev = sum(stat.stddev) / (len(stat.stddev) * 255.0)

    gray = reduced.convert("L")
    hist = gray.histogram()
    total_pixels = sum(hist) or 1
    dark_pixels = sum(hist[:dark_cutoff])
    dark_fraction = dark_pixels / total_pixels
    return mean, stddev, dark_fraction


def is_unloaded_tile(
    image_path: Path,
    mean_threshold: float,
    std_threshold: float,
    dark_fraction_threshold: float,
    dark_cutoff: int,
) -> Tuple[bool, Tuple[float, float, float]]:
    with Image.open(image_path) as img:
        mean, stddev, dark_fraction = low_frequency_metrics(img, dark_cutoff)
    unloaded = (
        mean < mean_threshold
        and stddev < std_threshold
        and dark_fraction > dark_fraction_threshold
    )
    return unloaded, (mean, stddev, dark_fraction)


def build_probability_path(left_path: Path, right_dir: Path) -> Path:
    return right_dir / f"{left_path.stem}_prob{left_path.suffix}"


def join_pair(left_path: Path, right_path: Path, output_path: Path) -> None:
    with Image.open(left_path) as left_img, Image.open(right_path) as right_img:
        left = left_img.convert("RGB")
        right = right_img.convert("RGB")

        if left.height != right.height:
            ratio = left.height / right.height
            new_width = max(1, int(right.width * ratio))
            right = right.resize((new_width, left.height))

        canvas = Image.new("RGB", (left.width + right.width, left.height))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        canvas.save(output_path)


def sorted_images(directory: Path) -> Iterable[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main() -> None:
    args = parse_args()
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_files = list(sorted_images(left_dir))
    if args.limit:
        left_files = left_files[: args.limit]

    if not left_files:
        print(f"No input images found in '{left_dir}'.")
        return

    print(f"Found {len(left_files)} left tiles in '{left_dir}'.")
    print(f"Writing combined images to '{output_dir}'.")
    if args.disable_filter:
        print("Low-information filter disabled; processing all tiles.")
    else:
        print(
            "Filtering tiles with mean<{:.3f}, std<{:.3f}, dark_fraction>{:.2f} "
            "using dark_cutoff {}.".format(
                args.mean_threshold, args.std_threshold, args.dark_fraction, args.dark_cutoff
            )
        )

    missing = 0
    filtered = 0
    written = 0
    filtered_examples = []

    for left_path in left_files:
        right_path = build_probability_path(left_path, right_dir)
        if not right_path.exists():
            missing += 1
            continue

        if not args.disable_filter:
            unload, metrics = is_unloaded_tile(
                left_path,
                args.mean_threshold,
                args.std_threshold,
                args.dark_fraction,
                args.dark_cutoff,
            )
            if unload:
                filtered += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append((left_path.name, metrics))
                continue

        output_name = f"{left_path.stem}_joined{left_path.suffix}"
        output_path = output_dir / output_name
        join_pair(left_path, right_path, output_path)
        written += 1

    print(f"Processed {len(left_files)} tiles.")
    print(f" - Missing probability pairs: {missing}")
    print(f" - Filtered low-information tiles: {filtered}")
    print(f" - Wrote combined images: {written}")

    if filtered_examples:
        print("First few filtered tiles (mean, stddev, dark_fraction):")
        for name, (mean, stddev, dark_fraction) in filtered_examples:
            print(f"   {name}: mean={mean:.4f}, std={stddev:.4f}, dark_fraction={dark_fraction:.3f}")


if __name__ == "__main__":
    main()
