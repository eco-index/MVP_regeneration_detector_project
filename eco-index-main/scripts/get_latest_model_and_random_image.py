import os
import random
from pathlib import Path
import argparse
import sys

def find_random_image(base_dir_str: str) -> str | None:
    """
    Finds a random image file within the specified base directory and its subdirectories.

    Args:
        base_dir_str (str): The base directory to search for images.

    Returns:
        str | None: The path to a randomly chosen image file as a string, 
                    or None if no images are found.
    """
    base_dir = Path(base_dir_str)
    if not base_dir.is_dir():
        print(f"Error: Image base directory '{base_dir_str}' not found or is not a directory.", file=sys.stderr)
        return None

    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = [
        p for p in base_dir.rglob('*') 
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"Error: No image files found in '{base_dir_str}'.", file=sys.stderr)
        return None
    
    return str(random.choice(image_files))

def find_latest_model(models_dir_str: str) -> str | None:
    """
    Finds the most recently modified model file (e.g., .pth, .pt) 
    within the specified models directory and its subdirectories.

    Args:
        models_dir_str (str): The base directory to search for models.

    Returns:
        str | None: The path to the latest model file as a string, 
                    or None if no models are found.
    """
    models_dir = Path(models_dir_str)
    if not models_dir.is_dir():
        print(f"Error: Models directory '{models_dir_str}' not found or is not a directory.", file=sys.stderr)
        return None

    model_extensions = {'.pth', '.pt'}
    model_files = []
    for ext in model_extensions:
        model_files.extend(list(models_dir.rglob(f'*{ext}')))
    
    # Filter for files only, just in case rglob picked up a directory with a matching suffix
    model_files = [p for p in model_files if p.is_file()]

    if not model_files:
        print(f"Error: No model files (with extensions {model_extensions}) found in '{models_dir_str}'.", file=sys.stderr)
        return None

    try:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    except FileNotFoundError:
        # This can happen if a file is deleted between listing and statting
        print(f"Error: Could not access one or more model files to determine modification time.", file=sys.stderr)
        return None
        
    return str(latest_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the latest model or a random image from specified directories.")
    
    parser.add_argument(
        "--action",
        type=str,
        choices=['get_image', 'get_model'],
        required=True,
        help="The action to perform: 'get_image' to find a random image, 'get_model' to find the latest model."
    )
    parser.add_argument(
        "--images_base_dir",
        type=str,
        default="data/output/",
        help="Base directory to search for images. (Default: 'data/output/')"
    )
    parser.add_argument(
        "--models_base_dir",
        type=str,
        default="models/sam_finetuned/",
        help="Base directory to search for models. (Default: 'models/sam_finetuned/')"
    )

    args = parser.parse_args()

    result_path = None
    if args.action == 'get_image':
        result_path = find_random_image(args.images_base_dir)
    elif args.action == 'get_model':
        result_path = find_latest_model(args.models_base_dir)

    if result_path:
        print(result_path) # Print to stdout for Makefile
    else:
        sys.exit(1) # Exit with error code if no path was found or an error occurred
