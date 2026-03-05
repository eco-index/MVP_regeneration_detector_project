import argparse
import datasets
import matplotlib.pyplot as plt
import random
from pathlib import Path # For path manipulation, though not strictly needed if dataset_path_str is used directly
from PIL import Image # For type hinting of images from dataset

def view_examples(dataset_path_str: str, num_examples: int):
    """
    Loads a Hugging Face dataset and displays a specified number of random
    image-mask pairs using matplotlib.

    Args:
        dataset_path_str (str): Path to the Hugging Face dataset directory.
        num_examples (int): Number of random examples to display.
    """
    try:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists() or not dataset_path.is_dir():
            print(f"Error: Dataset path '{dataset_path_str}' does not exist or is not a directory.")
            return

        print(f"Loading dataset from: {dataset_path_str}")
        # The dataset stores PIL.Image objects as per its creation in prepare_segmentation_dataset.py
        hf_dataset = datasets.load_from_disk(dataset_path_str)
        print(f"Dataset loaded successfully. Total examples: {len(hf_dataset)}")

    except FileNotFoundError:
        print(f"Error: Dataset not found at '{dataset_path_str}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    if len(hf_dataset) == 0:
        print("The dataset is empty. No examples to display.")
        return

    if num_examples <= 0:
        print("Number of examples to display must be positive.")
        return
        
    if num_examples > len(hf_dataset):
        print(f"Warning: Requested {num_examples} examples, but dataset only has {len(hf_dataset)}. "
              f"Displaying all {len(hf_dataset)} examples.")
        num_examples = len(hf_dataset)

    # Select random indices
    # Ensure that `range(len(hf_dataset))` is not empty before sampling
    if len(hf_dataset) > 0:
        selected_indices = random.sample(range(len(hf_dataset)), k=num_examples)
    else: # Should be caught by earlier check, but as a safeguard
        print("Cannot select samples from an empty dataset after initial checks.")
        return


    print(f"\nDisplaying {num_examples} random examples...")

    for i, index in enumerate(selected_indices):
        try:
            sample = hf_dataset[index]
            image: Image.Image = sample['image']
            mask: Image.Image = sample['mask'] # Mask should be 'L' mode (grayscale)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5)) # 1 row, 2 columns
            
            # Display raw image
            ax[0].imshow(image)
            ax[0].set_title(f"Image (Sample {index})")
            ax[0].axis('off') # Hide axes ticks

            # Display binary mask
            # Masks are 'L' mode, so cmap='gray' is appropriate
            ax[1].imshow(mask, cmap='gray') 
            ax[1].set_title(f"Mask (Sample {index})")
            ax[1].axis('off') # Hide axes ticks

            plt.suptitle(f"Example {i+1}/{num_examples}")
            plt.tight_layout() # Adjust layout to prevent overlapping titles
            plt.show() # Shows the current figure, execution pauses until window is closed

        except KeyError as e:
            print(f"Error: Sample at index {index} does not contain expected key: {e}. Skipping this sample.")
        except Exception as e:
            print(f"An error occurred while processing sample at index {index}: {e}. Skipping this sample.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View random image-mask examples from a Hugging Face dataset.")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset directory (e.g., data/hf_segmentation_dataset)."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of random examples to display (default: 5)."
    )
    
    args = parser.parse_args()
    
    print(f"Attempting to view {args.num_examples} examples from dataset: {args.dataset_path}")
    view_examples(args.dataset_path, args.num_examples)
    print("\nFinished viewing examples.")
