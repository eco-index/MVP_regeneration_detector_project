import argparse
import datasets
from huggingface_hub import HfApi, HfFolder, HfHubHTTPError # For push_to_hub and token management
import os
import requests # For requests.exceptions.HTTPError

def publish_dataset(local_dataset_path_str: str, repo_id: str, hf_token: str = None, is_private: bool = False):
    """
    Loads a local Hugging Face dataset and pushes it to the Hugging Face Hub.

    Args:
        local_dataset_path_str (str): Path to the local Hugging Face dataset directory.
        repo_id (str): The Hugging Face Hub repository ID (e.g., "username/dataset_name").
        hf_token (str, optional): Hugging Face API token. If None, tries to use
                                  locally saved token or HF_TOKEN env var.
        is_private (bool, optional): If True, makes the dataset private on the Hub.
                                     Defaults to False (public).
    """
    try:
        print(f"Loading local dataset from: {local_dataset_path_str}")
        dataset = datasets.load_from_disk(local_dataset_path_str)
        print(f"Dataset loaded successfully. Total examples: {len(dataset)}")
    except FileNotFoundError:
        print(f"Error: Local dataset not found at '{local_dataset_path_str}'. Please check the path.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the local dataset: {e}")
        raise

    # Check for token if not provided
    effective_token = hf_token
    if not effective_token:
        effective_token = HfFolder.get_token() # Checks token saved by `huggingface-cli login`
        if not effective_token:
            effective_token = os.environ.get("HF_TOKEN") # Checks HF_TOKEN environment variable
    
    if not effective_token:
        print("\nWarning: No Hugging Face API token provided via --hf_token argument, "
              "'huggingface-cli login', or HF_TOKEN environment variable.")
        print("Publishing will likely fail if the repository is private or requires specific permissions.")
        print("Consider logging in using 'huggingface-cli login' or providing a token.\n")
        # Proceeding without a token, push_to_hub will attempt anonymous push or use git credentials if available.

    try:
        print(f"\nPushing dataset to Hugging Face Hub repository: {repo_id}")
        print(f"Private: {is_private}")
        
        # The push_to_hub method handles token authentication internally if `token` is passed.
        # If token is None, it relies on the cached token from `huggingface-cli login` or git credentials.
        dataset.push_to_hub(
            repo_id=repo_id,
            token=effective_token, # Pass the resolved token
            private=is_private
        )
        
        hub_url_base = "https://huggingface.co/datasets/"
        dataset_url = f"{hub_url_base}{repo_id}"
        print("\nDataset published successfully!")
        print(f"View your dataset on the Hub: {dataset_url}")

    except HfHubHTTPError as e: # Specific error for Hugging Face Hub HTTP issues (auth, not found, etc.)
        print(f"Error during dataset push to Hugging Face Hub: {e}")
        print("This could be due to authentication issues (invalid token?), repository not found, "
              "or network problems.")
        if e.response:
            print(f"Server response: {e.response.status_code} - {e.response.text}")
        raise
    except requests.exceptions.HTTPError as e: # More generic HTTP errors
        print(f"A network error occurred: {e}")
        raise
    except ValueError as e: # Can occur if repo_id is invalid
        print(f"A ValueError occurred, possibly due to an invalid repo_id '{repo_id}': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during publishing: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a local Hugging Face dataset to the Hugging Face Hub.")
    
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        required=True,
        help="Path to the local Hugging Face dataset directory (e.g., data/hf_segmentation_dataset)."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The Hugging Face Hub repository ID (e.g., YourUsername/YourDatasetName)."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None, # Changed from required=False to default=None for clarity
        help="Optional Hugging Face API token. If not provided, relies on "
             "'huggingface-cli login' or HF_TOKEN environment variable."
    )
    parser.add_argument(
        "--private",
        action='store_true', # Makes this a flag; if present, value is True, else False
        help="Set this flag to make the dataset private on the Hub (defaults to public)."
    )
    
    args = parser.parse_args()
    
    print("--- Starting Dataset Publishing Process ---")
    print(f"Local dataset path: {args.local_dataset_path}")
    print(f"Target Hub repo ID: {args.repo_id}")
    print(f"Publish as private: {args.private}")
    if args.hf_token:
        print("Using provided Hugging Face token.")
    else:
        print("No explicit Hugging Face token provided via argument; will check environment/login.")

    try:
        publish_dataset(
            local_dataset_path_str=args.local_dataset_path,
            repo_id=args.repo_id,
            hf_token=args.hf_token,
            is_private=args.private
        )
        print("\n--- Dataset Publishing Process Finished Successfully ---")
    except FileNotFoundError:
        print("\nError: The specified local dataset path was not found. Please check the path and try again.")
        print("--- Dataset Publishing Process Failed ---")
    except HfHubHTTPError:
        print("\nError: Failed to publish to Hugging Face Hub due to an HTTP error (check logs above for details).")
        print("Ensure your token is valid and has write permissions to the repository.")
        print("--- Dataset Publishing Process Failed ---")
    except requests.exceptions.HTTPError:
        print("\nError: A network error occurred. Please check your internet connection and try again.")
        print("--- Dataset Publishing Process Failed ---")
    except ValueError:
        print("\nError: Invalid repository ID format. Please use 'username/dataset_name'.")
        print("--- Dataset Publishing Process Failed ---")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("--- Dataset Publishing Process Failed ---")
