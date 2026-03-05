import torch
from torch.utils.data import Dataset
import datasets # Hugging Face datasets library
from PIL import Image
from typing import Tuple, Optional
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os # For os.cpu_count()

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for image segmentation tasks.
    Loads data from a Hugging Face dataset saved to disk.
    Applies augmentations using Albumentations.
    """
    def __init__(self, hf_dataset_path: str, augmentations: Optional[A.Compose] = None):
        """
        Args:
            hf_dataset_path (str): Path to the Hugging Face dataset directory.
            augmentations (Optional[A.Compose]): Albumentations Compose object for augmentations.
        """
        super().__init__()
        self.hf_dataset_path = hf_dataset_path
        try:
            self.hf_dataset = datasets.load_from_disk(self.hf_dataset_path)
            print(f"Successfully loaded Hugging Face dataset from {self.hf_dataset_path}")
        except Exception as e:
            print(f"Error loading Hugging Face dataset from {self.hf_dataset_path}: {e}")
            raise
        
        self.augmentations = augmentations

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the sample (image and mask) at the given index and applies augmentations.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the augmented
                                               image tensor and mask tensor.
        """
        if idx < 0 or idx >= len(self.hf_dataset):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.hf_dataset)}")

        item = self.hf_dataset[idx]
        
        pil_image: Image.Image = item['image']
        pil_mask: Image.Image = item['mask']

        # Convert PIL images to NumPy arrays
        # Image is HWC, Mask is HW
        numpy_image = np.array(pil_image)
        numpy_mask = np.array(pil_mask) # Mask should be 2D (H, W)

        if self.augmentations:
            augmented = self.augmentations(image=numpy_image, mask=numpy_mask)
            image_tensor = augmented['image'] # Should be a tensor if ToTensorV2 is used
            mask_tensor = augmented['mask']   # Should be a tensor if ToTensorV2 is used
        else:
            # If no augmentations, or if ToTensorV2 is not part of augmentations,
            # we need to convert manually. For simplicity, assume ToTensorV2 is used if augmentations are present.
            # This block is a fallback or for cases where augmentations don't include ToTensorV2.
            # However, the standard practice is to include ToTensorV2 in the Compose pipeline.
            # For the image: Convert HWC uint8 to CHW float32 and scale to [0,1]
            image_tensor = torch.from_numpy(numpy_image.transpose((2, 0, 1))).float().div(255)
            # For the mask: Convert HW uint8 to HW (or 1HW for some losses) long
            mask_tensor = torch.from_numpy(numpy_mask).long() # Often .long() for CrossEntropyLoss
            # If mask needs to be (1, H, W) then: mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor

if __name__ == '__main__':
    print("Attempting to demonstrate SegmentationDataset with Albumentations...")
    
    example_hf_dataset_path = "data/hf_segmentation_dataset" 

    # Define an example augmentation pipeline
    # Normalization values are typical for ImageNet, adjust if your dataset is very different.
    # The mask is typically not normalized.
    # ToTensorV2 converts the image to CHW PyTorch tensor and scales to [0,1] if it's uint8.
    # It also converts the mask to a PyTorch tensor (usually HWC or HW, depending on input)
    # and maintains its data type or converts it appropriately.
    # For segmentation masks, ToTensorV2 typically converts HW numpy array to CHW tensor where C=1.
    # We might need to squeeze it later if the loss function expects HW.
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.75), # Adjusted scale and p
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Applied to image only
        ToTensorV2() # Converts image and mask to PyTorch Tensors
    ])

    try:
        # Create an instance of the dataset with augmentations
        augmented_dataset = SegmentationDataset(
            hf_dataset_path=example_hf_dataset_path,
            augmentations=train_transforms
        )
        
        print(f"Number of samples in the augmented dataset: {len(augmented_dataset)}")
        
        if len(augmented_dataset) > 0:
            # Get the first sample
            img_tensor, mask_tensor = augmented_dataset[0]
            
            print("\nFirst augmented sample details:")
            print(f"  Image tensor type: {img_tensor.dtype}")
            print(f"  Image tensor shape: {img_tensor.shape}") # Expected: C x H x W (e.g., 3 x 256 x 256)
            
            print(f"  Mask tensor type: {mask_tensor.dtype}")   # Expected: torch.long or torch.float
            print(f"  Mask tensor shape: {mask_tensor.shape}")  # Expected: H x W or 1 x H x W (e.g., 256 x 256 or 1 x 256 x 256)
            
            # ToTensorV2 for masks usually results in (C, H, W) where C=1 for grayscale masks.
            # Depending on the loss function, you might need to squeeze the channel dimension.
            # e.g., if mask_tensor.shape is (1, 256, 256), and you need (256, 256)
            # mask_tensor_squeezed = mask_tensor.squeeze(0)
            # print(f"  Mask tensor squeezed shape (if needed): {mask_tensor_squeezed.shape}")
            
            print("\n--- DataLoader Demonstration ---")
            # Use a sensible number of workers, os.cpu_count() can be too many if RAM is limited for each worker.
            # For an example, 2 is usually fine. For actual training, this is an important parameter.
            num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 2 # Cap at 4 for safety in diverse envs
            
            # Ensure there are enough samples for a few batches, or adjust batch_size/break condition
            if len(augmented_dataset) >= 4 : # Ensure at least one batch can be formed for batch_size=4
                dataloader = DataLoader(
                    augmented_dataset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True # Good practice if using CUDA
                )

                print(f"Created DataLoader with batch_size=4, shuffle=True, num_workers={num_workers}")
                
                for i, (batch_images, batch_masks) in enumerate(dataloader):
                    print(f"\nBatch {i+1}:")
                    print(f"  Batched images shape: {batch_images.shape}") # Expected: N x C x H x W
                    print(f"  Batched masks shape: {batch_masks.shape}")   # Expected: N x (1 or C) x H x W
                    
                    if i >= 2: # Show 3 batches (0, 1, 2)
                        print("\nDataLoader demonstration finished after 3 batches.")
                        break
                if i < 2 and len(augmented_dataset) > 0 : # If loop finished early due to small dataset size
                     print("\nDataLoader demonstration finished (dataset smaller than 3 batches).")

            else:
                print("\nSkipping DataLoader demonstration as dataset is too small for batch_size=4.")


        else:
            print("Dataset is empty. Cannot retrieve the first sample or demonstrate DataLoader.")
            
        # Example without augmentations (should still work if ToTensorV2 is not in a default pipeline)
        print("\nTesting dataset without explicit augmentations (relies on internal fallback or direct PIL handling if modified):")
        plain_dataset = SegmentationDataset(hf_dataset_path=example_hf_dataset_path, augmentations=None)
        if len(plain_dataset) > 0:
            img_plain, mask_plain = plain_dataset[0]
            print(f"  Plain Image tensor type: {img_plain.dtype}") # Should be float from manual conversion
            print(f"  Plain Image tensor shape: {img_plain.shape}") # C x H x W
            print(f"  Plain Mask tensor type: {mask_plain.dtype}")   # Should be long from manual conversion
            print(f"  Plain Mask tensor shape: {mask_plain.shape}")  # H x W
        else:
            print("Plain dataset is empty.")


    except FileNotFoundError:
        print(f"ERROR: The example dataset was not found at '{example_hf_dataset_path}'.")
        print("Please ensure you have run 'scripts/prepare_segmentation_dataset.py' successfully,")
        print("or provide the correct path to your Hugging Face dataset.")
    except Exception as e:
        print(f"An error occurred during the example usage: {e}")
        import traceback
        traceback.print_exc()


    print("\nSegmentationDataset with Albumentations demonstration finished.")
