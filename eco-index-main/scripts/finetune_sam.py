import torch
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import datasets  # Hugging Face datasets library
from PIL import Image
from typing import Tuple, Optional
import numpy as np
import os  # For os.cpu_count()
from transformers import SamModel, AutoProcessor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F  # For resizing masks
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm  # For progress bars
from torch.utils.data import random_split, Subset
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import numpy as np  # Ensure numpy is imported for sklearn metrics if not already for other reasons
import random
from typing import List, Tuple, Optional
import pathlib

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
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


class BalancedCropDataset(Dataset):
    """In-memory dataset that samples balanced random crops from image-mask pairs."""

    def __init__(
        self,
        data_dir: str,
        patch_size: int,
        pos_fraction: float,
        augmentations: Optional[A.Compose] = None,
        samples_per_epoch: int = 1000,
    ) -> None:
        """Load all images into memory and prepare for random cropping.

        Args:
            data_dir: Directory containing ``*_raw.png`` and ``*_mask.png`` files.
            patch_size: Side length of square crops to return.
            pos_fraction: Desired ratio of crops containing mask pixels.
            augmentations: Optional Albumentations pipeline applied to each crop.
            samples_per_epoch: Length reported by ``__len__``.
        """
        super().__init__()
        self.patch_size = patch_size
        self.pos_fraction = pos_fraction
        self.augmentations = augmentations
        self.samples_per_epoch = samples_per_epoch

        pairs = find_image_pairs(data_dir)
        if not pairs:
            raise ValueError(f"No *_raw.png/_mask.png pairs found in {data_dir}")

        self.items: List[Tuple[np.ndarray, np.ndarray]] = []
        for raw_path, mask_path in pairs:
            raw_img = Image.open(raw_path).convert("RGB")
            mask_img = generate_binary_mask(mask_path)
            # Store raw images as RGB numpy arrays
            self.items.append((np.array(raw_img), np.array(mask_img)))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_crop(self) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(50):
            img_np, mask_np = random.choice(self.items)
            h, w = mask_np.shape
            if h < self.patch_size or w < self.patch_size:
                continue
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img_crop = img_np[top : top + self.patch_size, left : left + self.patch_size]
            mask_crop = mask_np[top : top + self.patch_size, left : left + self.patch_size]
            return img_crop, mask_crop
        # Fallback if no crop could be sampled
        img_np, mask_np = random.choice(self.items)
        return img_np[: self.patch_size, : self.patch_size], mask_np[: self.patch_size, : self.patch_size]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_positive = random.random() < self.pos_fraction
        for _ in range(50):
            image_np, mask_np = self._sample_crop()
            has_mask = mask_np.sum() > 0
            if has_mask == target_positive:
                break
        else:
            # If desired balance can't be met, just use the last sample
            pass

        if self.augmentations:
            # Albumentations expects BGR images. Convert before augmentation
            augmented = self.augmentations(image=image_np[:, :, ::-1], mask=mask_np)
            image_aug = augmented["image"]
            mask_aug = augmented["mask"]

            if isinstance(image_aug, torch.Tensor):
                # ToTensorV2 may already convert to tensor
                if image_aug.shape[0] == 3:
                    img_tensor = image_aug[[2, 1, 0]].float() / 255.0
                else:
                    img_tensor = image_aug.float() / 255.0
            else:
                img_tensor = torch.from_numpy(image_aug[:, :, ::-1].transpose(2, 0, 1)).float() / 255.0

            if isinstance(mask_aug, torch.Tensor):
                mask_tensor = mask_aug.float() / 255.0
            else:
                mask_tensor = torch.from_numpy(mask_aug).float() / 255.0
        else:
            img_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).float() / 255.0

        return img_tensor, mask_tensor


# --- SegmentationDataset Class (copied from scripts/segmentation_dataloader.py) ---
class SegmentationDataset(torch.utils.data.Dataset):
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
            print(
                f"Successfully loaded Hugging Face dataset from {self.hf_dataset_path}"
            )
        except Exception as e:
            print(
                f"Error loading Hugging Face dataset from {self.hf_dataset_path}: {e}"
            )
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
            raise IndexError(
                f"Index {idx} out of bounds for dataset with length {len(self.hf_dataset)}"
            )

        item = self.hf_dataset[idx]

        pil_image: Image.Image = item["image"]
        pil_mask: Image.Image = item["mask"]

        # Convert PIL images to NumPy arrays
        # Image is HWC, Mask is HW
        numpy_image = np.array(pil_image)
        numpy_mask = np.array(pil_mask)  # Mask should be 2D (H, W)

        if self.augmentations:
            augmented = self.augmentations(image=numpy_image, mask=numpy_mask)
            image_tensor = augmented["image"]
            # Mask from ToTensorV2 (if uint8 input) is typically torch.uint8 with [0, 255]
            # We need to convert it to float32 and scale to [0.0, 1.0]
            # for BCEWithLogitsLoss targets and consistent metric evaluation.
            mask_tensor_from_aug = augmented["mask"]
            mask_tensor = mask_tensor_from_aug.float() / 255.0
        else:
            # If no augmentations, or if ToTensorV2 is not part of augmentations,
            # we need to convert manually. For simplicity, assume ToTensorV2 is used if augmentations are present.
            image_tensor = (
                torch.from_numpy(numpy_image.transpose((2, 0, 1))).float().div(255.0)
            )
            # numpy_mask is uint8 [0, 255], convert to float32 [0.0, 1.0]
            mask_tensor = torch.from_numpy(numpy_mask).float().div(255.0)

        return image_tensor, mask_tensor


# --- End of SegmentationDataset Class ---


# --- Evaluation Function ---
def evaluate_model(
    model, val_dataloader, processor, criterion, device, image_size, eval_metrics_list
):
    model.eval()
    total_val_loss = 0.0

    # Initialize metric storage
    metric_values = {metric: [] for metric in eval_metrics_list if metric != "loss"}
    all_preds_flat = []
    all_masks_flat = []

    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validating", leave=False)
        for batch_images, batch_masks_gt in progress_bar:
            batch_images = batch_images.to(device)
            batch_masks_gt = batch_masks_gt.to(device)

            if batch_masks_gt.dtype == torch.long:
                batch_masks_gt = batch_masks_gt.float()
            if batch_masks_gt.ndim == 3:
                batch_masks_gt = batch_masks_gt.unsqueeze(1)

            try:
                inputs = processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                outputs = model(pixel_values=pixel_values, multimask_output=False)
                predicted_masks_logits = outputs.pred_masks

                # Similar to the inference script, predicted mask logits may
                # contain extraneous singleton dimensions (e.g. [B, 1, 1, H, W]).
                # Collapse them to ensure a 4D tensor of shape [B, 1, H, W].
                predicted_masks_logits = predicted_masks_logits.squeeze()
                if predicted_masks_logits.ndim == 3:
                    predicted_masks_logits = predicted_masks_logits.unsqueeze(1)
                elif predicted_masks_logits.ndim == 2:
                    predicted_masks_logits = predicted_masks_logits.unsqueeze(0).unsqueeze(0)

                predicted_masks_resized = F.interpolate(
                    predicted_masks_logits,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )

                loss = criterion(predicted_masks_resized, batch_masks_gt)
                total_val_loss += loss.item()

                if any(m in eval_metrics_list for m in ["accuracy", "iou", "dice"]):
                    # Convert logits to binary predictions
                    preds_binary = (torch.sigmoid(predicted_masks_resized) > 0.5).byte()

                    # Flatten for sklearn metrics
                    # Ensure masks are [N, H, W] or [N, 1, H, W] then flatten
                    # For sklearn, we need 1D arrays
                    current_preds_flat = preds_binary.view(-1).cpu().numpy()
                    current_masks_flat = (
                        batch_masks_gt.view(-1).cpu().numpy()
                    )  # Ensure GT is also binary 0/1

                    all_preds_flat.extend(current_preds_flat)
                    all_masks_flat.extend(current_masks_flat)

            except Exception as e:
                print(f"Error during validation batch: {e}")
                continue

    avg_val_loss = total_val_loss / len(val_dataloader)
    results = {"loss": avg_val_loss}

    if "accuracy" in eval_metrics_list and len(all_masks_flat) > 0:
        results["accuracy"] = accuracy_score(all_masks_flat, all_preds_flat)
    if "iou" in eval_metrics_list and len(all_masks_flat) > 0:  # Jaccard
        results["iou"] = jaccard_score(
            all_masks_flat, all_preds_flat, average="binary", zero_division=0
        )
    if "dice" in eval_metrics_list and len(all_masks_flat) > 0:  # F1 Score
        results["dice"] = f1_score(
            all_masks_flat, all_preds_flat, average="binary", zero_division=0
        )

    model.train()  # Set back to train mode
    return results


def main(args):
    print("Starting finetuning script...")
    print(f"Arguments: {args}")

    # --- Output Directory ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Model checkpoints and final model will be saved to: {args.output_dir}")

    # --- 1. Define Augmentations ---
    print(
        f"\n1. Defining augmentations for image size: {args.image_size}x{args.image_size}"
    )
    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                size=(256, 256),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.33),
                p=1,
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75
            ),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(
                height=args.image_size, width=args.image_size
            ),  # Ensure consistent size for validation
            ToTensorV2(),
        ]
    )
    print("Train and Validation augmentation pipelines defined.")

    # --- 2. Dataset Loading and Splitting ---
    print(f"\n2. Loading BalancedCropDataset from: {args.dataset_path}")
    try:
        train_dataset_subset = BalancedCropDataset(
            data_dir=args.dataset_path,
            patch_size=args.image_size,
            pos_fraction=args.pos_fraction,
            augmentations=train_transforms,
            samples_per_epoch=args.samples_per_epoch,
        )
        val_samples = max(1, int(args.samples_per_epoch * args.val_split_ratio))
        val_dataset_subset = BalancedCropDataset(
            data_dir=args.dataset_path,
            patch_size=args.image_size,
            pos_fraction=args.pos_fraction,
            augmentations=val_transforms,
            samples_per_epoch=val_samples,
        )
        print(
            f"BalancedCropDataset created. Train samples per epoch: {len(train_dataset_subset)}, Validation samples per epoch: {len(val_dataset_subset)}"
        )
    except Exception as e:
        print(f"Error creating BalancedCropDataset: {e}")
        return

    # --- 3. DataLoader Initialization ---
    if not train_dataset_subset:  # Should not happen if logic above is correct
        print("Train dataset subset is None. Exiting.")
        return

    print(f"\n3. Initializing DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Training DataLoader initialized with {len(train_dataset_subset)} samples.")

    if val_dataset_subset and len(val_dataset_subset) > 0:
        val_dataloader = DataLoader(
            val_dataset_subset,
            batch_size=args.batch_size,  # Can use same or different batch size for validation
            shuffle=False,  # No need to shuffle validation data
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(
            f"Validation DataLoader initialized with {len(val_dataset_subset)} samples."
        )
    else:
        val_dataloader = None
        print(
            "No validation DataLoader initialized as validation set is empty or not created."
        )

    # Placeholder print for an example batch shape from train_dataloader
    if len(train_dataloader) > 0:
        try:
            first_batch_images, first_batch_masks = next(iter(train_dataloader))
            print(
                f"  Train DataLoader: Example batch - Images shape: {first_batch_images.shape}, Masks shape: {first_batch_masks.shape}"
            )
        except StopIteration:
            print(
                "  Train DataLoader: Could not retrieve a batch (dataset might be smaller than batch size or empty)."
            )
        except Exception as e:
            print(f"  Train DataLoader: Error retrieving batch: {e}")
    else:
        print("Train DataLoader is empty.")

    # --- 4. Model loading... (Section number unchanged for brevity in diff)
    # --- 4. Model and Processor Loading ---
    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA specified but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    try:
        print(f"Loading SAM model: {args.model_name}")
        model = SamModel.from_pretrained(args.model_name)
        model.to(device)
        print(f"SAM model loaded successfully and moved to {device}.")

        print(f"Loading SAM processor: {args.model_name}")
        processor = AutoProcessor.from_pretrained(args.model_name)
        print("SAM processor loaded successfully.")

        # Comments on SAM's architecture for segmentation:
        # - SAM's default architecture is designed for prompt-based segmentation.
        # - For fine-tuning on a new dataset for a specific segmentation task (like binary segmentation here),
        #   you might typically:
        #   1. Modify the model head (e.g., replace or add a segmentation head).
        #   2. Freeze parts of the model (e.g., the vision encoder) and only train the new head or specific layers.
        #   3. Ensure the loss function is compatible with the model's output format.
        #      SAM's `SamImageSegmentorOutput` includes `iou_scores` and `pred_masks`.
        #      For binary segmentation, you'll likely work with `pred_masks`.
        # For now, we are loading the base pre-trained model. Adjustments will be made in subsequent steps.

    except Exception as e:
        print(f"Error loading SAM model or processor: {e}")
        return  # Exit if model loading fails

    # --- 5. Training Components Definition ---
    print("\n5. Defining Training Components (Loss, Optimizer, Scheduler)...")

    # Loss Function
    if args.loss_function_type.lower() == "bcewithlogits":
        criterion = nn.BCEWithLogitsLoss()
        print(f"Using loss function: BCEWithLogitsLoss")
    # Add other loss functions here later (e.g., DiceLoss)
    # elif args.loss_function_type.lower() == 'dice':
    #    criterion = DiceLoss() # Assuming DiceLoss is defined/imported
    #    print(f"Using loss function: DiceLoss")
    else:
        print(
            f"Unsupported loss function type: {args.loss_function_type}. Defaulting to BCEWithLogitsLoss."
        )
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    if args.optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        print(f"Using optimizer: AdamW with learning rate {args.learning_rate}")
    # Add other optimizers here later (e.g., SGD)
    # elif args.optimizer_type.lower() == 'sgd':
    #    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    #    print(f"Using optimizer: SGD with learning rate {args.learning_rate}")
    else:
        print(
            f"Unsupported optimizer type: {args.optimizer_type}. Defaulting to AdamW."
        )
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Learning Rate Scheduler (Optional)
    # Example: StepLR scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # print("Using StepLR scheduler (currently as placeholder).")
    scheduler = None  # No scheduler by default for now
    print("No learning rate scheduler active by default.")

    # Comments on Loss Calculation for SAM:
    # - The SAM model, when not using prompts for zero-shot segmentation, can be made to produce masks.
    #   The `SamModel`'s forward pass can return `SamImageSegmentorOutput`, which contains `pred_masks`.
    #   These are typically low-resolution masks. For higher resolution, `upscaled_masks` are also available.
    # - `upscaled_masks` (e.g., shape [batch_size, num_masks_per_image, height, width]) are usually what you'd use for loss calculation.
    #   Since we're doing binary segmentation, we might only care about one mask per image, or need to handle multiple predicted masks.
    #   The `multimask_output=False` parameter in the model's forward pass can simplify this to one mask.
    # - Target masks (from DataLoader) should be:
    #   1. Resized to match the model's output mask dimensions (e.g., args.image_size x args.image_size, or the model's native output size if not resizing).
    #      Our current DataLoader provides masks at args.image_size.
    #   2. Of the correct data type (e.g., `torch.float32` for `BCEWithLogitsLoss`).
    #   3. Have the correct shape (e.g., `[batch_size, 1, height, width]` for `BCEWithLogitsLoss`).
    #      Our current SegmentationDataset provides masks as [H, W] or [1, H, W] which might need unsqueezing/squeezing.
    #      The ToTensorV2 usually makes it [C, H, W]. If C=1, it's suitable.
    # - The loss will be calculated by comparing the model's predicted masks (after sigmoid if using BCEWithLogitsLoss implicitly)
    #   with these ground truth masks.

    # --- 6. Training Loop ---
    print(f"\n6. Starting Training for {args.num_epochs} epochs...")

    # Initialize best validation score based on the chosen metric
    if args.save_best_metric == "loss":
        best_val_score = float("inf")  # Lower is better for loss
    else:  # For 'accuracy', 'iou', 'dice'
        best_val_score = float("-inf")  # Higher is better

    print(
        f"Monitoring '{args.save_best_metric}' for saving the best model. Initial best score: {best_val_score}"
    )

    for epoch in range(args.num_epochs):
        model.train()  # Ensure model is in training mode for each epoch start
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs} [Training]",
            leave=False,
        )

        for batch_idx, (batch_images, batch_masks_gt) in enumerate(progress_bar):
            batch_images = batch_images.to(device)
            batch_masks_gt = batch_masks_gt.to(
                device
            )  # Shape: [B, C, H, W], C=1 for our masks from ToTensorV2

            # Ensure ground truth masks are float32 and have channel dim [B, 1, H, W]
            if batch_masks_gt.dtype == torch.long:
                batch_masks_gt = batch_masks_gt.float()

            if batch_masks_gt.ndim == 3:  # If masks are [B, H, W]
                batch_masks_gt = batch_masks_gt.unsqueeze(1)  # Convert to [B, 1, H, W]

            # SAM processor expects images to be in a list if they are PIL/numpy,
            # or a batch of tensors. Our batch_images are already [B, C, H, W] tensors.
            # The processor will handle resizing to SAM's expected input size (e.g., 1024x1024)
            # and normalization if not already done (though our augmentations do it).
            try:
                inputs = processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)  # Processor output
            except Exception as e:
                print(
                    f"Error during processor step at epoch {epoch+1}, batch {batch_idx}: {e}"
                )
                print(
                    f"Batch images shape: {batch_images.shape}, dtype: {batch_images.dtype}"
                )
                continue  # Skip batch if processor fails

            # Forward pass
            try:
                outputs = model(pixel_values=pixel_values, multimask_output=False)
                # ``outputs.pred_masks`` may come with additional singleton
                # dimensions depending on the model variant.
                predicted_masks_logits = outputs.pred_masks
                predicted_masks_logits = predicted_masks_logits.squeeze()
                if predicted_masks_logits.ndim == 3:
                    predicted_masks_logits = predicted_masks_logits.unsqueeze(1)
                elif predicted_masks_logits.ndim == 2:
                    predicted_masks_logits = predicted_masks_logits.unsqueeze(0).unsqueeze(0)
            except Exception as e:
                print(
                    f"Error during model forward pass at epoch {epoch+1}, batch {batch_idx}: {e}"
                )
                continue

            # Resize predicted masks to match ground truth mask size (args.image_size)
            # batch_masks_gt shape is [B, 1, args.image_size, args.image_size]
            # predicted_masks_logits shape is [B, 1, H_model_out, W_model_out]
            try:
                predicted_masks_resized = F.interpolate(
                    predicted_masks_logits,
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            except Exception as e:
                print(
                    f"Error during mask resizing at epoch {epoch+1}, batch {batch_idx}: {e}"
                )
                print(
                    f"Predicted_masks_logits shape: {predicted_masks_logits.shape}, GT masks shape: {batch_masks_gt.shape}"
                )
                continue

            # Loss calculation
            try:
                loss = criterion(predicted_masks_resized, batch_masks_gt)
            except Exception as e:
                print(
                    f"Error during loss calculation at epoch {epoch+1}, batch {batch_idx}: {e}"
                )
                print(
                    f"Resized predicted masks shape: {predicted_masks_resized.shape}, GT masks shape: {batch_masks_gt.shape}"
                )
                continue

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

        # Calculate average training loss for the epoch
        if len(train_dataloader) > 0:
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(
                f"Epoch {epoch+1}/{args.num_epochs} - Average Training Loss: {avg_epoch_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{args.num_epochs} - Training DataLoader was empty. No training loss."
            )
            avg_epoch_loss = 0.0

        # --- Validation Phase ---
        if val_dataloader:
            print(f"Epoch {epoch+1}/{args.num_epochs} - Starting Validation...")
            val_results = evaluate_model(
                model,
                val_dataloader,
                processor,
                criterion,
                device,
                args.image_size,
                args.eval_metrics,
            )

            log_str = f"Epoch {epoch+1} Validation Results: "
            for metric_name, metric_value in val_results.items():
                log_str += f"{metric_name.capitalize()}: {metric_value:.4f}  "
            print(log_str)

            # Saving the best model based on validation metric
            if args.save_best_metric not in val_results:
                print(
                    f"Warning: Metric '{args.save_best_metric}' not found in validation results. Available metrics: {list(val_results.keys())}. Skipping best model check for this epoch."
                )
            else:
                current_score = val_results[args.save_best_metric]

                save_condition_met = False
                if args.save_best_metric == "loss":  # Lower is better
                    if current_score < best_val_score:
                        best_val_score = current_score
                        save_condition_met = True
                else:  # Higher is better for 'accuracy', 'iou', 'dice'
                    if current_score > best_val_score:
                        best_val_score = current_score
                        save_condition_met = True

                if save_condition_met and args.output_dir:
                    best_model_path = os.path.join(args.output_dir, "best_model.pth")
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        print(
                            f"*** New best validation {args.save_best_metric}: {best_val_score:.4f}. Saved best model to {best_model_path} ***"
                        )
                    except Exception as e:
                        print(f"Error saving best model: {e}")
                elif save_condition_met:  # Condition met but no output_dir
                    print(
                        f"*** New best validation {args.save_best_metric}: {best_val_score:.4f}. (Output directory not specified, model not saved) ***"
                    )

        else:
            print(
                f"Epoch {epoch+1}/{args.num_epochs} - No validation performed as validation DataLoader is not available."
            )

        if scheduler:
            # Common practice: step scheduler based on validation metric (e.g., ReduceLROnPlateau) or after each epoch.
            # If it's ReduceLROnPlateau, it needs a metric: scheduler.step(avg_val_loss or val_results['loss'])
            # For StepLR or similar, just scheduler.step() is fine.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if (
                    val_dataloader and "loss" in val_results
                ):  # check if val_results has loss
                    scheduler.step(val_results["loss"])
                    print(
                        f"Epoch {epoch+1}: ReduceLROnPlateau scheduler step taken with val_loss: {val_results['loss']:.4f}."
                    )
                else:  # Cannot step if no val_loss is available
                    print(
                        f"Epoch {epoch+1}: ReduceLROnPlateau scheduler not stepped as validation loss is unavailable."
                    )
            else:  # For other schedulers like StepLR
                scheduler.step()

            # It's good practice to log the new LR
            # Note: get_lr() is deprecated. Use get_last_lr()
            if hasattr(optimizer, "param_groups"):
                new_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1}: Scheduler stepped. Current LR: {new_lr}")

    print("\nTraining finished.")

    # Save the final model
    if args.output_dir:
        final_model_path = os.path.join(args.output_dir, "final_model.pth")
        try:
            torch.save(model.state_dict(), final_model_path)
            print(f"Saved final model to {final_model_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")
    else:
        print("Output directory not specified, final model not saved.")

    # Placeholder for final evaluation on a test set if available
    # TODO: Implement evaluation logic here (Subtask 5 - this was for validation, test is separate)
    print(
        "\n7. Final evaluation (placeholder)..."
    )  # Renamed from "Starting evaluation"
    print("Final evaluation placeholder (e.g., on a held-out test set).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a Segment Anything Model (SAM) for a custom semantic segmentation task. "
        "This script handles dataset loading, augmentation, training, validation, and model saving."
    )

    # --- Argument Groups ---
    dataset_args = parser.add_argument_group("Dataset and DataLoader Configuration")
    model_args = parser.add_argument_group("Model Configuration")
    training_args = parser.add_argument_group("Training Configuration")
    val_save_args = parser.add_argument_group("Validation and Saving Configuration")

    # --- Dataset and DataLoader Configuration ---
    dataset_args.add_argument(
        "--dataset_path",
        type=str,
        default="data/output",
        help="Directory containing *_raw.png and *_mask.png image pairs for BalancedCropDataset. Default: data/output",
    )
    dataset_args.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="The target size (height and width) for images and masks during augmentation and training. "
        "SAM is pre-trained on 1024x1024, but smaller sizes like 256x256 can be used for fine-tuning with RandomResizedCrop. Default: 256",
    )
    dataset_args.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation DataLoaders. Adjust based on GPU memory. Default: 4",
    )
    dataset_args.add_argument(
        "--pos_fraction",
        type=float,
        default=0.5,
        help="Desired fraction of crops containing mask pixels in BalancedCropDataset. Default: 0.5",
    )
    dataset_args.add_argument(
        "--samples_per_epoch",
        type=int,
        default=1000,
        help="Number of crop samples to generate per training epoch. Default: 1000",
    )
    dataset_args.add_argument(
        "--num_workers",
        type=int,
        default=min(os.cpu_count(), 4) if os.cpu_count() else 2,  # type: ignore
        help="Number of worker processes for DataLoader. Defaults to min(CPU cores, 4) or 2 if CPU cores cannot be determined.",
    )

    # --- Model Configuration ---
    model_args.add_argument(
        "--model_name",
        type=str,
        default="facebook/sam-vit-base",
        help="Name of the SAM model to load from Hugging Face Model Hub (e.g., 'facebook/sam-vit-base', 'facebook/sam-vit-large', 'facebook/sam-vit-huge'). Default: facebook/sam-vit-base",
    )
    model_args.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training ('cuda' or 'cpu'). Defaults to 'cuda' if a GPU is available, otherwise 'cpu'.",
    )

    # --- Training Configuration ---
    training_args.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate for the optimizer. Fine-tuning SAM often benefits from small learning rates. Default: 1e-5",
    )
    training_args.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Type of optimizer to use. 'adamw' is generally recommended for transformer models. Default: adamw",
    )
    training_args.add_argument(
        "--loss_function_type",
        type=str,
        default="bcewithlogits",
        choices=["bcewithlogits", "dice"],
        help="Type of loss function. 'bcewithlogits' is standard for binary segmentation. 'dice' could be added for DiceLoss. Default: bcewithlogits",
    )
    training_args.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train for. Default: 10",
    )

    # --- Validation and Saving Configuration ---
    val_save_args.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.15,
        help="Proportion of the dataset to use for validation (e.g., 0.1 for 10% split). Default: 0.15",
    )
    val_save_args.add_argument(
        "--eval_metrics",
        nargs="+",
        default=["loss", "accuracy", "iou", "dice"],
        choices=["loss", "accuracy", "iou", "dice"],
        help="List of metrics to compute during evaluation. 'iou' is Jaccard Index, 'dice' is F1 Score. Default: ['loss', 'accuracy', 'iou', 'dice']",
    )
    val_save_args.add_argument(
        "--output_dir",
        type=str,
        default="models/sam_finetuned",
        help="Directory to save the best model checkpoint and the final model. Default: models/sam_finetuned",
    )
    val_save_args.add_argument(
        "--save_best_metric",
        type=str,
        default="iou",
        choices=["loss", "accuracy", "iou", "dice"],
        help="Validation metric to monitor for saving the best model. 'loss' means lower is better, others higher is better. Default: iou",
    )

    args = parser.parse_args()

    main(args)

    print("\nFinetuning script finished.")
