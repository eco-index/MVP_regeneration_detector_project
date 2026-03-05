import random
from typing import List, Tuple, Optional
import pathlib

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from scripts.prepare_segmentation_dataset import find_image_pairs, generate_binary_mask


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


if __name__ == "__main__":
    print("BalancedCropDataset demonstration")
    from torch.utils.data import DataLoader

    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.Affine(scale=(0.8, 1.2), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, p=0.5),
        ToTensorV2(),
    ])

    dataset = BalancedCropDataset(
        data_dir="data/output",
        patch_size=256,
        pos_fraction=0.5,
        augmentations=augment,
        samples_per_epoch=10,
    )

    loader = DataLoader(dataset, batch_size=2)
    for imgs, masks in loader:
        print(imgs.shape, masks.shape)
        break

