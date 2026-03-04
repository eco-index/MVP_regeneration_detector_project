import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F


class PrithviDataset(Dataset):
    """Dataset loader for Prithvi finetuning with optional augmentations."""

    def __init__(
        self,
        csv_path: str,
        root_dir: str = None,
        image_size: int = 512,
        augment: bool = False,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir or os.path.dirname(csv_path)
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> str:
        path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file: {path}")
        return path

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image, dict]:
        row = self.df.iloc[idx]
        image_path = self._resolve_path(row["raw_path"])
        mask_path = self._resolve_path(row["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        meta = {
            "image_path": image_path,
            "mask_path": mask_path,
            "index": idx,
        }
        return image, mask, meta

    def _apply_augments(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random vertical flip
        if random.random() < 0.3:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Random rotation by multiples of 90 degrees
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])
            angle = k * 90
            image = F.rotate(image, angle, interpolation=Image.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=Image.NEAREST)

        # Color jitter on image only
        image = self.color_jitter(image)

        return image, mask

    def _prepare(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize keeping divisibility by 16
        image = F.resize(image, (self.image_size, self.image_size), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (self.image_size, self.image_size), interpolation=Image.NEAREST)

        image_tensor = F.to_tensor(image)  # scales to [0,1]
        mask_array = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_array > 127).astype("float32")).unsqueeze(0)

        return image_tensor, mask_tensor

    def __getitem__(self, idx: int):
        image, mask, meta = self._load_pair(idx)

        if self.augment:
            image, mask = self._apply_augments(image, mask)

        pixel_values, mask_tensor = self._prepare(image, mask)

        return {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "index": idx,
            "image_path": meta["image_path"],
            "mask_path": meta["mask_path"],
        }
