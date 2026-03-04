# sam2_dataset.py
import os
import random

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """
    Returns dict with:
        image      → uint8   [H,W,C] ∈ [0,255]
        mask       → uint8   [H,W]   (binary 0/1)
    """

    def __init__(self, csv_file, root_dir, limit=None, img_size=1024, augment=False):
        df = pd.read_csv(csv_file)
        self.entries = df.to_dict("records")[:limit] if limit else df.to_dict("records")
        self.root = root_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        img_path = os.path.join(self.root, row["raw_path"])
        mask_path = os.path.join(self.root, row["mask_path"])
        
        # Use cv2.IMREAD_COLOR for the image to ensure 3 channels
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use cv2.IMREAD_GRAYSCALE for the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask: {mask_path}")

        # --- No resizing here, we will do it consistently in the training script ---

        mask = (mask > 128).astype(np.uint8) # Binarize the mask to 0s and 1s

        if self.augment:
            img, mask = self._augment(img, mask)

        return {
            "image": img,
            "mask": mask,
        }

    def _augment(self, img, mask):
        """Apply paired geometric/photometric transforms."""
        # All ops work on numpy arrays; ensure copies so OpenCV doesn't share memory unexpectedly.
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.2:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        if random.random() < 0.2:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Random brightness/contrast jitter
        if random.random() < 0.5:
            alpha = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            beta = random.randint(-20, 20)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Optional blur to reduce overfitting to sharp edges
        if random.random() < 0.1:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img, mask
