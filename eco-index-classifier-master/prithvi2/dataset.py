from __future__ import annotations

import os
from typing import List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


def _build_transforms(image_size: int, augment: bool) -> A.Compose:
    transforms: List[A.BasicTransform] = []
    if augment:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.Resize(image_size, image_size))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


class Prithvi2Dataset(Dataset):
    """CSV-driven dataset for Prithvi v2 finetuning with optional augmentations."""

    def __init__(
        self,
        csv_path: str,
        root_dir: Optional[str] = None,
        image_size: int = 224,
        augment: bool = False,
        bands: Optional[List[str]] = None,
        band_mapping: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir or os.path.dirname(csv_path)
        self.image_size = image_size
        self.augment = augment
        self.bands = bands or ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
        self.band_mapping = band_mapping or [2, 1, 0, 0, 1, 2]
        if len(self.band_mapping) != len(self.bands):
            raise ValueError(
                "band_mapping length must match number of bands "
                f"({len(self.band_mapping)} vs {len(self.bands)})"
            )
        self.transform = _build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> str:
        path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file: {path}")
        return path

    def _expand_bands(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
        if any(idx < 0 or idx >= image.shape[2] for idx in self.band_mapping):
            raise ValueError(f"band_mapping indices must be within [0, {image.shape[2] - 1}]")
        bands = [image[..., idx] for idx in self.band_mapping]
        return np.stack(bands, axis=-1)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = self._resolve_path(row["raw_path"])
        mask_path = self._resolve_path(row["mask_path"])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)

        image = self._expand_bands(image)

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"].float() / 255.0
        mask_tensor = transformed["mask"].long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }


class Prithvi2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: Optional[str] = None,
        root_dir: Optional[str] = None,
        image_size: int = 224,
        batch_size: int = 16,
        num_workers: int = 4,
        bands: Optional[List[str]] = None,
        band_mapping: Optional[List[int]] = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bands = bands
        self.band_mapping = band_mapping
        self.augment = augment

        self.train_dataset: Optional[Prithvi2Dataset] = None
        self.val_dataset: Optional[Prithvi2Dataset] = None
        self.test_dataset: Optional[Prithvi2Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Prithvi2Dataset(
                self.train_csv,
                root_dir=self.root_dir,
                image_size=self.image_size,
                augment=self.augment,
                bands=self.bands,
                band_mapping=self.band_mapping,
            )
            self.val_dataset = Prithvi2Dataset(
                self.val_csv,
                root_dir=self.root_dir,
                image_size=self.image_size,
                augment=False,
                bands=self.bands,
                band_mapping=self.band_mapping,
            )
        if stage in (None, "test") and self.test_csv:
            self.test_dataset = Prithvi2Dataset(
                self.test_csv,
                root_dir=self.root_dir,
                image_size=self.image_size,
                augment=False,
                bands=self.bands,
                band_mapping=self.band_mapping,
            )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup('fit') first.")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup('fit') first.")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Call setup('test') first.")
        return self._loader(self.test_dataset, shuffle=False)
