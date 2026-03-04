from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F


def sigmoid_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * torch.sum(probs * targets) + eps
    den = torch.sum(probs) + torch.sum(targets) + eps
    dice = 1 - num / den
    return dice


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds_bin = preds > 0.5
    targets_bin = targets > 0.5
    inter = torch.sum(preds_bin & targets_bin, dim=(1, 2))
    union = torch.sum(preds_bin | targets_bin, dim=(1, 2))
    iou = (inter + eps) / (union + eps)
    return iou


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def overlay_segmentation(image: torch.Tensor, prob_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Create overlay: combine prediction (green) and GT (magenta)."""
    import numpy as np

    image_np = image.cpu().numpy()
    prob_np = prob_map.cpu().numpy()
    mask_np = mask.cpu().numpy()

    pred_bin = (prob_np > 0.5).astype(np.float32)
    mask_bin = mask_np.astype(np.float32)

    overlay = image_np.copy()
    overlay = overlay.transpose(1, 2, 0)

    overlay[..., 1] = np.clip(overlay[..., 1] + pred_bin, 0, 1)  # green
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_bin, 0, 1)  # red channel for GT -> magenta with blue
    overlay[..., 2] = np.clip(overlay[..., 2] + mask_bin, 0, 1)

    return torch.from_numpy(overlay.transpose(2, 0, 1)).float()
