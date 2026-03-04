from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from prithvi2.utils import ensure_dir


def infer_rgb_indices(bands: Iterable[str]) -> List[int]:
    band_list = [band.upper() for band in bands]
    try:
        red = band_list.index("RED")
        green = band_list.index("GREEN")
        blue = band_list.index("BLUE")
        return [red, green, blue]
    except ValueError:
        if len(band_list) >= 3:
            return [0, 1, 2]
        return [0, 0, 0]


def _select_rgb(image: torch.Tensor, rgb_indices: Optional[List[int]]) -> torch.Tensor:
    if rgb_indices and len(rgb_indices) == 3:
        return image[rgb_indices]
    if image.size(0) >= 3:
        return image[:3]
    return image.repeat(3, 1, 1)


def _compute_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) == 1:
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=1)[:, 1:2]
    return probs


def _overlay_segmentation(image: torch.Tensor, pred_mask: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    overlay = image.clone()
    overlay[1] = torch.clamp(overlay[1] + pred_mask[0], 0.0, 1.0)
    overlay[0] = torch.clamp(overlay[0] + mask[0], 0.0, 1.0)
    overlay[2] = torch.clamp(overlay[2] + mask[0], 0.0, 1.0)
    return overlay


def save_visuals_from_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    out_dir: Path,
    epoch: int,
    total_target: int,
    pos_target: int,
    neg_target: int,
    threshold: float = 0.5,
    rgb_indices: Optional[List[int]] = None,
) -> None:
    if total_target <= 0:
        return

    ensure_dir(out_dir)
    out_epoch_dir = out_dir / f"epoch{epoch}"
    ensure_dir(out_epoch_dir)

    pos_saved = 0
    neg_saved = 0
    total_saved = 0

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            logits = model(images).output
            probs = _compute_probabilities(logits).detach()

            probs_cpu = probs.cpu()
            masks_cpu = masks.cpu().float()
            images_cpu = images.cpu().float().clamp(0.0, 1.0)

            mask_area = masks_cpu.view(masks_cpu.size(0), -1).sum(dim=1)
            for i in range(images_cpu.size(0)):
                if total_saved >= total_target:
                    if was_training:
                        model.train()
                    return

                is_positive = mask_area[i].item() > 0.5
                if is_positive and pos_saved >= pos_target:
                    continue
                if not is_positive and neg_saved >= neg_target:
                    continue

                rgb = _select_rgb(images_cpu[i], rgb_indices)
                prob = probs_cpu[i]
                mask = masks_cpu[i]
                pred_bin = (prob >= threshold).float()
                overlay = _overlay_segmentation(rgb, pred_bin, mask)
                grid = torch.cat(
                    [
                        rgb,
                        mask.repeat(3, 1, 1),
                        prob.repeat(3, 1, 1),
                        overlay,
                    ],
                    dim=2,
                )

                prefix = "pos" if is_positive else "neg"
                fname = out_epoch_dir / f"{prefix}_{total_saved:05d}.png"
                save_image(grid, fname)

                total_saved += 1
                if is_positive:
                    pos_saved += 1
                else:
                    neg_saved += 1

    if was_training:
        model.train()
