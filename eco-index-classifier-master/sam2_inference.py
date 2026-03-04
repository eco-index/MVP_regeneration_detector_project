# sam2_inference.py
import argparse
import os
import importlib.util
from pathlib import Path

import pandas as pd
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2

IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_point_grid(n_per_side: int) -> np.ndarray:
    """Generate an evenly spaced grid of points within the 1024x1024 image."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points * IMAGE_SIZE


def load_model(weights_path: str):
    """Load the fine-tuned SAM2 model for inference."""
    pkg_root = Path(importlib.util.find_spec("sam2").origin).parent
    cfg_file = pkg_root / "configs/sam2.1/sam2.1_hiera_t.yaml"
    ckpt = pkg_root / "../checkpoints/sam2.1_hiera_tiny.pt"

    model = build_sam2(
        config_file="/" + str(cfg_file.resolve()),
        ckpt_path="/" + str(ckpt.resolve()),
        device=DEVICE,
        mode="eval",
    )
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()
    return model


def find_mask_path(raw_image: Path) -> Path:
    """Locate the ground truth mask path using the pairs.csv file."""
    dataset_dir = raw_image.parent.parent  # e.g., dataset/train
    csv_path = dataset_dir / "pairs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"pairs.csv not found in {dataset_dir}")

    rel_raw = os.path.relpath(raw_image, dataset_dir)
    df = pd.read_csv(csv_path)
    match = df[df["raw_path"] == rel_raw]
    if match.empty:
        raise ValueError(f"{rel_raw} not found in {csv_path}")
    mask_rel = match.iloc[0]["mask_path"]
    return dataset_dir / mask_rel


def run_inference(model, img_path: Path, mask_path: Path, grid_pts: np.ndarray, out_path: Path):
    """Run model inference and save a visualization quad."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask {mask_path}")
    mask = (mask > 128).astype(np.uint8)

    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    image_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mask_t = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float()

    image_t = image_t.to(DEVICE)
    mask_t = mask_t.to(DEVICE)

    grid_points_t = torch.from_numpy(grid_pts).float().unsqueeze(0).to(DEVICE)
    point_labels = torch.ones((1, grid_pts.shape[0]), dtype=torch.long, device=DEVICE)

    bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    with torch.no_grad():
        backbone_out = model.forward_image(image_t)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *size)
            for feat, size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        point_inputs = {"point_coords": grid_points_t, "point_labels": point_labels}
        _, _, _, _, mask_logits, _, _ = model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            high_res_features=high_res_feats,
            multimask_output=False,
        )

        merged_logits, _ = torch.max(mask_logits, dim=1)
        pred_mask = torch.sigmoid(merged_logits)

    img_np = (image_t[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_np = (mask_t[0, 0].cpu().numpy() * 255).astype(np.uint8)
    pred_np = (pred_mask[0].cpu().numpy() * 255).astype(np.uint8)

    overlay = img_np.copy()
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + pred_np, 0, 255)

    gt_color = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2RGB)
    pred_color = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2RGB)

    top = np.concatenate([img_np, gt_color], axis=1)
    bottom = np.concatenate([pred_color, overlay], axis=1)
    quad = np.concatenate([top, bottom], axis=0)

    cv2.imwrite(str(out_path), cv2.cvtColor(quad, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run SAM2 inference on a dataset image and save a visualization quad")
    ap.add_argument("--image", required=True, help="Path to a raw image from the dataset")
    ap.add_argument("--weights", required=True, help="Path to fine-tuned model weights (.pt)")
    ap.add_argument("--grid_points", type=int, default=32, help="Number of grid points per side")
    ap.add_argument("--output", default="inference_output.png", help="Path to save the output quad image")
    args = ap.parse_args()

    img_path = Path(args.image)
    model = load_model(args.weights)
    mask_path = find_mask_path(img_path)
    grid = generate_point_grid(args.grid_points)

    run_inference(model, img_path, mask_path, grid, Path(args.output))
    print(f"Saved output to {args.output}")
