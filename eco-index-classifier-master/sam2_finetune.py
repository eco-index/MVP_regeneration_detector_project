# sam2_finetune.py
import argparse
import importlib.util
import os
import uuid
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2
from sam2_dataset import PairDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024 # SAM2 is trained on 1024x1024 images
LOSS_WEIGHTS = {"loss_mask": 20.0, "loss_dice": 1.0, "loss_iou": 1.0, "loss_class": 1.0}


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss / num_objects

    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_objects


def sigmoid_focal_loss(inputs, targets, num_objects, alpha=0.25, gamma=2, loss_on_multimask=False):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects

    # flatten across all but batch dim when needed
    if loss.dim() > 2:
        loss = loss.flatten(1)
    elif loss.dim() == 1:
        loss = loss.unsqueeze(1)

    return loss.mean(1).sum() / num_objects


def iou_loss(inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False):
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")

    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def compute_sam2_loss(
    pred_masks,
    gt_masks,
    pred_ious,
    object_score_logits=None,
    loss_weights=None,
    click_counts=None,
):
    """Reproduce the SAM2 training loss (focal + dice + IoU + object score)."""
    if loss_weights is None:
        loss_weights = LOSS_WEIGHTS

    if not isinstance(pred_masks, (list, tuple)):
        pred_masks = [pred_masks]
    if not isinstance(pred_ious, (list, tuple)):
        pred_ious = [pred_ious]
    if object_score_logits is None:
        object_score_logits = [None] * len(pred_masks)
    elif not isinstance(object_score_logits, (list, tuple)):
        object_score_logits = [object_score_logits]

    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)

    B = gt_masks.size(0)
    device = gt_masks.device
    if click_counts is None:
        click_counts = torch.full((B,), len(pred_masks), dtype=torch.long, device=device)
    else:
        click_counts = click_counts.to(device)

    obj_present = (gt_masks.sum(dim=(2, 3)) > 0).float()
    num_objects = max(float(obj_present.sum().item()), 1.0)

    weight_mask = loss_weights.get("loss_mask", 1.0)
    weight_dice = loss_weights.get("loss_dice", 1.0)
    weight_iou = loss_weights.get("loss_iou", 0.0)
    weight_class = loss_weights.get("loss_class", 0.0)

    total_loss = gt_masks.new_tensor(0.0)
    total_mask = gt_masks.new_tensor(0.0)
    total_dice = gt_masks.new_tensor(0.0)
    total_iou = gt_masks.new_tensor(0.0)
    total_class = gt_masks.new_tensor(0.0)
    normalizer = gt_masks.new_tensor(0.0)

    for step_idx, (pred_mask_step, pred_iou_step, obj_logit_step) in enumerate(
        zip(pred_masks, pred_ious, object_score_logits)
    ):
        if pred_mask_step.dim() == 3:
            pred_mask_step = pred_mask_step.unsqueeze(1)
        target_masks = gt_masks.float().expand_as(pred_mask_step)
        pred_iou_step = pred_iou_step.float()

        valid_mask = (click_counts > step_idx).float().unsqueeze(1)
        if valid_mask.sum() <= 0:
            continue

        loss_mask_all = sigmoid_focal_loss(
            pred_mask_step,
            target_masks,
            num_objects,
            alpha=0.25,
            gamma=2.0,
            loss_on_multimask=True,
        )
        loss_dice_all = dice_loss(
            pred_mask_step,
            target_masks,
            num_objects,
            loss_on_multimask=True,
        )
        loss_iou_all = iou_loss(
            pred_mask_step,
            target_masks,
            pred_iou_step,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=True,
        )

        multiplier = obj_present.to(loss_mask_all.dtype)
        if multiplier.dim() == 1:
            multiplier = multiplier.unsqueeze(1)
        mask_multiplier = multiplier * valid_mask.to(multiplier.dtype)
        if mask_multiplier.size(1) != loss_mask_all.size(1):
            mask_multiplier = mask_multiplier.expand(-1, loss_mask_all.size(1))

        if loss_mask_all.size(1) > 1:
            combo = loss_mask_all * weight_mask + loss_dice_all * weight_dice
            best_idx = torch.argmin(combo, dim=1)
            batch_idx = torch.arange(combo.size(0), device=combo.device)
            loss_mask_all = loss_mask_all[batch_idx, best_idx].unsqueeze(1)
            loss_dice_all = loss_dice_all[batch_idx, best_idx].unsqueeze(1)
            loss_iou_all = loss_iou_all[batch_idx, best_idx].unsqueeze(1)

        step_mask = (loss_mask_all * mask_multiplier).sum()
        step_dice = (loss_dice_all * mask_multiplier).sum()
        step_iou = (loss_iou_all * mask_multiplier).sum()

        total_mask = total_mask + step_mask
        total_dice = total_dice + step_dice
        total_iou = total_iou + step_iou

        if obj_logit_step is not None and weight_class != 0.0:
            logits = obj_logit_step.float()
            target_obj = multiplier[:, : logits.size(1)].to(logits.dtype)
            prob = torch.sigmoid(logits)
            ce_loss = F.binary_cross_entropy_with_logits(logits, target_obj, reduction="none")
            p_t = prob * target_obj + (1 - prob) * (1 - target_obj)
            focal = ce_loss * ((1 - p_t) ** 0.0)  # gamma = 0
            total_class = total_class + (focal * target_obj).sum() / max(num_objects, 1.0)

        normalizer = normalizer + mask_multiplier.sum()

    if normalizer.item() == 0:
        normalizer = gt_masks.new_tensor(1.0)

    total_loss = (
        weight_mask * total_mask
        + weight_dice * total_dice
        + weight_iou * total_iou
        + weight_class * total_class
    ) / normalizer

    return {
        "loss_total": total_loss,
        "loss_mask": total_mask / normalizer,
        "loss_dice": total_dice / normalizer,
        "loss_iou": total_iou / normalizer,
        "loss_class": total_class / normalizer,
    }


def sample_click_sequences(masks, max_clicks, device):
    """Sample alternating positive/negative click sequences for a batch."""
    masks_np = masks.detach().cpu().numpy()
    B, _, H, W = masks_np.shape
    coords = torch.zeros(B, max_clicks, 2, device=device, dtype=torch.float32)
    labels = torch.full((B, max_clicks), -1, dtype=torch.long, device=device)
    counts = torch.zeros(B, dtype=torch.long, device=device)

    rng = np.random.default_rng()
    for i in range(B):
        pos_idx = np.argwhere(masks_np[i, 0] > 0.5)
        neg_idx = np.argwhere(masks_np[i, 0] <= 0.5)
        if pos_idx.size == 0 and neg_idx.size == 0:
            continue

        for step in range(max_clicks):
            desired_positive = (step % 2 == 0)
            candidate = pos_idx if desired_positive else neg_idx
            fallback = neg_idx if desired_positive else pos_idx
            pool = candidate if candidate.size else fallback
            if pool.size == 0:
                break
            idx = pool[rng.integers(len(pool))]
            y, x = int(idx[0]), int(idx[1])
            coords[i, step, 0] = float(x) + 0.5
            coords[i, step, 1] = float(y) + 0.5
            labels[i, step] = 1 if masks_np[i, 0, y, x] > 0.5 else 0
            counts[i] += 1

    # Ensure every sample has at least one click to avoid empty prompt lists.
    empty = counts == 0
    if empty.any():
        for idx in torch.nonzero(empty, as_tuple=False).squeeze(1):
            y, x = H // 2, W // 2
            coords[idx, 0, 0] = float(x) + 0.5
            coords[idx, 0, 1] = float(y) + 0.5
            labels[idx, 0] = 1 if masks_np[idx, 0, y, x] > 0.5 else 0
            counts[idx] = 1

    return coords, labels, counts


def run_multistep_sam_forward(
    model,
    image_embed,
    high_res_feats,
    point_coords,
    point_labels,
    click_counts,
    multimask_output=True,
):
    """Iteratively apply SAM2 heads for a sequence of clicks."""
    outputs = {
        "multistep_pred_multimasks_high_res": [],
        "multistep_pred_ious": [],
        "multistep_object_score_logits": [],
        "multistep_high_res_best_masks": [],
    }

    max_steps = int(click_counts.max().item())
    last_best_masks = None
    for step_idx in range(max_steps):
        step_coords = point_coords[:, : step_idx + 1, :]
        step_labels = point_labels[:, : step_idx + 1]
        out = model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs={"point_coords": step_coords, "point_labels": step_labels},
            high_res_features=high_res_feats,
            multimask_output=multimask_output,
        )
        (
            _,
            high_res_multimasks,
            ious,
            _,
            high_res_masks,
            _,
            object_score_logits,
        ) = out
        outputs["multistep_pred_multimasks_high_res"].append(high_res_multimasks)
        outputs["multistep_pred_ious"].append(ious)
        outputs["multistep_object_score_logits"].append(object_score_logits)
        outputs["multistep_high_res_best_masks"].append(high_res_masks)
        last_best_masks = high_res_masks

    return outputs, last_best_masks

def evaluate(model, dataloader, device, num_examples=0, example_dir=None, max_clicks=4):
    """Run validation and optionally save prediction examples."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    if num_examples and example_dir:
        os.makedirs(example_dir, exist_ok=True)

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            images = batch["image"].permute(0, 3, 1, 2).float() / 255.0
            masks = batch["mask"].unsqueeze(1).float()
            
            # Resize and move to device
            images = torch.nn.functional.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).to(device)
            masks = torch.nn.functional.interpolate(masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").to(device)

            bs = images.size(0)

            # Forward pass
            backbone_out = model.forward_image(images)
            _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
            if model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(bs, -1, *size)
                for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
            ][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            point_coords, point_labels, click_counts = sample_click_sequences(
                masks, max_clicks=max_clicks, device=device
            )
            multistep_outputs, best_masks = run_multistep_sam_forward(
                model,
                image_embed,
                high_res_feats,
                point_coords,
                point_labels,
                click_counts,
                multimask_output=True,
            )

            losses = compute_sam2_loss(
                multistep_outputs["multistep_pred_multimasks_high_res"],
                masks,
                multistep_outputs["multistep_pred_ious"],
                multistep_outputs["multistep_object_score_logits"],
                click_counts=click_counts,
            )
            loss = losses["loss_total"]

            prd_mask = torch.sigmoid(best_masks[:, 0])

            gt_fg = (masks[:, 0] > 0)
            pred_fg = prd_mask > 0.5
            inter = (gt_fg & pred_fg).sum((1, 2)).float()
            union = (gt_fg.sum((1, 2)) + pred_fg.sum((1, 2)) - inter).float()

            iou = torch.zeros(bs, device=device)
            non_empty = union > 0
            iou[non_empty] = inter[non_empty] / (union[non_empty] + 1e-6)
            iou[~non_empty] = 1.0  # perfect score when both prediction and GT are empty

            total_loss += loss.item()
            total_iou += iou.mean().item()
            count += 1

            # Save visualization logic (mostly unchanged)
            if num_examples and b_idx < num_examples and example_dir:
                for i in range(images.size(0)):
                    img = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    gt = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                    pred = (prd_mask[i].cpu().numpy() * 255).astype(np.uint8)

                    overlay = img.copy()
                    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + pred, 0, 255) # Use green for prediction

                    gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
                    pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

                    top = np.concatenate([img, gt_color], axis=1)
                    bottom = np.concatenate([pred_color, overlay], axis=1)
                    quad = np.concatenate([top, bottom], axis=0)

                    cv2.imwrite(os.path.join(example_dir, f"{b_idx}_{i}.png"), cv2.cvtColor(quad, cv2.COLOR_RGB2BGR))

    model.train()
    return total_loss / max(count, 1), total_iou / max(count, 1)


def main(cfg):
    # Setup
    run_id = cfg.run_id or uuid.uuid4().hex[:6]
    print("Run ID", run_id)
    checkpoint_out_dir = Path(f"runs/{run_id}/checkpoints")
    validation_examples_out_dir = Path(f"runs/{run_id}/validation_examples")
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    validation_examples_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-trained model
    pkg_root = Path(importlib.util.find_spec("sam2").origin).parent
    cfg_file = pkg_root / "configs/sam2.1/sam2.1_hiera_t.yaml"
    ckpt = pkg_root / "../checkpoints/sam2.1_hiera_tiny.pt"

    sam_net = build_sam2(
        config_file="/" + str(cfg_file.resolve()),
        ckpt_path="/" + str(ckpt.resolve()),
        device=DEVICE,
        mode="train",
    )
    for p in sam_net.parameters():
        p.requires_grad = True

    if getattr(cfg, "train_decoder_only", False):
        for name, param in sam_net.named_parameters():
            param.requires_grad = any(k in name for k in ["prompt_encoder", "mask_decoder"])
    elif cfg.freeze_image_encoder:
        for name, param in sam_net.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = False

    sam_net.train()

    trainable_params = [p for p in sam_net.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Adjust freezing options.")

    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # 3. Prepare data
    train_dataset = PairDataset(
        cfg.train_csv,
        os.path.dirname(cfg.train_csv),
        img_size=IMAGE_SIZE,
        augment=cfg.augment,
    )
    dl = DataLoader(
        train_dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_dl = None
    if cfg.test_csv and os.path.exists(cfg.test_csv):
        val_dataset = PairDataset(
            cfg.test_csv,
            os.path.dirname(cfg.test_csv),
            img_size=IMAGE_SIZE,
            augment=False,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=cfg.bs,
            shuffle=False,
            num_workers=max(1, cfg.num_workers // 2),
            pin_memory=True,
        )

    scheduler = None
    if cfg.use_scheduler:
        total_steps = max(len(dl) * cfg.epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=cfg.lr * 0.1
        )

    use_autocast = DEVICE.startswith("cuda")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        component_totals = {"loss_mask": 0.0, "loss_dice": 0.0, "loss_iou": 0.0, "loss_class": 0.0}
        sam_net.train()
        for step, batch in enumerate(dl):
            images = batch["image"].permute(0, 3, 1, 2).float() / 255.0
            masks = batch["mask"].unsqueeze(1).float()

            images = torch.nn.functional.interpolate(
                images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
            ).to(DEVICE)
            masks = torch.nn.functional.interpolate(
                masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest"
            ).to(DEVICE)

            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_autocast
                else nullcontext()
            )
            with ctx:
                backbone_out = sam_net.forward_image(images)
                _, vision_feats, _, feat_sizes = sam_net._prepare_backbone_features(backbone_out)
                if sam_net.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + sam_net.no_mem_embed

                feats = [
                    feat.permute(1, 2, 0).view(images.size(0), -1, *size)
                    for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
                ][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                point_coords, point_labels, click_counts = sample_click_sequences(
                    masks, max_clicks=cfg.max_clicks, device=images.device
                )
                multistep_outputs, _ = run_multistep_sam_forward(
                    sam_net,
                    image_embed,
                    high_res_feats,
                    point_coords,
                    point_labels,
                    click_counts,
                    multimask_output=True,
                )

                loss_dict = compute_sam2_loss(
                    multistep_outputs["multistep_pred_multimasks_high_res"],
                    masks,
                    multistep_outputs["multistep_pred_ious"],
                    multistep_outputs["multistep_object_score_logits"],
                    click_counts=click_counts,
                )
                loss = loss_dict["loss_total"]

            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            for key in component_totals:
                component_totals[key] += loss_dict[key].item()

            if step % cfg.log_interval == 0:
                lr = optim.param_groups[0]["lr"]
                print(
                    f"ep{epoch} step{step}/{len(dl)} loss={loss.item():.4f} "
                    f"mask={loss_dict['loss_mask'].item():.4f} dice={loss_dict['loss_dice'].item():.4f} "
                    f"iou={loss_dict['loss_iou'].item():.4f} lr={lr:.2e}"
                )

        denom = max(len(dl), 1)
        train_loss = epoch_loss / denom
        avg_components = {k: v / denom for k, v in component_totals.items()}

        # Evaluation
        val_loss, val_iou = 0.0, 0.0
        if val_dl:
            example_dir = (
                validation_examples_out_dir / f"epoch{epoch}"
                if cfg.examples > 0
                else None
            )
            val_loss, val_iou = evaluate(
                sam_net,
                val_dl,
                DEVICE,
                num_examples=cfg.examples,
                example_dir=example_dir,
                max_clicks=cfg.max_clicks,
            )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"(mask={avg_components['loss_mask']:.4f}, dice={avg_components['loss_dice']:.4f}, iou={avg_components['loss_iou']:.4f}) "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        torch.save(sam_net.state_dict(), checkpoint_out_dir / f"sam2_finetuned_ep{epoch}.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="dataset/train/pairs.csv")
    ap.add_argument("--test_csv", default="dataset/test/pairs.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--examples", type=int, default=5, help="Number of validation examples to save per epoch")
    ap.add_argument("--run_id", default=None, help="Optional run ID for saving outputs")
    ap.add_argument("--max_clicks", type=int, default=4, help="Maximum number of simulated clicks per image")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--freeze_image_encoder", action="store_true", help="Freeze the image encoder during finetuning")
    ap.add_argument("--train_decoder_only", action="store_true", help="Train only the prompt encoder and mask decoder")
    ap.add_argument("--no_augment", action="store_true", help="Disable data augmentation for training")
    ap.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing learning-rate scheduler")
    cfg = ap.parse_args()
    cfg.augment = not cfg.no_augment
    main(cfg)
