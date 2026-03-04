import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prithvi.dataset import PrithviDataset
from prithvi.model import PrithviSegmentationModel
from prithvi.utils import ensure_dir, overlay_segmentation, sigmoid_dice_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Prithvi for native planting segmentation")
    parser.add_argument("--train_csv", default="dataset/train/pairs.csv")
    parser.add_argument("--val_csv", default="dataset/test/pairs.csv")
    parser.add_argument("--root_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--checkpoint_dir", default="runs")
    parser.add_argument("--save_visuals", type=int, default=40)
    parser.add_argument("--visuals_pos", type=int, default=20)
    parser.add_argument("--visuals_neg", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    return parser.parse_args()


def create_dataloaders(args: argparse.Namespace):
    train_dataset = PrithviDataset(
        args.train_csv,
        root_dir=os.path.dirname(args.train_csv),
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = PrithviDataset(
        args.val_csv,
        root_dir=os.path.dirname(args.val_csv),
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace) -> PrithviSegmentationModel:
    model = PrithviSegmentationModel()
    return model


def set_encoder_trainable(model: PrithviSegmentationModel, requires_grad: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


def split_parameters(model: torch.nn.Module, encoder_lr: float, decoder_lr: float):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decoder_params, "lr": decoder_lr},
    ]


def compute_metrics(logits: torch.Tensor, masks: torch.Tensor):
    probs = torch.sigmoid(logits)
    loss_bce = F.binary_cross_entropy_with_logits(logits, masks)
    loss_dice = sigmoid_dice_loss(logits, masks)
    loss = loss_bce + loss_dice

    preds = (probs > 0.5).float()
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks - preds * masks).sum(dim=(1, 2, 3))
    iou = (inter + 1e-6) / (union + 1e-6)

    pos_mask = (masks.sum(dim=(1, 2, 3)) > 0)
    if pos_mask.any():
        pos_iou = iou[pos_mask].mean().item()
    else:
        pos_iou = float("nan")

    neg_mask = ~pos_mask
    if neg_mask.any():
        neg_iou = iou[neg_mask].mean().item()
    else:
        neg_iou = float("nan")

    return loss, loss_bce.item(), loss_dice.item(), pos_iou, neg_iou


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.checkpoint_dir) / f"prithvi_{run_id}"
    checkpoint_dir = run_root / "checkpoints"
    visuals_dir = run_root / "val_examples"
    ensure_dir(checkpoint_dir)
    ensure_dir(visuals_dir)

    print(f"Run directory: {run_root}")

    train_loader, val_loader = create_dataloaders(args)
    model = build_model(args).to(device)
    if args.freeze_epochs > 0:
        set_encoder_trainable(model, False)


    params = split_parameters(model, args.encoder_lr, args.lr)
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            masks = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(pixel_values)
                loss, loss_bce, loss_dice, _, _ = compute_metrics(logits, masks)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_bce += loss_bce
            epoch_dice += loss_dice

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = epoch_loss / max(1, len(train_loader))
        train_bce = epoch_bce / max(1, len(train_loader))
        train_dice = epoch_dice / max(1, len(train_loader))

        val_loss, val_bce, val_dice, val_pos_iou, val_neg_iou = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} (bce={train_bce:.4f} dice={train_dice:.4f}) "
            f"val_loss={val_loss:.4f} posIoU={val_pos_iou:.4f} negIoU={val_neg_iou:.4f}"
        )
        if device.type == "cuda":
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"  peak GPU memory: {max_mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"epoch{epoch}.pt"
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

        if val_pos_iou > best_val_iou:
            best_val_iou = val_pos_iou
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, checkpoint_dir / "best.pt")

        # Save validation visualizations
        save_visuals_from_loader(
            model,
            val_loader,
            device,
            visuals_dir,
            epoch,
            total_target=args.save_visuals,
            pos_target=args.visuals_pos,
            neg_target=args.visuals_neg,
        )

        if args.freeze_epochs > 0 and epoch + 1 == args.freeze_epochs:
            set_encoder_trainable(model, True)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    pos_ious = []
    neg_ious = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            masks = batch["mask"].to(device)
            logits = model(pixel_values)
            loss, loss_bce, loss_dice, pos_iou, neg_iou = compute_metrics(logits, masks)

            total_loss += loss.item()
            total_bce += loss_bce
            total_dice += loss_dice

            if not np.isnan(pos_iou):
                pos_ious.append(pos_iou)
            if not np.isnan(neg_iou):
                neg_ious.append(neg_iou)

    denom = max(1, len(dataloader))
    mean_pos_iou = float(np.mean(pos_ious)) if pos_ious else 0.0
    mean_neg_iou = float(np.mean(neg_ious)) if neg_ious else 0.0
    return (
        total_loss / denom,
        total_bce / denom,
        total_dice / denom,
        mean_pos_iou,
        mean_neg_iou,
    )


def save_visuals_from_loader(
    model,
    dataloader,
    device,
    out_dir: Path,
    epoch: int,
    total_target: int,
    pos_target: int,
    neg_target: int,
):
    model.eval()
    out_epoch_dir = out_dir / f"epoch{epoch}"
    ensure_dir(out_epoch_dir)

    pos_saved = 0
    neg_saved = 0
    total_saved = 0

    from torchvision.utils import save_image

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values)
            probs = torch.sigmoid(logits).cpu()
            images = batch["pixel_values"].cpu()
            masks = batch["mask"].cpu()
            indices = batch.get("index")

            mask_area = masks.view(masks.size(0), -1).sum(dim=1)
            for i in range(images.size(0)):
                if total_saved >= total_target:
                    return

                is_positive = mask_area[i].item() > 0.5
                if is_positive and pos_saved >= pos_target:
                    continue
                if not is_positive and neg_saved >= neg_target:
                    continue

                overlay = overlay_segmentation(images[i], probs[i], masks[i])
                grid = torch.cat(
                    [
                        images[i],
                        masks[i].repeat(3, 1, 1),
                        probs[i].repeat(3, 1, 1),
                        overlay,
                    ],
                    dim=2,
                )

                if indices is not None:
                    if torch.is_tensor(indices):
                        sample_id = int(indices[i].item())
                    elif isinstance(indices, list):
                        sample_id = int(indices[i])
                    else:
                        sample_id = total_saved
                else:
                    sample_id = total_saved

                prefix = "pos" if is_positive else "neg"
                fname = out_epoch_dir / f"{prefix}_{sample_id:05d}.png"
                save_image(grid, fname)

                total_saved += 1
                if is_positive:
                    pos_saved += 1
                else:
                    neg_saved += 1


if __name__ == "__main__":
    main()
