import argparse
import time
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from prithvi2.dataset import Prithvi2DataModule
from prithvi2.model import build_prithvi2_task
from prithvi2.utils import ensure_dir, parse_int_list, parse_str_list, set_seed
from prithvi2.visuals import infer_rgb_indices, save_visuals_from_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Prithvi v2 for native planting segmentation")
    parser.add_argument("--train_csv", default="dataset/train/pairs.csv")
    parser.add_argument("--val_csv", default="dataset/test/pairs.csv")
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--checkpoint_dir", default="runs")
    parser.add_argument("--checkpoint_monitor", default="val/mIoU")
    parser.add_argument("--checkpoint_mode", default="max")
    parser.add_argument("--save_visuals", type=int, default=40)
    parser.add_argument("--visuals_pos", type=int, default=20)
    parser.add_argument("--visuals_neg", type=int, default=20)
    parser.add_argument("--visuals_every", type=int, default=1)
    parser.add_argument("--visuals_threshold", type=float, default=0.5)
    parser.add_argument("--visuals_rgb_indices", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bands", default="BLUE,GREEN,RED,NIR_BROAD,SWIR_1,SWIR_2")
    parser.add_argument("--band_mapping", default="2,1,0,0,1,2")
    parser.add_argument("--backbone", default="prithvi_eo_v2_300")
    # parser.add_argument("--backbone", default="prithvi_eo_v2_600")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_channels", default="128,64")
    parser.add_argument("--neck_indices", default="5,11,17,23")
    # parser.add_argument("--neck_indices", default="7,15,23,31")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_epochs", type=int, default=0)
    parser.add_argument("--loss", default="focal")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--scheduler", default="StepLR")
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--no_augment", action="store_true")
    return parser.parse_args()


def _resolve_device(device_arg: str):
    if device_arg == "cpu":
        return "cpu", 1
    if device_arg in ("cuda", "gpu"):
        return "gpu", 1
    return "auto", "auto"


class BackboneUnfreezeCallback(pl.Callback):
    def __init__(self, unfreeze_epoch: int) -> None:
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self._done = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._done or self.unfreeze_epoch <= 0:
            return
        if trainer.current_epoch < self.unfreeze_epoch:
            return

        model = getattr(pl_module, "model", None)
        encoder = getattr(model, "encoder", None) if model is not None else None
        if encoder is None:
            for param in pl_module.parameters():
                param.requires_grad_(True)
        else:
            for param in encoder.parameters():
                param.requires_grad_(True)

        self._done = True


class ValidationVisualsCallback(pl.Callback):
    def __init__(
        self,
        out_dir: Path,
        total_target: int,
        pos_target: int,
        neg_target: int,
        every_n_epochs: int,
        threshold: float,
        rgb_indices: list[int] | None,
    ) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.total_target = total_target
        self.pos_target = pos_target
        self.neg_target = neg_target
        self.every_n_epochs = every_n_epochs
        self.threshold = threshold
        self.rgb_indices = rgb_indices

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or self.total_target <= 0:
            return
        if self.every_n_epochs <= 0:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            return

        val_loader = datamodule.val_dataloader()
        save_visuals_from_loader(
            pl_module.model,
            val_loader,
            self.out_dir,
            trainer.current_epoch,
            total_target=self.total_target,
            pos_target=self.pos_target,
            neg_target=self.neg_target,
            threshold=self.threshold,
            rgb_indices=self.rgb_indices,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.encoder_lr is not None:
        args.lr = args.encoder_lr

    bands = parse_str_list(args.bands)
    band_mapping = parse_int_list(args.band_mapping)
    head_channels = parse_int_list(args.head_channels)
    neck_indices = parse_int_list(args.neck_indices)
    rgb_indices = (
        parse_int_list(args.visuals_rgb_indices)
        if args.visuals_rgb_indices is not None
        else infer_rgb_indices(bands)
    )

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.checkpoint_dir) / f"prithvi2_{run_id}"
    checkpoint_dir = run_root / "checkpoints"
    ensure_dir(checkpoint_dir)

    logger = TensorBoardLogger(save_dir=str(run_root), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor=args.checkpoint_monitor,
        mode=args.checkpoint_mode,
        dirpath=str(checkpoint_dir),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
    )

    callbacks = [checkpoint_callback]
    if args.freeze_epochs > 0:
        callbacks.append(BackboneUnfreezeCallback(args.freeze_epochs))
    if args.save_visuals > 0:
        visuals_dir = run_root / "val_examples"
        callbacks.append(
            ValidationVisualsCallback(
                out_dir=visuals_dir,
                total_target=args.save_visuals,
                pos_target=args.visuals_pos,
                neg_target=args.visuals_neg,
                every_n_epochs=args.visuals_every,
                threshold=args.visuals_threshold,
                rgb_indices=rgb_indices,
            )
        )

    accelerator, devices = _resolve_device(args.device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy="auto",
        devices=devices,
        precision=args.precision,
        num_nodes=1,
        logger=logger,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        callbacks=callbacks,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else 0.0,
    )

    datamodule = Prithvi2DataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bands=bands,
        band_mapping=band_mapping,
        augment=not args.no_augment,
    )

    model = build_prithvi2_task(
        backbone=args.backbone,
        bands=bands,
        neck_indices=neck_indices,
        num_classes=args.num_classes,
        head_dropout=args.head_dropout,
        head_channels=head_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        freeze_backbone=args.freeze_backbone or args.freeze_epochs > 0,
    )

    print(f"Run directory: {run_root}")
    trainer.fit(model, datamodule=datamodule)

    if args.test_csv:
        ckpt_path = checkpoint_callback.best_model_path or None
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
