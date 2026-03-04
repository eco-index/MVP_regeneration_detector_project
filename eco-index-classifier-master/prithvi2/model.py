from __future__ import annotations

from typing import List


def build_prithvi2_task(
    *,
    backbone: str,
    bands: List[str],
    neck_indices: List[int],
    num_classes: int,
    head_dropout: float,
    head_channels: List[int],
    lr: float,
    weight_decay: float,
    loss: str,
    optimizer: str,
    scheduler: str,
    scheduler_step: int,
    scheduler_gamma: float,
    freeze_backbone: bool,
):
    try:
        from terratorch.tasks import SemanticSegmentationTask
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "terratorch is required for Prithvi v2 training. "
            "Install it with `pip install terratorch==0.99.8`."
        ) from exc

    model_args = dict(
        backbone_pretrained=True,
        backbone=backbone,
        backbone_bands=bands,
        backbone_num_frames=1,
        decoder="UperNetDecoder",
        decoder_channels=256,
        decoder_scale_modules=True,
        num_classes=num_classes,
        head_dropout=head_dropout,
        head_channel_list=head_channels,
        necks=[
            dict(name="SelectIndices", indices=neck_indices),
            dict(name="ReshapeTokensToImage"),
        ],
        rescale=True,
    )

    task = SemanticSegmentationTask(
        model_args=model_args,
        plot_on_val=False,
        loss=loss,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_hparams={"step_size": scheduler_step, "gamma": scheduler_gamma},
        optimizer_hparams=dict(weight_decay=weight_decay),
        ignore_index=-1,
        freeze_backbone=freeze_backbone,
        freeze_decoder=False,
        model_factory="EncoderDecoderFactory",
    )
    return task
