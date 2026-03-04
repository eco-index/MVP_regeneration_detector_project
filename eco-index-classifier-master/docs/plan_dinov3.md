# Plan: Using DINOv3 Features for Native Planting Segmentation

This document outlines how to repurpose a DINOv3 self-supervised ViT encoder as the backbone for a semantic-segmentation model that detects new NZ native plantings.

## 1. Background

DINOv3 (Caron et al., 2023) provides self-supervised Vision Transformer models trained on large-scale image data. While not remote-sensing specific, their representations transfer well with limited labels. The strategy is to:

1. Load a pretrained DINOv3 ViT (e.g., `dinov3_vitb14_reg4`).
2. Attach a decoder (Mask2Former-style FPN or simple UNet head).
3. Fine-tune on the native planting dataset.

Because DINOv3 models typically accept ImageNet-normalized RGB inputs, we must ensure imagery is normalized accordingly.

## 2. Dependencies & Environment

- Add `requirements_dinov3.txt`:
  ```
  torchvision
  timm>=0.9
  albumentations
  einops
  ```
- Install into the existing `.venv`.

## 3. Model Architecture

1. **Encoder:**
   ```python
   import timm
   encoder = timm.create_model(
       'dinov3_vitb14_reg4',
       pretrained=True,
       img_size=IMAGE_SIZE,
       num_classes=0,
       global_pool=''
   )
   ```
   - Freeze patch embedding & first N transformer blocks (optional) for stability.
   - Extract intermediate token embeddings (`forward_features` returns final tokens; use hooks to capture early layers).

2. **Decoder Options:**
   - **FPN decoder:** map token grid back to feature map using `encoder.patch_embed.grid_size`, upsample using PixelShuffle, merge multi-scale features via lateral convs.
   - **Mask Decoder:** use linear projection of class token & pixel tokens similar to Segmenter.

3. **Classifier:** final 1×1 conv -> single-channel mask.

## 4. Data Pipeline

1. Reuse `PairDataset` with augmentations (flip, rotate, color jitter, blur).
2. Normalize using ImageNet stats `(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
3. Optional: random resize/crop between 512–1024 to reduce GPU load.

## 5. Training Loop (`dinov3/train.py`)

1. CLI arguments similar to other plans (train/val CSV, epochs, LR, max size, freeze blocks).
2. Build dataloaders with `bs=4` (~1024 resolution) and `pin_memory=True`.
3. Loss function: BCEWithLogits + Dice; optionally add boundary loss.
4. Optimizer: `AdamW`, `lr_head=5e-4`, `lr_backbone=5e-5`, `weight_decay=0.05`.
5. Scheduler: cosine with warmup (5% steps).
6. Mixed precision with `torch.cuda.amp.autocast`.
7. Gradient clipping `clip_grad_norm_(params, 1.0)`.
8. Logging: per-epoch loss components, foreground IoU, precision/recall.

## 6. Validation

1. Run evaluation each epoch; compute metrics separately for positive vs negative tiles.
2. Save overlays to `runs/dinov3_<run_id>/val_examples/epoch{E}`.
3. Monitor `val_iou_fg` to prevent the dominance of empty tiles.

## 7. Inference

1. Implement `dinov3/predict.py` with sliding window + overlap.
2. Use `torch.sigmoid(logits)` to ensure probabilities; threshold at 0.4–0.5.
3. Optional morphological cleanup (opening, small-component removal).

## 8. Notes on Feasibility

- DINOv3 was trained on natural imagery; adaptation to satellite data may require more augmentation (color jitter, blur, histogram equalization) to bridge the domain gap.
- Consider initializing from `dinov2` or `DINO-ResNet50` if GPU memory is limited.
- If results lag behind remote-sensing-pretrained models, combine DINOv3 encoder with pseudo-labels or supplement data via self-training.

## 9. Timeline

1. Implement loader + normalization (0.5 day).
2. Build decoder & training script (1 day).
3. Pilot fine-tune & hyperparameter tuning (1–2 days).
4. Inference tooling & documentation (0.5 day).

