# Plan: Fine-Tuning a Remote-Sensing Foundation Model (Prithvi/ViT) for Native Planting Segmentation

This plan outlines how to leverage a foundation model pretrained on satellite data (e.g., NASA’s Prithvi or Microsoft’s ViT-based geospatial encoders) and adapt it for binary segmentation of NZ native plantings.

## 1. Motivation

Foundation models supply a strong encoder that already understands vegetation textures, seasonal variation, and sensor noise. By attaching a lightweight decoder, we can fine-tune with limited labeled data while retaining automatic inference.

## 2. Model Choice

- **Encoder:** `prithvi_vit_base_patch16` (available via Hugging Face `nasa/prithvi-base-100m`).
- **Decoder:** Simple Feature Pyramid Network (FPN) or UNet-like decoder implemented locally.
- **Input:** RGB imagery at 512–1024 resolution (extend to multispectral bands if available).

## 3. Environment & Dependencies

1. Create `requirements_prithvi.txt` with:
   - `torchvision`
   - `transformers>=4.35`
   - `timm`
   - `albumentations`
   - `rasterio` (optional for GeoTIFF output)

2. Install dependencies in the virtualenv.

## 4. Repository Changes

1. Add `prithvi/` module with:
   - `model.py`: encoder wrapper + decoder head.
   - `dataset.py`: data loader supporting optional multispectral bands.
   - `train.py`, `infer.py`.

2. **Dataset adjustments:**
   - Extend `PairDataset` to optionally load extra bands (if available) and resample/rescale to 512 or 768.
   - Apply heavy augmentations via `albumentations` (flip, rotate, random crop, hue/saturation, CLAHE, blur).

## 5. Model Implementation Details

1. **Encoder:**
   ```python
   from transformers import AutoModel
   encoder = AutoModel.from_pretrained(
       "nasa/prithvi-base-100m",
       add_pooling_layer=False,
       output_hidden_states=True,
   )
   ```
   Extract hidden states at multiple layers (e.g., 4, 8, 12) for skip connections.

2. **Decoder:**
   - Build an FPN with lateral 1×1 convs; upsample to original spatial resolution using bilinear interpolation.
   - Final classifier `Conv2d` -> 1 channel.

3. **Loss:** BCEWithLogits + SoftDice + optional focal term for class imbalance.

4. **Optimization:**
   - Freeze lower encoder layers for first 5 epochs (`encoder.embeddings`, first 4 blocks).
   - Use discriminative learning rates: `1e-5` for encoder, `5e-4` for decoder.
   - Optimizer: `AdamW`, `weight_decay=0.01`.
   - Scheduler: cosine with warmup (5% steps).

5. **Mixed Precision:** Use `torch.cuda.amp`.

## 6. Training Pipeline (`prithvi/train.py`)

1. Parse ARGS: `train_csv`, `val_csv`, `epochs`, `lr_head`, `lr_encoder`, `bs`, `freeze_epochs`, `out_dir`.
2. Build dataloaders with `PairDataset` (augment=True for train, False for val).
3. Forward pass: run encoder, collect hidden states, decode to mask.
4. Compute loss/metrics, log to console + TensorBoard.
5. Unfreeze encoder layers after `freeze_epochs`.
6. Save best checkpoint by positive IoU.

## 7. Evaluation & Visualization

1. Save sample outputs every epoch using the 2×2 panel format.
2. Compute metrics separately for positive and negative tiles.
3. Optionally add calibration metrics (AUROC) for threshold tuning.

## 8. Inference for Large Areas

1. Implement `prithvi/infer.py` with sliding-window inference and GPU batching.
2. Optionally fuse overlapping windows with averaging to reduce seams.
3. Export probability map as GeoTIFF; run connected-component/largest-component filtering to remove noise.

## 9. Resource Estimates

- Batch size 2–4 (1024×1024) with grad accumulation to achieve effective batch 8.
- 24–48 GB GPU recommended; otherwise crop to 768 and use accumulation.
- Training time: ~6–8 hours for 50 epochs on a single A100; longer on smaller GPUs.

## 10. Timeline & Tasks

1. Install dependencies & download checkpoint (0.5 day).
2. Implement dataset augmentations & loader (0.5 day).
3. Build encoder–decoder wrapper, verify forward shapes (1 day).
4. Implement training loop with freeze/unfreeze schedule (1 day).
5. Run pilot fine-tune, adjust thresholds/augmentations (1–2 days).
6. Deliver inference scripts + docs (0.5 day).

