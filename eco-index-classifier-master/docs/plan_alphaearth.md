# Plan: Leveraging Google AlphaEarth Embeddings for Native Planting Detection

This document evaluates how to build a segmentation system on top of Google’s AlphaEarth embedding datasets and outlines an implementation strategy within this repository.

## 1. Background & Suitability

- **AlphaEarth** provides global satellite image embeddings (patch descriptors) trained via large-scale self-supervision across multiple sensors and seasons.
- Embeddings are typically provided as dense grids per image tile (e.g., 224×224 patch -> 128/256D vector) that capture vegetation, built-up area, and phenology information.
- Using these embeddings as a fixed feature extractor drastically reduces the amount of labeled data required for downstream tasks, making them ideal for the small NZ dataset.

**Pros**
- Encapsulates rich spectral + temporal context learned from petabytes of imagery.
- Lower training cost: we only train a lightweight decoder on top of frozen embeddings.
- Flexible: embeddings can be ingested by any model (CNN/Transformer/MLP).

**Cons**
- Requires preprocessing pipeline to fetch/store embeddings per tile (Google Cloud storage, TFRecord format).
- Resolution is typically coarser than raw imagery (e.g., 10–20 m). If our masks are 1024×1024 high-res RGB, we must align them to the embedding grid.
- Embedding coverage may miss latest acquisitions; need to verify temporal coverage for NZ.

## 2. Data Acquisition & Preprocessing

1. **Access**
   - Request access to AlphaEarth embeddings (AlphaEarth+ release) via Google Research or Google Cloud (dataset often provided via BigQuery/Cloud Storage).
   - Download tile indices corresponding to dataset imagery (Lat/Lon bounding boxes or Sentinel-2 tile IDs).

2. **Alignment Workflow**
   - For each training tile:
     1. Use georeference to query AlphaEarth embedding tiles (likely 224×224 patches covering 2.5 km).
     2. Resample embeddings to match the imagery grid. Options:
        - Nearest-neighbor to 1024×1024 using `rasterio`/`GDAL`.
        - Or keep embeddings at native resolution and train decoder at lower res, upsample predictions.
     3. Cache embeddings as `.npy` or `.pt` files in `embeddings/` to avoid repeated downloads.

3. **Data Storage**
   - Extend CSVs with an `embedding_path` column linking each image to its precomputed embedding tensor.

## 3. Repository Integration

1. Create `alphaearth/` module with:
   - `dataset.py`: loads RGB image (optional), mask, and embedding tensor.
   - Config flag `--use_rgb` to optionally fuse raw imagery with embeddings (concat channels or dual-branch network).

2. Add `requirements_alphaearth.txt` (if not already installed):
   - `rasterio`, `gcsfs`, `numpy`, `torch`, `einops`.

## 4. Model Architectures

### Option A: Frozen Embeddings + Lightweight Decoder

1. Treat embeddings as input channels (e.g., `[C=256,H=H_e,W=W_e]`).
2. Apply a shallow UNet/FPN head:
   - Downsample/upsample path with 3–4 levels, 3×3 convs, skip connections.
   - Output single-channel logits.
3. Loss: BCEWithLogits + Dice.

### Option B: Fusion of RGB + Embedding Streams

1. Dual-branch network: one branch for 3-channel RGB (e.g., ResNet34 encoder), another for embeddings (1×1 conv to reduce dimension, followed by ResNet blocks).
2. Fuse at multiple scales (concatenate features + conv).
3. Decoder (FPN) outputs segmentation mask.

### Option C: Transformer Head

1. Flatten embeddings -> tokens (`H_e*W_e × D`).
2. Add learnable decoder tokens (Segmenter-style) + cross-attention to produce mask tokens.
3. Reshape to map -> binary mask.

**Recommendation:** Start with Option A for simplicity; iterate to fusion once alignment is stable.

## 5. Training Pipeline (`alphaearth/train.py`)

1. CLI arguments: `--train_csv`, `--val_csv`, `--epochs`, `--lr`, `--bs`, `--use_rgb`, `--embedding_dim`, `--embedding_scale`.
2. Dataloader returns dictionary `{ "embedding": tensor, "image": tensor?, "mask": tensor }`.
3. Use `torch.cuda.amp` mixed precision.
4. Optimizer: `AdamW (lr=1e-3)` for decoder-only training; reduce to `1e-4` when fusing RGB branch.
5. Scheduler: cosine annealing with warmup (or ReduceLROnPlateau on val IoU).
6. Metrics: Foreground IoU, precision, recall, AUROC (for threshold tuning).
7. Save best checkpoint.

## 6. Validation & Visualization

- Save 2×2 panels as with other experiments (embedding-only overlays may require upsampling to original resolution).
- Plot probability maps over RGB imagery.
- Evaluate IoU separately for positive/negative tiles.

## 7. Inference

1. Precompute embeddings for inference tiles (batch from GCS -> local).
2. Run decoder to obtain probability map, threshold (0.5) to produce mask.
3. Optionally refine via morphological operations or CRF.

## 8. Practical Considerations

- **Spatial Resolution:** Embeddings often correspond to Sentinel-2 (10 m). Native planting polygons must be large enough to appear distinct at that scale; otherwise incorporate high-res RGB branch.
- **Temporal Matching:** Ensure embedding timestamp aligns with imagery; use nearest-date embeddings or average across seasons to highlight permanent plantings.
- **Storage Costs:** Embedding grids can be large (~100 MB per tile). Consider compressing (float16) or streaming from cloud during training.
- **Licensing:** Verify dataset terms allow commercial use/redistribution; incorporate attribution.

## 9. Timeline

1. Secure dataset access & download sample embeddings (0.5–1 day).
2. Implement preprocessing/alignment pipeline (1–2 days).
3. Build dataset loader + baseline decoder (1 day).
4. Run pilot training; tune LR, decoder depth (1–2 days).
5. Explore RGB fusion or transformer head if needed (1–2 days).

