# SegFormer Finetuning Plan for NZ Native Planting Detection

We will integrate a SegFormer encoder-decoder (MIT-B4/B5 backbone) pretrained on aerial/remote-sensing data, fine-tune it on the local dataset, and produce automatic segmentation masks.

## 1. Overview

- Leverage Hugging Face `SegFormerForSemanticSegmentation` checkpoints (e.g., `pretrained/segformer-b4-loveda`).
- Convert CSV dataset into PyTorch dataset compatible with Hugging Face transformers.
- Train with gradient checkpointing & mixed precision to handle higher resolution (1024×1024) tiles.

## 2. Environment

1. Add dependency file `requirements_segformer.txt` containing:
   - `transformers>=4.35`
   - `datasets`
   - `torchvision`
   - `accelerate`
   - `albumentations`

2. Install: `pip install -r requirements_segformer.txt`.

## 3. Data Pipeline

1. Implement `segformer/dataset.py`:
   - Reuse `PairDataset` but convert to `transformers.ImageProcessor` format.
   - Apply augmentations with `albumentations` (resize/crop to 512–768, random rotations, color jitter, blur, fog).
   - Normalize via processor (`image_processor = SegformerImageProcessor.from_pretrained(...)`).
   - Output dictionary: `{ "pixel_values": tensor, "labels": mask_tensor }`.

2. Provide `segformer/data_collator.py` to pad to uniform size (multiples of 32) and convert masks to `long` with 0/1 classes.

## 4. Model Configuration

1. Load checkpoint:
   ```python
   from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
   model = SegformerForSemanticSegmentation.from_pretrained(
       "pretrained/segformer-b4-loveda",
       num_labels=2,
       ignore_mismatched_sizes=True,
   )
   model.decode_head.classifier = nn.Conv2d(model.decode_head.classifier.in_channels, 1, kernel_size=1)
   ```
2. Wrap outputs to compute `BCEWithLogitsLoss + Dice` using `model.forward` override or Lightning-style training loop.

## 5. Training Script (`segformer/train.py`)

1. Parse CLI arguments: dataset CSV paths, resolution, batch size, epochs, output dir.
2. Instantiate dataset/dataloader.
3. Use `accelerate.Accelerator` for multi-GPU or gradient accumulation. Configure mixed precision (`fp16`).
4. Optimizer: `AdamW` with `weight_decay=0.01`, `lr=3e-5`.
5. Scheduler: `get_cosine_schedule_with_warmup(optimizer, warmup_steps=0.1 * total_steps, total_steps=epochs * len(dl))`.
6. Training loop per epoch:
   - Forward pass -> compute logits.
   - Compute BCE + SoftDice.
   - Step optimizer, scheduler.
   - Log scalar metrics via `accelerator.log` or manual `print`.
7. Validation loop every epoch using evaluation dataset (no augmentations, center crop/rescale to original size).
8. Save checkpoints (`best.pt`, `last.pt`) with `accelerator.unwrap_model(model).state_dict()`.

## 6. Evaluation & Visualization

1. Create `segformer/visualize.py` to replicate SAM-style 2×2 panels: original, ground truth, logits heatmap, overlay.
2. During validation, sample `N` tiles and save to `runs/segformer-{run_id}/val_examples/epoch{E}`.
3. Metrics: foreground IoU, precision, recall, F1. Log positives vs negatives separately.

## 7. Inference Pipeline

1. Implement `segformer/predict.py` using sliding window with overlap (e.g., 256 stride) for >1024 images.
2. Use `torch.sigmoid` threshold 0.4–0.5; optionally apply morphological opening to clean noise.
3. Support batch inference using `accelerate` for GPU evaluation.

## 8. Hyperparameter Suggestions

- Batch size: 4 (1024 resolution) with grad accumulation to match effective 16.
- Epochs: 40–60 with early stopping on validation IoU.
- Augmentations: rotate 90 degrees, random scale [0.75, 1.25], random brightness/contrast, gaussian blur, random erasing.

## 9. Timeline & TODOs

1. Scaffold folder structure & install deps (0.5 day).
2. Implement dataset/collator & verify shape compatibility (0.5 day).
3. Build training loop with accelerate (1 day).
4. Run pilot training, tune LR & threshold (1–2 days).
5. Implement inference & visualization (0.5 day).
6. Document results, create README for segformer module (0.5 day).

