# DeepLabv3+ Finetuning Plan for Native Planting Segmentation

This plan describes how to adapt this repository to fine-tune a pretrained DeepLabv3+ model (ResNet-101 backbone) on the NZ native planting masks.

## 1. High-Level Approach

- Use a DeepLabv3+ implementation from TorchGeo or MMSegmentation that already includes weights trained on remote-sensing datasets (e.g., LoveDA or DeepGlobe).
- Replace the classifier head with a binary segmentation head.
- Fine-tune on the existing `dataset/train` / `dataset/test` CSV pairs.
- Add negative sampling/augmentation consistent with the current pipeline.

## 2. Environment Setup

1. **Dependencies:**
   - Add `torchvision>=0.15`, `torchgeo>=0.5`, `mmcv`, `mmsegmentation` (choose one framework; this plan assumes TorchGeo for simplicity).
   - Update `sam2_requirements.txt` or create `requirements_deeplab.txt` with the new packages.

2. **Virtual Environment:**
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements_deeplab.txt`

## 3. Repository Integration

1. **Directory Structure:**
   - Create `deeplab/` containing `train.py`, `dataset.py`, `config.yaml`.

2. **Dataset Loader:**
   - Reuse `sam2_dataset.PairDataset` but add normalization (`mean/std`) and optional resizing inside the loader.
   - Convert masks to `torch.float32` of shape `[1,H,W]`.

3. **Model Definition (`deeplab/model.py`):**
   - Load TorchGeo checkpoint: `from torchgeo.models import get_model`
   - `model = get_model("deeplabv3_resnet101_loveda", num_classes=2)`
   - Replace classifier head: `model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)` and wrap with sigmoid in loss.

4. **Loss & Metrics:**
   - Use `BCEWithLogitsLoss` + `SoftDiceLoss` (implement dice in `utils/losses.py`).
   - Track IoU, Precision, Recall on foreground-only tiles.

5. **Training Script (`deeplab/train.py`):**
   - Parse CLI for `--train_csv`, `--val_csv`, `--epochs`, `--lr`, `--bs`, `--checkpoint_path`.
   - Build dataloaders with heavy augmentations (flip, rotate, color jitter, random crop).
   - Optimizer: `AdamW` with `lr=5e-5`, `weight_decay=1e-4`.
   - Scheduler: `ReduceLROnPlateau` on validation IoU or `CosineAnnealingLR`.
   - Mixed precision via `torch.cuda.amp`.
   - Save best checkpoint by validation IoU.

6. **Inference Script (`deeplab/infer.py`):**
   - Load checkpoint, run sliding-window inference on large images.
   - Apply `torch.sigmoid` -> threshold 0.5.

## 4. Training Procedure

1. Prepare data: ensure CSVs reference imagery/masks; optionally add `dataset/val` split separate from `test` for final evaluation.
2. Run `python deeplab/train.py --train_csv dataset/train/pairs.csv --val_csv dataset/test/pairs.csv --epochs 50 --bs 8 --lr 5e-5 --checkpoint_path runs/deeplab`
3. Monitor logs: report per-epoch `loss`, `dice`, `iou_fg`, `precision`, `recall`.
4. Use TensorBoard or CSV logging.

## 5. Evaluation Criteria

- Foreground IoU ≥ 0.6 on validation.
- Precision/Recall balance (avoid vegetation false positives).
- Visual inspection of `runs/deeplab/val_examples/*.png` similar to existing SAM2 format.

## 6. Deployment

- Freeze model weights, export to TorchScript or ONNX for large-scale inference.
- Build `deeplab/batch_infer.py` to process thousands of tiles with multiprocessing.
- Integrate results into GeoTIFF or shapefile outputs if needed (use rasterio).

## 7. Timeline & Tasks

1. Environment + dependency setup (0.5 day).
2. Implement dataset + dataloaders (0.5 day).
3. Implement model wrapper + training loop (1 day).
4. Run pilot fine-tune, adjust hyperparameters (1–2 days).
5. Build inference pipeline (0.5 day).
6. Documentation & cleanup (0.5 day).

