# Prithvi v2: Data Prep, Training, and Inference

This doc explains the repo workflow for preparing data, training the Prithvi v2 segmentation model, and running tiled inference. It focuses on the `prithvi2` pipeline (not SAM2 or Prithvi v1).

## Relevant scripts

- `process_plantings_validated_data.py`: renames `*_raw.png`/`*_mask.png` pairs into `polygon_{i}_raw.png` + `polygon_{i}_mask.png`.
- `flatten_data.py`: flattens nested `polygon_*` pairs and converts magenta masks (255, 0, 255) to binary.
- `create_dataset.py`: crops large polygon images into train/test splits and writes `pairs.csv`.
- `prithvi2/train.py`: PyTorch Lightning training entrypoint.
- `prithvi2_tile_inference.py`: tiled inference over large images.

All other scripts and code are legacy artifacts from earlier stages of development e.g. ChatGPT based classifier experiments, Sam2 training, inference and debugging, Prithvi (v1) training and inference, etc.

## Prerequisites

This codebase is designed to run on an Ubuntu machine with an NVIDIA GPU and CUDA installed. Note that their may be other undocumented requirements and setup steps depending on the exact machine used. All development was done on an NVIDIA A40 GPU (48GB VRAM) enabled [Runpod](https://runpod.io) Virtual Machine.

## Setup and requirements

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

### Optional: Start from plantings-validated data (use this to standardize file names)

`process_plantings_validated_data.py` reads `*_raw.png` + `*_mask.png` files and writes polygon-style names:

```
python process_plantings_validated_data.py
```

Edit `INPUT_DATA_DIR` and `PROCESSED_DATA_DIR` inside the script if needed.

### Starting from a (optionally nested) folder of polygon\_\* pairs

If your raw/mask pairs are already named `polygon_{n}_raw.png`/`polygon_{n}_mask.png` but live in nested directories, run:

```
python flatten_data.py --input_dir data/output --output_dir satellite_images
```

This also converts magenta masks (255, 0, 255) to binary.

### Create train/test crops + CSVs

`create_dataset.py` expects a directory containing paired PNGs:

```
polygon_0_raw.png
polygon_0_mask.png
polygon_1_raw.png
polygon_1_mask.png
...
```

- Raw images are RGB.
- Masks should be single-channel or RGB; anything >127 after conversion is treated as foreground.
- The script uses the mask fraction to label each crop as positive/negative.

Once you have a flat directory of `polygon_*` pairs, generate the training set:

```
python create_dataset.py \
  --input_dir satellite_images \
  --output_dir dataset \
  --size 512 \
  --num_pairs 10000 \
  --threshold 0.1 \
  --pos_neg_ratio 1 \
  --train_split 0.8
```

Output structure:

```
dataset/
  train/
    pairs.csv
    raw_images/
    mask_images/
  test/
    pairs.csv
    raw_images/
    mask_images/
```

`pairs.csv` columns:

- `raw_path`, `mask_path` (paths are relative to the CSV location)
- `masked_fraction` (for reference; not used by training)

If you already have a `pairs.csv` in the expected format, you can skip `create_dataset.py`.

## Training Prithvi v2

Basic command:

```
python -m prithvi2.train \
  --train_csv dataset/train/pairs.csv \
  --val_csv dataset/test/pairs.csv
```

Optional test pass (uses best checkpoint):

```
python -m prithvi2.train \
  --train_csv dataset/train/pairs.csv \
  --val_csv dataset/test/pairs.csv \
  --test_csv dataset/test/pairs.csv
```

Common flags to know:

- `--image_size` (default 224): crops are resized to this size.
- `--batch_size`, `--epochs`, `--lr`, `--weight_decay`
- `--backbone` (default `prithvi_eo_v2_300`)
- `--bands` and `--band_mapping`: defaults expand RGB to 6 bands (`2,1,0,0,1,2`).
- `--freeze_backbone` or `--freeze_epochs` for staged finetuning.
- `--device` (`cpu`, `cuda`, or `auto`) and `--precision` (default `bf16-mixed`)
- `--checkpoint_dir` (default `runs`), `--checkpoint_monitor` (default `val/mIoU`)
- `--save_visuals` / `--visuals_*` to dump validation overlays.

Output layout (example):

```
runs/prithvi2_YYYYMMDD-HHMMSS/
  checkpoints/
    best-checkpoint-epoch=03-val_loss=0.12.ckpt
  logs/
  val_examples/   # only if save_visuals > 0
```

## Inference (tiled)

Use `prithvi2_tile_inference.py` to run a trained checkpoint over a directory of RGB tiles.

Example:

```
python prithvi2_tile_inference.py \
  --image-dir data/Waiwhakaiho_grid \
  --weights runs/prithvi2_YYYYMMDD-HHMMSS/checkpoints/best-checkpoint-epoch=03-val_loss=0.12.ckpt \
  --output-dir waiwhakaiho_predictions_prithvi2 \
  --window-size 512 \
  --model-size 224 \
  --stride 64 \
  --threshold 0.5
```

Important inference flags:

- `--window-size`: crop size on the input tiles.
- `--model-size`: resize crops to match training `--image_size`.
- `--stride`: overlap between crops (must be <= window size).
- `--batch-size`, `--device`
- `--bands`, `--band-mapping`: override checkpoint defaults if needed.
- `--clip N`: process only N random tiles.
- `--compile` / `--compile-cache`: use `torch.compile` for speed.

Outputs:

```
<output-dir>/
  probabilities/
    <tile>_prob.png
  masks/
    <tile>_mask.png
```
