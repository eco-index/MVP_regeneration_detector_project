# Eco-Index Satellite Dataset

This repository aims to build a high-quality satellite image dataset of newly planted native New Zealand trees and bush. The dataset will be used to train an image segmentation model that automatically identifies recent native plantings in aerial and satellite imagery.

## Project Goals
- Compile reliable ground-truth data describing where native vegetation has recently been planted in New Zealand.
- Pair this vector data with high-resolution satellite imagery to create a training dataset.
- Train and evaluate a segmentation model capable of mapping new native plantings across the country.

## Data Sources
The repository currently contains KML files under `data/kml/` that describe known planting polygons:

- **core** – Shapefiles derived from the Trees That Count database (2019–2022).
- **extra** – Additional KML files from individual contributors.
- **generated** – Grid-based KML files created from larger region polygons, used for systematic screenshotting.

These files provide the labels for model training and evaluation.

## Workflow Overview

The core workflow for creating the dataset and training a model is as follows:

1.  **Define a Region**: Start with a KML file containing a large polygon defining an area of interest (e.g., `data/Waiwhakaiho.kml`).
2.  **Generate a Grid**: Use `scripts/generate_grid_kml.py` to create a new KML file containing a grid of `<Point>` placemarks that fall within the original region. This is essential for the modern screenshotting script.
3.  **Clean KML Files**: Run `scripts/clean_all_kml.py` to normalize styles and ensure visibility for all raw and generated KML files before processing.
4.  **Capture Screenshots**: Execute `scripts/screenshot_all_kml.py` to automatically capture satellite imagery for every point in the cleaned KML grid files.
5.  **Prepare Dataset**: Once screenshots are captured and masks are manually created (e.g., `*_raw.png` and `*_mask.png`), run `scripts/prepare_segmentation_dataset.py` to convert them into a structured Hugging Face dataset.
6.  **Train the Model**: Use the `make train` command or run `scripts/finetune_sam.py` directly to train the segmentation model on the prepared dataset.

---

## 1. Cleaning KML Files

Raw KML exports can have inconsistent styling, hidden folders, or placemarks. The `scripts/clean_kml.py` script normalizes these files by removing visibility tags, applying a unified style, and ensuring all folders are set to be open.

To clean a single file:
```bash
python scripts/clean_kml.py \
    --input data/kml/raw/core/TTC_Verified_part2.kml \
    --output data/kml/cleaned/core/TTC_Verified_part2.kml
```

To clean all files in `data/kml/raw/` at once, use the Makefile command:
```bash
make clean-all-kml
```

## 2. Generating a Grid of Points

To systematically capture imagery over a large area, first generate a grid of points within a polygon defined in a KML file. The `scripts/generate_grid_kml.py` script calculates the bounding box of the polygon(s) in a KML, creates a grid of coordinates, and filters out points that fall outside the actual geometry.

Example usage:
```bash
python scripts/generate_grid_kml.py \
    --region_kml data/Waiwhakaiho.kml \
    --output_kml data/kml/generated/Waiwhakaiho_grid.kml \
    --step_size_deg 0.003
```
This will produce a KML file populated with `<Point>` placemarks, ready for the screenshotting step.

## 3. Capturing Google Earth Screenshots

The `scripts/screenshot_google_earth.py` script automates capturing satellite imagery from Google Earth. It is optimized for KML files containing `<Point>` placemarks, like those created by the grid generation script.

### How it Works
Instead of automating UI clicks in a search box, the script now navigates directly to each point's coordinates by constructing a specific Google Earth URL. This method is significantly faster and more reliable.

### Installation
Install the required Python packages and the Playwright browsers:
```bash
pip install playwright
playwright install
```
*Note: This script no longer depends on `geopandas`.*

### Usage
Run the script with a KML file and an output directory:
```bash
python scripts/screenshot_google_earth.py \
    --kml data/kml/generated/Waiwhakaiho_grid.kml \
    --output imagery/ \
    --wait 20
```

### Resuming Interrupted Jobs
The script is designed to handle large jobs and interruptions:
- **Automatic Skipping**: It automatically checks if a screenshot for a placemark already exists in the output directory and skips it.
- **Manual Resuming**: You can use the `--start_at` flag with a placemark name to resume the process from a specific point.
  ```bash
  python scripts/screenshot_google_earth.py \
      --kml data/kml/generated/Waiwhakaiho_grid.kml \
      --output imagery/ \
      --start_at "tile_row_35_col_9"
  ```

### Batch Processing
To capture imagery for every cleaned KML file in one pass, run:
```bash
python scripts/screenshot_all_kml.py
```
This will process all files in `data/kml/cleaned` and save the screenshots to corresponding folders inside `data/output/`.

## 4. Dataset Preparation and Loading

This section describes scripts used to prepare the captured imagery into a structured dataset for model training and how to load this dataset in PyTorch.

### `scripts/prepare_segmentation_dataset.py`

This script processes pairs of raw images (`*_raw.png`) and their corresponding manually created mask images (`*_mask.png`). It generates binary masks (where pink regions become white) and packages these (image, binary_mask) pairs into a Hugging Face `datasets` object, saving it to disk.

**Example Usage:**
```bash
python scripts/prepare_segmentation_dataset.py \
    --input_dir data/output/ \
    --output_dataset_path data/hf_segmentation_dataset \
    --patch_size 256 \
    --min_mask_fraction 0.01
```
This command scans `data/output/`, generates binary masks, creates patches, and saves the resulting dataset to `data/hf_segmentation_dataset`, ensuring at least 1% of all pixels in the dataset belong to a mask.

### `scripts/segmentation_dataloader.py`

This script provides a custom PyTorch `Dataset` class (`SegmentationDataset`) designed to work with the Hugging Face dataset created by `prepare_segmentation_dataset.py`. It is intended to be imported and used within a model training script.

### `scripts/in_memory_crop_dataset.py`

This module provides `BalancedCropDataset`, a PyTorch dataset that loads all image/mask pairs into memory and returns balanced random square crops. It ensures a specified fraction of sampled crops contain positive examples, which is highly effective for training on sparse data.

**Dataset Browser (`app.py`)**
The interactive dataset browser uses `BalancedCropDataset` to let you visually inspect augmented training samples. Run `app.py` to launch a web viewer where you can adjust augmentation parameters in real-time and see their effect on the generated crops.
```bash
python app.py
```

## 5. Viewing and Publishing the Dataset

### Viewing Dataset Examples

`scripts/view_dataset_examples.py` is a utility to visually inspect the image-mask pairs stored in a Hugging Face dataset. It loads the dataset and displays a specified number of random examples.

**Example Usage:**
```bash
python scripts/view_dataset_examples.py \
    --dataset_path data/hf_segmentation_dataset \
    --num_examples 3
```

### Publishing to Hugging Face Hub

`scripts/publish_dataset.py` allows you to upload your locally created Hugging Face dataset to the Hub.

**Example Usage:**
```bash
# Ensure you are logged in: huggingface-cli login
python scripts/publish_dataset.py \
    --local_dataset_path data/hf_segmentation_dataset \
    --repo_id "your-username/your-dataset-name"
```

## 6. Training and Inference

### Training the Model

The primary training script is `scripts/finetune_sam.py`. A convenient way to start training is by using the `Makefile`:
```bash
make train
```
This command uses a pre-defined set of hyperparameters. You can customize the training run by editing the `Makefile` or by running the script directly with your desired arguments. For more details, see `scripts/README_finetuning.md`.

### Running Inference

Once a model is trained, you can run inference on a single image using `scripts/inference.py`. The script automatically handles large images by breaking them into smaller, overlapping patches, running inference on each, and stitching the results back together.

**Example Usage:**
```bash
python scripts/inference.py \
    path/to/your/image.png \
    path/to/your/model.pth \
    --output_path path/to/output_overlay.png \
    --mask_output_path path/to/output_mask.png
```
