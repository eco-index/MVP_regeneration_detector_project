# Finetuning SAM for Semantic Segmentation

This document provides instructions for using the `finetune_sam.py` script to fine-tune a pre-trained Segment Anything Model (SAM) for a custom semantic segmentation task. The script handles dataset loading, data augmentation, training, validation, and model saving.

## Prerequisites

1.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Clone the repository and install the necessary dependencies using the `requirements.txt` file:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    pip install -r requirements.txt
    ```

2.  **Key Libraries**:
    The script relies on several key Python libraries:
    *   `torch`: For deep learning operations.
    *   `transformers`: For loading SAM models and processors from Hugging Face.
    *   `datasets`: For loading and handling the dataset in Hugging Face format.
    *   `albumentations`: For data augmentation.
    *   `scikit-learn`: For calculating evaluation metrics.
    *   `tqdm`: For progress bars during training and evaluation.

## Dataset Preparation

The `finetune_sam.py` script expects your dataset to be in the Hugging Face `datasets` format, saved to disk. This typically includes 'image' and 'mask' columns, where masks are single-channel images representing segmentation classes.

To prepare your dataset from raw images and corresponding segmentation masks, you can use the provided `prepare_segmentation_dataset.py` script.

**Example command to prepare the dataset:**
```bash
python scripts/prepare_segmentation_dataset.py \
    --image_dir path/to/your/source_images \
    --mask_dir path/to/your/source_masks \
    --output_dataset_path data/hf_segmentation_dataset \
    --image_extension .jpg \
    --mask_extension .png
```
Replace `path/to/your/source_images` and `path/to/your/source_masks` with the actual paths to your image and mask directories. Adjust extensions as needed. The output will be saved to `data/hf_segmentation_dataset`.

## Running the Finetuning Script

Once your dataset is prepared and saved in the Hugging Face format, you can run the `finetune_sam.py` script.

**Example command to run finetuning:**
```bash
python scripts/finetune_sam.py \
    --dataset_path data/hf_segmentation_dataset \
    --model_name facebook/sam-vit-base \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --image_size 256 \
    --output_dir models/sam_custom_segmentation \
    --save_best_metric iou \
    --device cuda
```
Adjust the arguments as needed for your specific setup and dataset.

## Command-Line Arguments

The `finetune_sam.py` script offers a variety of command-line arguments to configure the fine-tuning process. For a full list of arguments, their descriptions, and default values, use the help message:

```bash
python scripts/finetune_sam.py --help
```

**Key arguments include:**

*   `--dataset_path`: Path to the Hugging Face dataset directory.
*   `--model_name`: Name of the SAM model from Hugging Face Model Hub (e.g., `facebook/sam-vit-base`).
*   `--image_size`: Target size for images and masks during training (e.g., 256 for 256x256).
*   `--num_epochs`: Number of training epochs.
*   `--batch_size`: Batch size for training and validation.
*   `--learning_rate`: Initial learning rate for the optimizer.
*   `--optimizer_type`: Type of optimizer (e.g., `adamw`).
*   `--loss_function_type`: Type of loss function (e.g., `bcewithlogits`).
*   `--val_split_ratio`: Proportion of the dataset to use for validation.
*   `--eval_metrics`: Metrics to compute during evaluation (e.g., `loss`, `accuracy`, `iou`, `dice`).
*   `--output_dir`: Directory to save model checkpoints and the final model.
*   `--save_best_metric`: Validation metric to monitor for saving the best model (e.g., `iou`).
*   `--device`: Device to use for training (`cuda` or `cpu`).

## Output

The script will produce the following outputs:

1.  **Console Logs**:
    *   Progress of dataset loading, model loading, training epochs, and validation.
    *   Average training loss per epoch.
    *   Validation metrics (loss, accuracy, IoU/Jaccard, Dice/F1) per epoch.
    *   Confirmation messages for saved models.

2.  **Trained Models** (saved in the directory specified by `--output_dir`):
    *   `best_model.pth`: The model checkpoint that achieved the best score on the validation set for the metric specified by `--save_best_metric`.
    *   `final_model.pth`: The model checkpoint from the final training epoch.

## GPU Usage

Training deep learning models is computationally intensive and significantly faster on a GPU.
*   The script will attempt to use a CUDA-enabled GPU by default if `torch.cuda.is_available()` is true and `--device cuda` (default) is specified.
*   If you do not have a CUDA-enabled GPU or wish to run on the CPU, you can specify `--device cpu`. However, training on a CPU will be very slow.

Make sure your PyTorch installation is compatible with your CUDA drivers if you intend to use a GPU.
