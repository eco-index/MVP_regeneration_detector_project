#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Dinov2Model, Dinov2PreTrainedModel, AutoImageProcessor
from transformers.modeling_outputs import SemanticSegmenterOutput
import evaluate # HuggingFace Evaluate

# Ensure the BalancedCropDataset can be imported
sys.path.append(str(Path(__file__).parent.parent))
from scripts.in_memory_crop_dataset import BalancedCropDataset

# DINOv2 specific constants (default for ViT-B/14)
DINOV2_PATCH_RESOLUTION = 14
# ImageNet mean/std are commonly used for DINOv2 fine-tuning
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

# --- Model Definition (adapted from the tutorial) ---
class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW, tokenH, num_labels):
        super(LinearClassifier, self).__init__()
        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Conv2d(in_channels, num_labels, kernel_size=1, stride=1)

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return self.classifier(embeddings)

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config, num_labels, image_size, patch_resolution):
        super().__init__(config)
        self.num_labels = num_labels
        self.image_size_h = image_size # Store for upsampling reference
        self.image_size_w = image_size
        self.dinov2 = Dinov2Model(config)

        tokenW = image_size // patch_resolution
        tokenH = image_size // patch_resolution
        
        self.classifier = LinearClassifier(
            in_channels=config.hidden_size,
            tokenW=tokenW,
            tokenH=tokenH,
            num_labels=self.num_labels
        )
        # Using ignore_index=0 means class 0 (background) is ignored in loss.
        # Ensure labels are 0 for background, 1 for foreground.
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 

    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.dinov2(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        logits = self.classifier(patch_embeddings)

        # Upsample logits to match target label size (e.g., self.image_size_h, self.image_size_w)
        target_size = (self.image_size_h, self.image_size_w)
        upsampled_logits = nn.functional.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        
        loss = None
        if labels is not None:
            # Ensure labels are of shape (B, H, W) and type long
            if labels.ndim == 4 and labels.shape[1] == 1: # If (B, 1, H, W)
                labels = labels.squeeze(1)
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = self.criterion(upsampled_logits, labels)

        if not return_dict:
            output = (upsampled_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=upsampled_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def evaluate_model(model, dataloader, device, metric_calculator, num_labels_for_metric, ignore_index_for_metric):
    model.eval()
    total_loss = 0
    
    # metric_calculator should be reset if it accumulates state, 
    # but evaluate.load() creates a fresh instance.
    # For safety, one could re-initialize here: metric_calculator = evaluate.load("mean_iou")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images, masks_float = batch # masks_float is (B, 1, H, W), float [0,1]
            images = images.to(device)
            # Convert float masks [0,1] to long integer masks {0,1} for model's criterion and metric
            masks_long_for_loss_and_metric = (masks_float > 0.5).squeeze(1).long().to(device) # (B,H,W)

            outputs = model(pixel_values=images, labels=masks_long_for_loss_and_metric)
            loss = outputs.loss
            logits = outputs.logits # These are already upsampled by the model's forward

            if loss is not None: # loss can be None if labels are not provided to model.forward
                 total_loss += loss.item()
            
            predicted_labels = torch.argmax(logits, dim=1) # (B, H, W) tensor
            
            # .add_batch expects numpy arrays
            metric_calculator.add_batch(predictions=predicted_labels.cpu().numpy(), 
                                        references=masks_long_for_loss_and_metric.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    
    eval_results = {}
    try:
        computed_metrics = metric_calculator.compute(
            num_labels=num_labels_for_metric,
            ignore_index=ignore_index_for_metric,
            reduce_labels=False # Very important when using ignore_index
        )
        eval_results.update(computed_metrics)
    except Exception as e:
        print(f"Error computing metrics with evaluate: {e}")
        # Provide default/fallback values if metrics computation fails
        eval_results['mean_iou'] = 0.0 
        eval_results['mean_accuracy'] = 0.0
        if num_labels_for_metric > 0 and 'per_category_iou' not in eval_results:
             eval_results['per_category_iou'] = np.array([0.0] * num_labels_for_metric)


    eval_results['loss'] = avg_loss
    return eval_results


def main(args):
    print(f"Starting DINOv2 finetuning for binary segmentation. Arguments: {args}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    id2label = {0: "background", 1: "new_planting"}
    label2id = {v: k for k, v in id2label.items()}
    num_model_labels = len(id2label) 

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    print("Setting up datasets and dataloaders...")
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])

    train_dataset = BalancedCropDataset(
        data_dir=args.dataset_path,
        patch_size=args.image_size,
        pos_fraction=args.pos_fraction,
        augmentations=train_transforms,
        samples_per_epoch=args.samples_per_epoch
    )
    # For validation, use a fixed set of samples or a deterministic way if possible,
    # but BalancedCropDataset is inherently random. We'll create another instance.
    val_samples_count = max(100, int(args.samples_per_epoch * 0.2))
    val_dataset = BalancedCropDataset(
        data_dir=args.dataset_path,
        patch_size=args.image_size,
        pos_fraction=args.pos_fraction, 
        augmentations=val_transforms,
        samples_per_epoch=val_samples_count
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Train Dataloader: {len(train_dataloader)} batches. Val Dataloader: {len(val_dataloader)} batches.")

    # --- Model ---
    print(f"Loading DINOv2 model: {args.dinov2_model_name}")
    # Load config first, then instantiate custom model with it
    # This way, our custom Dinov2ForSemanticSegmentation inherits from Dinov2PreTrainedModel
    # which handles config and pretrained weight loading correctly if we were to load full model weights.
    # Here, we only load backbone weights.
    pretrained_dinov2_model_for_config = Dinov2Model.from_pretrained(args.dinov2_model_name)
    dinov2_config = pretrained_dinov2_model_for_config.config
    del pretrained_dinov2_model_for_config # Free memory

    model = Dinov2ForSemanticSegmentation(
        config=dinov2_config,
        num_labels=num_model_labels,
        image_size=args.image_size,
        patch_resolution=DINOV2_PATCH_RESOLUTION
    )
    # Load pretrained weights for the dinov2 backbone part
    model.dinov2 = Dinov2Model.from_pretrained(args.dinov2_model_name, config=dinov2_config)
    
    # Freeze DINOv2 backbone
    for name, param in model.dinov2.named_parameters():
        param.requires_grad = False
    print("DINOv2 backbone frozen. Only the classifier head will be trained.")
    
    model.to(device)

    # --- Optimizer ---
    # Only optimize classifier parameters
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW with lr={args.learning_rate} for trainable parameters.")

    # --- Metrics ---
    # Load metric once
    mean_iou_metric = evaluate.load("mean_iou")
    
    best_val_iou = 0.0

    # --- Training Loop ---
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Training]", leave=False)

        for i, (images, masks_float) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = (masks_float > 0.5).squeeze(1).long().to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch+1}, batch {i}. Skipping update.")
                # Optional: Dump batch for inspection
                # torch.save({"images": images.cpu(), "masks": masks_float.cpu()}, f"nan_batch_epoch{epoch+1}_batch{i}.pt")
                # Consider stopping if NaNs are persistent.
                epoch_train_loss += 0 # Or some other handling for averaging
                train_progress_bar.set_postfix(loss='nan')
                continue 
            
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / (len(train_dataloader) - np.isnan(epoch_train_loss).sum()) if len(train_dataloader) > 0 else float('nan') # Avoid division by zero if all batches were nan
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        if val_dataloader and len(val_dataloader) > 0:
            print(f"Epoch {epoch+1} - Starting Validation...")
            # Pass a fresh metric object or ensure reset if `add_batch` is used
            val_metrics_results = evaluate_model(model, val_dataloader, device, evaluate.load("mean_iou"), num_model_labels, label2id["background"])
            
            log_msg = f"Epoch {epoch+1} - Validation Loss: {val_metrics_results['loss']:.4f}"
            if 'mean_iou' in val_metrics_results:
                log_msg += f", Mean IoU: {val_metrics_results['mean_iou']:.4f}"
            if 'mean_accuracy' in val_metrics_results:
                 log_msg += f", Pixel Accuracy: {val_metrics_results['mean_accuracy']:.4f}"
            print(log_msg)
            
            if 'per_category_iou' in val_metrics_results and isinstance(val_metrics_results['per_category_iou'], (np.ndarray, list)) and len(val_metrics_results['per_category_iou']) > label2id["new_planting"]:
                iou_foreground = val_metrics_results['per_category_iou'][label2id['new_planting']]
                print(f"  IoU for 'new_planting' (class 1): {iou_foreground:.4f}")
            else:
                print(f"  IoU for 'new_planting' (class 1): N/A (per_category_iou not found or malformed: {val_metrics_results.get('per_category_iou')})")


            current_val_iou = val_metrics_results.get('mean_iou', 0.0) 
            if isinstance(current_val_iou, np.ndarray): # If it's an array (e.g. from older evaluate versions)
                current_val_iou = current_val_iou.item()

            if current_val_iou > best_val_iou:
                best_val_iou = current_val_iou
                best_model_path = Path(args.output_dir) / "best_model_dinov2.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best validation IoU: {best_val_iou:.4f}. Saved model to {best_model_path}")
        else:
            print(f"Epoch {epoch+1} - No validation performed (validation dataloader empty or not available).")


    # Save final model
    final_model_path = Path(args.output_dir) / "final_model_dinov2.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for binary semantic segmentation.")
    
    parser.add_argument("--dataset_path", type=str, default="data/output", help="Directory with image-mask pairs.")
    parser.add_argument("--output_dir", type=str, default="models/dinov2_binary_segmentation", help="Directory to save models.")
    parser.add_argument("--dinov2_model_name", type=str, default="facebook/dinov2-base", help="Pretrained DINOv2 model name from Hugging Face.")
    
    parser.add_argument("--image_size", type=int, default=224, help="Size to resize images to (must be multiple of DINOv2 patch resolution, e.g., 14).")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.") # Reduced default
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the classifier.") # Reduced default
    parser.add_argument("--pos_fraction", type=float, default=0.5, help="Target fraction of positive samples in BalancedCropDataset.")
    parser.add_argument("--samples_per_epoch", type=int, default=200, help="Number of samples per epoch for BalancedCropDataset.") # Reduced default
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 2), help="Number of DataLoader workers.") # Reduced default
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for training.")
    parser.add_argument("--clip_grad_norm", type=float, default=None, help="Max norm for gradient clipping. Default: None (no clipping). Try 1.0 if NaN loss persists.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")


    args = parser.parse_args()

    if args.image_size % DINOV2_PATCH_RESOLUTION != 0:
        print(f"Warning: --image_size ({args.image_size}) is not perfectly divisible by DINOv2 patch resolution ({DINOV2_PATCH_RESOLUTION}). This might lead to slight mismatches in feature map sizes. Consider sizes like 224, 448, etc.")

    main(args)