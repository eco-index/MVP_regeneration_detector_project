#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Dinov2Config, Dinov2Model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys

# Add project root to sys.path to allow importing finetune_dinov2
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.finetune_dinov2 import Dinov2ForSemanticSegmentation, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DINOV2_PATCH_RESOLUTION

def load_model_for_inference(model_path: str, dinov2_base_model_name: str, num_labels: int, image_size: int, device: str):
    """Loads the finetuned DINOv2 model for inference."""
    print(f"Loading base DINOv2 config for {dinov2_base_model_name}...")
    config = Dinov2Config.from_pretrained(dinov2_base_model_name)

    print("Initializing Dinov2ForSemanticSegmentation model structure...")
    model = Dinov2ForSemanticSegmentation(
        config=config,
        num_labels=num_labels,
        image_size=image_size, # This is the size model expects for its classifier token grid
        patch_resolution=DINOV2_PATCH_RESOLUTION
    )

    print(f"Loading finetuned weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=True) 
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights (strict=True): {e}")
        print("Attempting to load with strict=False...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded with strict=False.")
        except Exception as e2:
            print(f"Failed to load model weights even with strict=False: {e2}")
            return None
            
    model.to(device)
    model.eval()
    return model

def create_patches_for_inference(image_pil: Image.Image, patch_size: int, overlap_px: int):
    """
    Creates overlapping patches from a large image for inference.
    Pads the image if its dimensions are not a multiple of (patch_size - overlap_px)
    to ensure full coverage.
    """
    img_w, img_h = image_pil.size
    stride = patch_size - overlap_px

    # Calculate padding needed
    pad_w = (stride - (img_w - patch_size) % stride) % stride if img_w > patch_size else 0
    pad_h = (stride - (img_h - patch_size) % stride) % stride if img_h > patch_size else 0

    # Pad the image if necessary (e.g., reflect padding)
    # Using numpy for easier padding
    image_np = np.array(image_pil)
    if pad_h > 0 or pad_w > 0:
        image_np_padded = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        padded_image_pil = Image.fromarray(image_np_padded)
    else:
        padded_image_pil = image_pil
    
    padded_w, padded_h = padded_image_pil.size
    
    patches_info = []
    for y_start in range(0, padded_h - patch_size + 1, stride):
        for x_start in range(0, padded_w - patch_size + 1, stride):
            patch_pil = padded_image_pil.crop((x_start, y_start, x_start + patch_size, y_start + patch_size))
            patches_info.append({
                'patch_pil': patch_pil,
                'x_start': x_start,
                'y_start': y_start,
                'patch_size_w': patch_pil.width, # Should be patch_size unless at edge of padded
                'patch_size_h': patch_pil.height
            })
    print(f"Created {len(patches_info)} patches of size {patch_size}x{patch_size} with stride {stride} from padded image ({padded_w}x{padded_h}).")
    return patches_info, (padded_w, padded_h)


def preprocess_patch(patch_pil: Image.Image, image_size_model_expects: int, device: str):
    """Prepares a single patch for DINOv2 inference."""
    # The model's classifier was defined with image_size, so patches should be this size.
    transform = A.Compose([
        A.Resize(height=image_size_model_expects, width=image_size_model_expects), # Ensure patch is the size model was trained on
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])
    patch_np = np.array(patch_pil.convert("RGB"))
    augmented = transform(image=patch_np)
    patch_tensor = augmented['image'].unsqueeze(0).to(device) # Add batch dimension
    return patch_tensor

def run_inference_on_patch(model, patch_tensor: torch.Tensor):
    """Runs inference on a single preprocessed patch tensor.
       Returns raw logits at the model's classifier output resolution.
    """
    with torch.no_grad():
        # The model's forward pass internally upsamples logits to `self.image_size_h, self.image_size_w`
        # which corresponds to `args.image_size` from training.
        outputs = model(pixel_values=patch_tensor) 
        logits = outputs.logits # Shape: (1, num_classes, model_image_size, model_image_size)
    return logits


def stitch_masks_from_patches(
    patches_info: list, 
    final_mask_logits_shape: tuple, # (num_classes, padded_h, padded_w)
    original_img_w: int, 
    original_img_h: int,
    patch_size_model_output: int, # The H/W dimension of the logits from each patch (e.g., args.image_size)
    overlap_px: int
    ):
    """
    Stitches individual patch logits/probabilities into a single full-resolution probability map,
    then converts to binary mask. Handles overlaps by averaging probabilities.
    """
    num_classes, padded_h, padded_w = final_mask_logits_shape
    
    # Accumulators for logits and counts (for averaging in overlap regions)
    sum_logits_map = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    stride = patch_size_model_output - overlap_px

    for patch_data in tqdm(patches_info, desc="Stitching Patches"):
        patch_logits = patch_data['logits_output'].cpu().numpy() # (1, num_classes, H_patch_out, W_patch_out)
        patch_logits = patch_logits.squeeze(0) # (num_classes, H_patch_out, W_patch_out)

        x_start, y_start = patch_data['x_start'], patch_data['y_start']
        
        # Ensure patch_logits are resized to actual patch_size if they came from fixed model output size
        # This happens if patch_pil was smaller than patch_size_model_output due to being at image edge
        # However, our create_patches_for_inference pads to ensure all patches are patch_size
        # and preprocess_patch resizes to image_size_model_expects. So, patch_logits should already be at this size.
        
        # Define the window for placing this patch's contribution
        y_end_in_map = y_start + patch_size_model_output
        x_end_in_map = x_start + patch_size_model_output

        sum_logits_map[:, y_start:y_end_in_map, x_start:x_end_in_map] += patch_logits
        count_map[y_start:y_end_in_map, x_start:x_end_in_map] += 1

    # Average the logits in overlapping regions
    # Add epsilon to count_map to avoid division by zero where count is 0 (should not happen with proper patching)
    avg_logits_map = sum_logits_map / (count_map[None, :, :] + 1e-8) 

    # Convert averaged logits to probabilities and then to binary mask
    # Taking argmax over the class dimension
    predicted_labels_map_padded = np.argmax(avg_logits_map, axis=0) # (padded_h, padded_w)
    
    # Crop back to original image dimensions
    binary_mask_np = predicted_labels_map_padded[:original_img_h, :original_img_w].astype(np.uint8)
    
    return binary_mask_np


def overlay_mask_on_image(original_image_pil: Image.Image, mask_np: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.4):
    """Overlays a binary mask on the original image."""
    if mask_np.shape != (original_image_pil.height, original_image_pil.width):
        # This should not happen if stitching and cropping are correct
        print(f"Critical Error: Mask shape {mask_np.shape} after stitching and cropping "
              f"differs from image shape {(original_image_pil.height, original_image_pil.width)}.")
        # Fallback: resize mask, but this indicates a bug in stitching/cropping
        mask_pil = Image.fromarray(mask_np * 255) 
        mask_pil = mask_pil.resize(original_image_pil.size, Image.NEAREST)
        mask_np_resized = np.array(mask_pil) // 255
    else:
        mask_np_resized = mask_np

    overlay_img_rgba = original_image_pil.convert("RGBA")
    mask_color_rgba = Image.new("RGBA", original_image_pil.size, (*color, int(alpha * 255)))
    mask_for_composite_pil = Image.fromarray((mask_np_resized * 255).astype(np.uint8), mode='L')
    overlay_img_rgba.paste(mask_color_rgba, (0,0), mask_for_composite_pil)
    return overlay_img_rgba.convert("RGB")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    num_labels = 2 # Background, New Planting

    model = load_model_for_inference(
        model_path=args.model_path,
        dinov2_base_model_name=args.dinov2_model_name,
        num_labels=num_labels,
        image_size=args.image_size, # This is the size the model's classifier expects patches at
        device=str(device)
    )
    if not model:
        print("Failed to load model. Exiting.")
        return

    try:
        image_pil_original = Image.open(args.image_path).convert("RGB")
        original_w, original_h = image_pil_original.size
        print(f"Loaded image: {args.image_path}, Original size: ({original_w}, {original_h})")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Create patches for inference
    # args.image_size is the size each patch will be resized to before feeding to the model
    # args.patch_size_inference is the actual crop size from the large image
    patches_info, (padded_w, padded_h) = create_patches_for_inference(
        image_pil_original, 
        args.patch_size_inference, 
        args.overlap_px
    )

    if not patches_info:
        print("No patches created. Exiting.")
        return

    print("Running inference on patches...")
    for patch_data in tqdm(patches_info, desc="Inferring Patches"):
        patch_tensor = preprocess_patch(patch_data['patch_pil'], args.image_size, device)
        # logits_output from model will be (1, num_classes, args.image_size, args.image_size)
        patch_data['logits_output'] = run_inference_on_patch(model, patch_tensor)
    
    print("Stitching masks...")
    # The stitched mask's logits will be at the resolution of args.image_size per patch area
    # So the full padded logits map will have dimensions based on args.image_size for each patch slot
    # However, the stitch function needs to know the size of the elements being stitched (patch_size_model_output)
    final_mask_np = stitch_masks_from_patches(
        patches_info,
        final_mask_logits_shape=(num_labels, padded_h, padded_w), # This assumes logits are stitched at original padded res
                                                                  # but patch_logits are at args.image_size
                                                                  # This needs correction. We stitch at patch_size_model_output resolution for each patch.
                                                                  # Let's adjust.
                                                                  # stitch_masks needs to know the output resolution of each patch's mask.
        original_img_w=original_w,
        original_img_h=original_h,
        patch_size_model_output=args.image_size, # Logits from each patch are at this resolution
        overlap_px=args.overlap_px * (args.image_size / args.patch_size_inference) # Scale overlap to model output resolution
    )
    
    print(f"Masks stitched. Final mask shape: {final_mask_np.shape}, Non-zero pixels: {np.count_nonzero(final_mask_np)}")

    if args.mask_output_path:
        try:
            mask_to_save_pil = Image.fromarray(final_mask_np * 255)
            mask_to_save_pil.save(args.mask_output_path)
            print(f"Raw binary mask saved to: {args.mask_output_path}")
        except Exception as e:
            print(f"Error saving raw mask: {e}")

    output_image_pil = overlay_mask_on_image(image_pil_original, final_mask_np)
    
    if args.output_path:
        try:
            output_image_pil.save(args.output_path)
            print(f"Output image with overlay saved to: {args.output_path}")
        except Exception as e:
            print(f"Error saving output image: {e}")
    else:
        print("Displaying output image (output_path not provided).")
        try:
            output_image_pil.show()
        except Exception as e:
            print(f"Error displaying image: {e}. Try providing --output_path to save it.")

    print("Patch-based inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run patch-based inference with a finetuned DINOv2 model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("model_path", type=str, help="Path to the finetuned .pth model file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output image with mask overlay.")
    parser.add_argument("--mask_output_path", type=str, default=None, help="Path to save the raw binary mask.")
    
    parser.add_argument("--dinov2_model_name", type=str, default="facebook/dinov2-base", help="Base DINOv2 model name used for config.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size the model's classifier was trained with (patches are resized to this).")
    parser.add_argument("--patch_size_inference", type=int, default=224, help="Size of square patches to crop from the large input image.")
    parser.add_argument("--overlap_px", type=int, default=32, help="Pixel overlap between adjacent patches for cropping.")
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference.")

    args = parser.parse_args()
    
    if args.patch_size_inference < args.image_size:
        print(f"Warning: patch_size_inference ({args.patch_size_inference}) is smaller than image_size ({args.image_size}). "
              f"Patches will be upscaled to {args.image_size} before model input. This might not be ideal.")
    if args.overlap_px >= args.patch_size_inference:
        raise ValueError("overlap_px must be less than patch_size_inference.")

    main(args)