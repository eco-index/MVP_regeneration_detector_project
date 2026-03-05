import os
import torch
from transformers import SamModel, AutoProcessor
import argparse
from PIL import Image
import numpy as np

def load_model(model_name_or_path: str, device_str: str, base_model_name: str = "facebook/sam-vit-base"):
    """Load a SAM model for inference.

    This helper tries three strategies:
    1. Load the model directly via ``from_pretrained`` (for Hugging Face style directories).
    2. If ``model_name_or_path`` is a directory containing ``*.pth`` weights, load those
       weights into ``base_model_name``.
    3. If ``model_name_or_path`` is a ``.pth`` file, load it into ``base_model_name``.

    Args:
        model_name_or_path: Hugging Face model name or path to weights directory/file.
        device_str: Torch device string.
        base_model_name: Base model to instantiate when loading ``.pth`` weights.

    Returns:
        Tuple[SamModel, AutoProcessor] or ``(None, None)`` on failure.
    """

    # First attempt: treat the path as a Hugging Face model directory/name
    try:
        model = SamModel.from_pretrained(model_name_or_path)
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model.to(device_str)
        print(f"Model {model_name_or_path} loaded successfully on {device_str}.")
        return model, processor
    except Exception as hf_error:
        print(f"Direct model load failed for {model_name_or_path}: {hf_error}")

    # Second attempt: load from a directory containing .pth weights
    if os.path.isdir(model_name_or_path):
        pth_files = [f for f in os.listdir(model_name_or_path) if f.endswith(".pth")]
        if pth_files:
            # Prefer best_model.pth or final_model.pth if present
            preferred = ["best_model.pth", "final_model.pth"]
            chosen = None
            for pref in preferred:
                if pref in pth_files:
                    chosen = pref
                    break
            if not chosen:
                chosen = pth_files[0]
            weights_path = os.path.join(model_name_or_path, chosen)
            try:
                print(f"Loading weights from {weights_path} using base model {base_model_name}.")
                model = SamModel.from_pretrained(base_model_name)
                state_dict = torch.load(weights_path, map_location=device_str)
                model.load_state_dict(state_dict)
                model.to(device_str)
                processor = AutoProcessor.from_pretrained(base_model_name)
                return model, processor
            except Exception as e:
                print(f"Error loading weights from {weights_path}: {e}")

    # Third attempt: model_name_or_path is a single .pth file
    if os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".pth"):
        try:
            print(f"Loading weights from {model_name_or_path} using base model {base_model_name}.")
            model = SamModel.from_pretrained(base_model_name)
            state_dict = torch.load(model_name_or_path, map_location=device_str)
            model.load_state_dict(state_dict)
            model.to(device_str)
            processor = AutoProcessor.from_pretrained(base_model_name)
            return model, processor
        except Exception as e:
            print(f"Error loading weights from {model_name_or_path}: {e}")

    print(f"Failed to load model from {model_name_or_path}.")
    return None, None

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the given path and converts it to RGB.

    Args:
        image_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image object, or None if loading fails.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully from {image_path}.")
        return image
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_patches(image: Image.Image, patch_size: int, overlap_px: int) -> list[dict]:
    """
    Creates overlapping patches from an image.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The desired square dimension of each patch.
        overlap_px (int): The number of pixels to overlap between adjacent patches.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    the patch image and its coordinates in the original image.
    """
    patches = []
    img_width, img_height = image.size
    stride = patch_size - overlap_px

    for y_start in range(0, img_height, stride):
        for x_start in range(0, img_width, stride):
            # Define patch boundaries
            x_end = min(x_start + patch_size, img_width)
            y_end = min(y_start + patch_size, img_height)

            # Crop the patch
            patch_image = image.crop((x_start, y_start, x_end, y_end))
            
            patches.append({
                'patch_image': patch_image,
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
            })
    
    print(f"Created {len(patches)} patches.")
    return patches

def run_inference_on_patch(
    patch_image: Image.Image, 
    model, 
    processor, 
    device: str, 
    target_model_input_size: int = 1024
) -> np.ndarray:
    """
    Runs SAM inference on a single image patch.

    Args:
        patch_image (PIL.Image.Image): The PIL Image for the current patch.
        model: The loaded SAM model.
        processor: The loaded SAM processor.
        device (str): The torch device ('cuda' or 'cpu').
        target_model_input_size (int): The size SAM's image encoder expects.

    Returns:
        np.ndarray: The binary mask as a NumPy array (uint8), or None if inference fails.
    """
    try:
        original_patch_width, original_patch_height = patch_image.size

        # Preprocess the patch_image
        inputs = processor(images=[patch_image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        model.eval()

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, multimask_output=False)

        predicted_logits = outputs.pred_masks

        # ``predicted_logits`` may include singleton dimensions depending on the
        # underlying model configuration (e.g. [B, 1, 1, H, W]).  To ensure
        # ``torch.nn.functional.interpolate`` receives a 4D tensor in
        # (N, C, H, W) format, squeeze any extra dimensions and then add the
        # channel dimension back if necessary.
        predicted_logits = predicted_logits.squeeze()
        if predicted_logits.ndim == 2:
            predicted_logits = predicted_logits.unsqueeze(0).unsqueeze(0)
        elif predicted_logits.ndim == 3:
            predicted_logits = predicted_logits.unsqueeze(1)

        # Upsample predicted mask logits to target_model_input_size
        # predicted_logits shape should be (batch_size, num_masks, H, W)
        upscaled_logits = torch.nn.functional.interpolate(
            predicted_logits,
            size=(target_model_input_size, target_model_input_size),
            mode="bilinear",
            align_corners=False,
        )

        # Apply sigmoid and threshold
        probs = torch.sigmoid(upscaled_logits)
        binary_mask_at_model_res = (probs > 0.5).squeeze() # Shape: (target_model_input_size, target_model_input_size)

        # Resize binary mask back to original patch dimensions
        mask_to_resize = binary_mask_at_model_res.float().unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)
        resized_binary_mask_tensor = torch.nn.functional.interpolate(
            mask_to_resize,
            size=(original_patch_height, original_patch_width),
            mode="nearest"
        ).squeeze() # Shape: (original_patch_height, original_patch_width)

        # Convert to NumPy array
        final_mask_np = resized_binary_mask_tensor.cpu().numpy().astype(np.uint8)
        return final_mask_np

    except Exception as e:
        print(f"Error during inference on patch: {e}")
        return None

def stitch_masks(patches_with_masks: list[dict], original_width: int, original_height: int) -> np.ndarray:
    """
    Stitches individual patch masks into a single full-resolution mask.

    Args:
        patches_with_masks (list[dict]): List of patch dictionaries, each including
                                         'mask_array', 'x_start', 'y_start', 'x_end', 'y_end'.
        original_width (int): Width of the original input image.
        original_height (int): Height of the original input image.

    Returns:
        np.ndarray: The final stitched binary mask for the entire image.
    """
    full_mask = np.zeros((original_height, original_width), dtype=np.uint8)

    for patch_data in patches_with_masks:
        if 'mask_array' not in patch_data:
            print(f"Warning: Patch at ({patch_data['x_start']}, {patch_data['y_start']}) is missing 'mask_array'. Skipping.")
            continue

        mask_array = patch_data['mask_array']
        x_start, y_start = patch_data['x_start'], patch_data['y_start']
        # x_end and y_end from patch_data are for the original image space
        # mask_array is already the correct size for this patch
        x_end = x_start + mask_array.shape[1] 
        y_end = y_start + mask_array.shape[0]

        # Ensure we don't write out of bounds if something went wrong with patch/mask sizes
        # This primarily protects against mask_array being larger than the space defined by x_start/y_start to original_width/height
        effective_x_end = min(x_end, original_width)
        effective_y_end = min(y_end, original_height)
        
        # Slice the mask_array if it would go out of bounds of the full_mask
        mask_to_place = mask_array[0:(effective_y_end - y_start), 0:(effective_x_end - x_start)]

        current_mask_in_full = full_mask[y_start:effective_y_end, x_start:effective_x_end]
        
        if current_mask_in_full.shape != mask_to_place.shape:
            print(f"Warning: Shape mismatch when stitching. ROI: {current_mask_in_full.shape}, Mask: {mask_to_place.shape}. Patch: ({x_start},{y_start}). Skipping.")
            continue

        updated_roi = np.maximum(current_mask_in_full, mask_to_place)
        full_mask[y_start:effective_y_end, x_start:effective_x_end] = updated_roi
        
    print(f"Stitched full mask of shape: {full_mask.shape}")
    return full_mask

def overlay_mask_on_image(original_image: Image.Image, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5) -> Image.Image:
    """
    Overlays a binary mask on the original image with a specified color and transparency.

    Args:
        original_image (PIL.Image.Image): The original input PIL Image.
        mask (np.ndarray): The binary mask (0s and 1s) with the same dimensions as original_image.
        color (tuple): RGB tuple for the overlay color.
        alpha (float): Transparency for the overlay (0.0 to 1.0).

    Returns:
        PIL.Image.Image: The image with the mask overlaid.
    """
    overlay_image = original_image.convert("RGBA")
    
    # Create an RGBA numpy array for the colored mask
    colored_mask_np = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    
    # Where mask is 1, set the color and alpha
    colored_mask_np[mask == 1] = [color[0], color[1], color[2], int(alpha * 255)]
    
    # Convert this numpy array to a PIL Image
    color_layer = Image.fromarray(colored_mask_np, 'RGBA')
    
    # Composite the color_layer onto the overlay_image
    output_image = Image.alpha_composite(overlay_image, color_layer)
    
    return output_image

if __name__ == "__main__":
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser(description="Segment an image using a SAM model with patch-based inference.")
    
    # Required arguments
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="Name of the SAM model from Hugging Face Model Hub or path to a local fine-tuned model directory.",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output image with the mask overlay. If not provided, the image will be displayed. (Default: None)",
    )
    parser.add_argument(
        "--mask_output_path",
        type=str,
        default=None,
        help="Optional path to save the raw binary mask image before the overlay is applied. (Default: None)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the square patches to divide the image into. (Default: 256)",
    )
    parser.add_argument(
        "--overlap_px",
        type=int,
        default=32,
        help="Pixel overlap between adjacent patches. (Default: 32)",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=256,
        help="The target input size for the SAM model's image encoder (e.g., 1024 for SAM). (Default: 1024)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"Device to use for inference (e.g., 'cuda', 'cpu'). (Default: {default_device})",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="facebook/sam-vit-base",
        help="Base SAM model to use when loading weights from a .pth file. (Default: facebook/sam-vit-base)",
    )
    
    args = parser.parse_args()

    print("Starting SAM image segmentation process...")
    print(f"Arguments: Image Path: {args.image_path}, Model: {args.model_name_or_path}, Device: {args.device}")

    print("Loading model...")
    model, processor = load_model(
        args.model_name_or_path,
        args.device,
        base_model_name=args.base_model_name,
    )

    if model is None or processor is None:
        print("Exiting due to model loading failure.")
    else:
        print("Model loaded successfully.")
        print(f"Loading image from: {args.image_path}")
        image = load_image(args.image_path)

        if image is None:
            print("Exiting due to image loading failure.")
        else:
            print(f"Image loaded successfully. Dimensions: {image.size}")
            print(f"Creating patches: Patch Size: {args.patch_size}, Overlap: {args.overlap_px}px")
            patch_details_list = create_patches(image, args.patch_size, args.overlap_px)
            
            if not patch_details_list:
                print("No patches were created. Exiting.")
            else:
                print(f"Successfully created {len(patch_details_list)} patches.")
                print("Running inference on patches...")
                
                successful_patches = 0
                for i, patch_detail in enumerate(patch_details_list):
                    print(f"  Processing patch {i+1}/{len(patch_details_list)} at ({patch_detail['x_start']},{patch_detail['y_start']})")
                    
                    mask_array = run_inference_on_patch(
                        patch_image=patch_detail['patch_image'],
                        model=model,
                        processor=processor,
                        device=args.device,
                        target_model_input_size=args.model_input_size
                    )
                    
                    if mask_array is not None:
                        patch_detail['mask_array'] = mask_array
                        successful_patches += 1
                        if successful_patches == 1: # Log details for the first successful mask
                             print(f"    First patch mask generated. Shape: {mask_array.shape}, dtype: {mask_array.dtype}")
                    else:
                        print(f"    Failed to generate mask for patch {i+1}.")
                
                if successful_patches == 0:
                    print("No masks were generated from any patches. Exiting.")
                else:
                    print(f"Successfully generated masks for {successful_patches}/{len(patch_details_list)} patches.")
                    print("Stitching masks...")
                    original_img_width, original_img_height = image.size
                    final_stitched_mask = stitch_masks(patch_details_list, original_img_width, original_img_height)
                    print(f"Masks stitched. Final mask shape: {final_stitched_mask.shape}, Non-zero pixels: {np.count_nonzero(final_stitched_mask)}")


                    if args.mask_output_path:
                        try:
                            mask_image_to_save = Image.fromarray((final_stitched_mask * 255).astype(np.uint8))
                            mask_image_to_save.save(args.mask_output_path)
                            print(f"Mask image saved to: {args.mask_output_path}")
                        except Exception as e:
                            print(f"Error saving mask image to {args.mask_output_path}: {e}")

                    print("Overlaying mask on the original image...")
                    output_image_with_overlay = overlay_mask_on_image(image, final_stitched_mask)

                    if args.output_path:
                        try:
                            output_image_with_overlay.save(args.output_path)
                            print(f"Output image with mask overlay saved to: {args.output_path}")
                        except Exception as e:
                            print(f"Error saving output image to {args.output_path}: {e}")
                    else:
                        print("Displaying image with mask overlay (output_path not provided).")
                        try:
                            output_image_with_overlay.show()
                        except Exception as e:
                            print(f"Error displaying image: {e}. (Note: Displaying images might depend on your environment/OS setup).")
    print("SAM image segmentation process finished.")
