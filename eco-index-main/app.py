import pathlib
import io
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.in_memory_crop_dataset import BalancedCropDataset

app = Flask(__name__)

# --- Configuration ---
DATA_DIR = "data/output"
ITEMS_PER_LOAD = 10  # Number of items to load per dynamic request

# --- Global variables ---
crop_dataset = None
dataset_params = {}
dataset_index = 0

def calculate_mask_percentage_tensor(mask_tensor) -> float:
    """Calculates mask coverage for a tensor mask (0..1)."""
    if mask_tensor.numel() == 0:
        return 0.0
    return float(mask_tensor.mean() * 100)

def pil_to_base64(pil_image: Image.Image, image_format="PNG") -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    if pil_image.mode == 'P':
        pil_image = pil_image.convert("RGBA" if image_format == "PNG" else "RGB")
    elif pil_image.mode == 'L' and image_format == "JPEG":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{img_str}"

def build_dataset(params):
    """Build a BalancedCropDataset with augmentations from dict."""
    aug_list = []
    h_prob = float(params.get('flip_h_prob', 0))
    if h_prob > 0:
        aug_list.append(A.HorizontalFlip(p=h_prob))

    v_prob = float(params.get('flip_v_prob', 0))
    if v_prob > 0:
        aug_list.append(A.VerticalFlip(p=v_prob))

    rotate_deg = float(params.get('rotation_deg', 0))
    if rotate_deg > 0:
        aug_list.append(A.Rotate(limit=rotate_deg, p=0.5))

    zoom_min = float(params.get('zoom_min', 1.0))
    zoom_max = float(params.get('zoom_max', 1.0))
    if zoom_min != 1.0 or zoom_max != 1.0:
        aug_list.append(A.Affine(scale=(zoom_min, zoom_max), p=0.5))

    jitter_val = float(params.get('jitter', 0))
    if jitter_val > 0:
        aug_list.append(
            A.ColorJitter(
                brightness=jitter_val,
                contrast=jitter_val,
                saturation=jitter_val,
                hue=0,
                p=0.5,
            )
        )

    aug_list.append(ToTensorV2())
    aug = A.Compose(aug_list)
    return BalancedCropDataset(
        data_dir=DATA_DIR,
        patch_size=params.get('patch_size', 256),
        pos_fraction=params.get('pos_fraction', 0.5),
        augmentations=aug,
        samples_per_epoch=1000000,
    )


@app.route('/')
def browse_page():
    return render_template(
        'browse.html',
        initial_error_message=None,
        patch_size=dataset_params.get('patch_size', 256),
        pos_fraction=dataset_params.get('pos_fraction', 0.5),
        flip_h_prob=dataset_params.get('flip_h_prob', 0.0),
        flip_v_prob=dataset_params.get('flip_v_prob', 0.0),
        rotation_deg=dataset_params.get('rotation_deg', 0.0),
        zoom_min=dataset_params.get('zoom_min', 1.0),
        zoom_max=dataset_params.get('zoom_max', 1.0),
        jitter=dataset_params.get('jitter', 0.0),
        ITEMS_PER_LOAD=ITEMS_PER_LOAD,
        is_ready=True,
    )


@app.route('/load_items')
def load_items_route():
    global crop_dataset, dataset_params, dataset_index

    min_perc = request.args.get('min_perc', 0.0, type=float)
    max_perc = request.args.get('max_perc', 100.0, type=float)

    params = {
        'patch_size': request.args.get('patch_size', 256, type=int),
        'pos_fraction': request.args.get('pos_fraction', 0.5, type=float),
        'flip_h_prob': request.args.get('flip_h_prob', 0.0, type=float),
        'flip_v_prob': request.args.get('flip_v_prob', 0.0, type=float),
        'rotation_deg': request.args.get('rotation_deg', 0.0, type=float),
        'zoom_min': request.args.get('zoom_min', 1.0, type=float),
        'zoom_max': request.args.get('zoom_max', 1.0, type=float),
        'jitter': request.args.get('jitter', 0.0, type=float),
    }

    if crop_dataset is None or params != dataset_params:
        dataset_params = params
        crop_dataset = build_dataset(params)
        dataset_index = 0

    processed_items = []
    tries = 0
    while len(processed_items) < ITEMS_PER_LOAD and tries < ITEMS_PER_LOAD * 10:
        img_tensor, mask_tensor = crop_dataset[dataset_index % len(crop_dataset)]
        dataset_index += 1
        perc = calculate_mask_percentage_tensor(mask_tensor)
        if min_perc <= perc <= max_perc:
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype('uint8')
            mask_np = (mask_tensor.numpy() * 255).astype('uint8')
            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np, mode='L')
            processed_items.append({
                'id': dataset_index,
                'image_b64': pil_to_base64(img_pil, 'JPEG'),
                'mask_b64': pil_to_base64(mask_pil, 'PNG'),
                'mask_percentage_display': f"{perc:.2f}%",
            })
        tries += 1

    html_content = render_template('_items_partial.html', items=processed_items)

    return jsonify({
        'html': html_content,
        'has_more': True,
    })


if __name__ == '__main__':
    dataset_params = {
        'patch_size': 256,
        'pos_fraction': 0.5,
        'flip_h_prob': 0.0,
        'flip_v_prob': 0.0,
        'rotation_deg': 0.0,
        'zoom_min': 1.0,
        'zoom_max': 1.0,
        'jitter': 0.0,
    }
    crop_dataset = build_dataset(dataset_params)
    app.run(debug=True, host='0.0.0.0', port=8888, use_reloader=False)
