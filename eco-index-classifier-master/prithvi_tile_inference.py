#!/usr/bin/env python3
"""Run a Prithvi checkpoint over tiled imagery using overlapping crops."""
import argparse
import math
import random
import threading
import queue
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.serialization
from tqdm import tqdm

from prithvi.model import PrithviSegmentationModel
from prithvi.utils import ensure_dir

try:
    from torch._dynamo.eval_frame import OptimizedModule  # type: ignore

    torch.serialization.add_safe_globals([OptimizedModule])
except Exception:  # pragma: no cover - best effort
    pass

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prithvi tiled inference with overlapping crops")
    parser.add_argument(
        "--image-dir",
        default="Waiwhakaiho_grid/Waiwhakaiho_grid",
        help="Directory containing tile images",
    )
    parser.add_argument(
        "--weights",
        default="runs/prithvi_20251030-124604/checkpoints/epoch9.pt",
        help="Path to the checkpoint (.pt) to load",
    )
    parser.add_argument(
        "--output-dir",
        default="waiwhakaiho_predictions",
        help="Directory to store probability maps and binary masks",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Crop (tile) size for inference",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride between overlapping crops",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting probabilities into binary masks",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when clip mode is enabled",
    )
    parser.add_argument(
        "--clip",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="If >0, randomly process only this many tiles (default 0 = use all tiles)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of crops to process together on the GPU",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on the Prithvi model for potential speedups",
    )
    parser.add_argument(
        "--compile-cache",
        default=None,
        help="Optional path to store/load the compiled model artifact",
    )
    return parser.parse_args()


def find_images(image_dir: Path) -> List[Path]:
    paths = [p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS and p.is_file()]
    return sorted(paths)


def compute_steps(size: int, window: int, stride: int) -> List[int]:
    if size <= window:
        return [0]
    steps = list(range(0, size - window + 1, stride))
    if steps[-1] != size - window:
        steps.append(size - window)
    return steps


def get_core_model(model):
    return getattr(model, "_orig_mod", model)


def configure_position_embeddings(
    model,
    window_size: int,
    device: torch.device,
    disable_auto: bool,
):
    core = get_core_model(model)
    if not hasattr(core, "_ensure_backbone_pos_embed"):
        return

    with torch.no_grad():
        dummy = torch.zeros(1, 3, window_size, window_size, device=device)
        prepared = core._prepare_input(dummy)
        core._ensure_backbone_pos_embed(prepared.shape[-3:], prepared.device)

    if disable_auto and hasattr(core, "auto_pos_embed"):
        core.auto_pos_embed = False


def estimate_num_batches(height: int, width: int, window: int, stride: int, batch_size: int) -> int:
    padded_h = max(height, window)
    padded_w = max(width, window)
    y_steps = compute_steps(padded_h, window, stride)
    x_steps = compute_steps(padded_w, window, stride)
    total_tiles = len(y_steps) * len(x_steps)
    return max(1, math.ceil(total_tiles / batch_size))


def load_model(weights_path: Path, device: torch.device) -> PrithviSegmentationModel:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = PrithviSegmentationModel()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _resolve_cache_path(weights_path: Path, window_size: int, cache_arg: Optional[str]) -> Path:
    if cache_arg:
        return Path(cache_arg)
    cache_dir = Path("runs") / "compiled_models"
    ensure_dir(cache_dir)
    cache_name = f"{weights_path.stem}_ws{window_size}.ptc"
    return cache_dir / cache_name


def _try_load_compiled(cache_path: Path, device: torch.device, weights_path: Path, window: int):
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location=device, weights_only=False)
        meta = payload.get("meta", {})
        expected_weight = str(weights_path.resolve())
        if meta.get("weights") != expected_weight or meta.get("window") != window:
            raise RuntimeError("cache metadata mismatch")
        compiled_model = payload["model"]
        compiled_model.to(device)
        compiled_model.eval()
        print(f"Loaded compiled model from {cache_path}")
        return compiled_model
    except Exception as exc:
        print(f"Failed to load compiled cache ({exc}). Recompiling...")
        return None


def _compile_and_cache_model(
    model: PrithviSegmentationModel,
    cache_path: Path,
    device: torch.device,
    window_size: int,
    weights_path: Path,
) -> PrithviSegmentationModel:
    compiled_model = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, window_size, window_size, device=device)
        _ = compiled_model(dummy)
    ensure_dir(cache_path.parent)
    payload = {
        "model": compiled_model,
        "meta": {
            "weights": str(weights_path.resolve()),
            "window": window_size,
        },
    }
    torch.save(payload, cache_path)
    print(f"Saved compiled model to {cache_path}")
    compiled_model.eval()
    return compiled_model


def run_model_on_batch(
    model: PrithviSegmentationModel, batch_tensor: torch.Tensor, device: torch.device
) -> np.ndarray:
    use_cuda = device.type == "cuda"

    if use_cuda:
        compute_stream = torch.cuda.Stream(device=device)
        default_stream = torch.cuda.current_stream(device=device)

        with torch.cuda.stream(compute_stream):
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch_tensor)
            probs = torch.sigmoid(logits).detach()

        default_stream.wait_stream(compute_stream)
        return probs.squeeze(1).cpu().numpy()

    with nullcontext():
        batch_tensor = batch_tensor.to(device)
        logits = model(batch_tensor)
        probs = torch.sigmoid(logits).detach()
    return probs.squeeze(1).cpu().numpy()


def tile_inference(
    model: PrithviSegmentationModel,
    image: np.ndarray,
    window_size: int,
    stride: int,
    device: torch.device,
    batch_size: int,
    batch_progress: Optional[tqdm] = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    pad_bottom = max(0, window_size - height)
    pad_right = max(0, window_size - width)
    if pad_bottom or pad_right:
        padded = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    else:
        padded = image

    padded_h, padded_w = padded.shape[:2]
    prob_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)

    y_steps = compute_steps(padded_h, window_size, stride)
    x_steps = compute_steps(padded_w, window_size, stride)

    coords = [(y, x) for y in y_steps for x in x_steps]

    prefetch_queue: "queue.Queue[Optional[tuple[List[tuple[int, int]], torch.Tensor]]]" = queue.Queue(maxsize=4)
    stop_token = object()

    def prepare_batches():
        try:
            for start in range(0, len(coords), batch_size):
                batch_coords = coords[start : start + batch_size]
                crops = [padded[y : y + window_size, x : x + window_size] for y, x in batch_coords]
                batch = np.stack(crops, axis=0)
                batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
                if device.type == "cuda":
                    batch_tensor = batch_tensor.pin_memory()
                prefetch_queue.put((batch_coords, batch_tensor))
        except Exception as exc:  # pragma: no cover - surfacing errors
            prefetch_queue.put(exc)
        finally:
            prefetch_queue.put(stop_token)

    worker = threading.Thread(target=prepare_batches, daemon=True)
    worker.start()

    while True:
        item = prefetch_queue.get()
        if item is stop_token:
            break
        if isinstance(item, Exception):
            worker.join()
            raise item
        batch_coords, batch_tensor = item
        probs_batch = run_model_on_batch(model, batch_tensor, device)

        for (y, x), probs in zip(batch_coords, probs_batch):
            y_end = y + window_size
            x_end = x + window_size
            prob_sum[y:y_end, x:x_end] += probs
            weight_sum[y:y_end, x:x_end] += 1.0
        if batch_progress is not None:
            batch_progress.update(1)

    worker.join()

    prob_map = prob_sum / np.clip(weight_sum, 1e-6, None)
    prob_map = prob_map[:height, :width]
    return np.clip(prob_map, 0.0, 1.0)


def save_outputs(prob_map: np.ndarray, out_root: Path, stem: str, threshold: float) -> None:
    prob_dir = out_root / "probabilities"
    mask_dir = out_root / "masks"
    ensure_dir(prob_dir)
    ensure_dir(mask_dir)

    prob_img = (prob_map * 255.0).round().astype(np.uint8)
    mask_img = (prob_map >= threshold).astype(np.uint8) * 255

    Image.fromarray(prob_img).save(prob_dir / f"{stem}_prob.png")
    Image.fromarray(mask_img).save(mask_dir / f"{stem}_mask.png")


def already_processed(out_root: Path, stem: str) -> bool:
    """Check whether both probability and mask outputs exist for the stem."""
    prob_path = out_root / "probabilities" / f"{stem}_prob.png"
    mask_path = out_root / "masks" / f"{stem}_mask.png"
    return prob_path.exists() and mask_path.exists()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.window_size <= 0:
        raise ValueError("window-size must be positive")
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.stride > args.window_size:
        raise ValueError("stride must be less than or equal to window-size to ensure overlap")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = find_images(image_dir)
    if not images:
        raise RuntimeError(f"No images with extensions {sorted(IMG_EXTENSIONS)} found in {image_dir}")

    if args.clip > 0:
        clip_count = min(args.clip, len(images))
        images = random.sample(images, clip_count)
        images.sort()
        plural = "tile" if clip_count == 1 else "tiles"
        print(f"Clip mode enabled -> running {clip_count} {plural}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("CUDA requested but not available. Falling back to CPU.")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = load_model(weights_path, device)
    if args.compile:
        configure_position_embeddings(model, args.window_size, device, disable_auto=True)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        cache_path = _resolve_cache_path(weights_path, args.window_size, args.compile_cache)
        compiled_model = _try_load_compiled(cache_path, device, weights_path, args.window_size)
        if compiled_model is None:
            try:
                compiled_model = _compile_and_cache_model(
                    model,
                    cache_path,
                    device,
                    args.window_size,
                    weights_path,
                )
                print("torch.compile enabled for Prithvi model")
            except Exception as exc:  # pragma: no cover - fallback path
                print(f"torch.compile failed ({exc}); falling back to eager mode")
                model = load_model(weights_path, device)
                compiled_model = None
        if compiled_model is not None:
            model = compiled_model
    configure_position_embeddings(model, args.window_size, device, disable_auto=args.compile)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    ensure_dir(out_root / "probabilities")
    ensure_dir(out_root / "masks")

    for image_path in tqdm(images, desc="Tiles", position=0):
        if already_processed(out_root, image_path.stem):
            tqdm.write(f"Skipping {image_path.name} (already processed)")
            continue
        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))
        batch_total = estimate_num_batches(
            image.shape[0], image.shape[1], args.window_size, args.stride, args.batch_size
        )
        # batch_bar = tqdm(
        #     total=batch_total,
        #     desc=image_path.name,
        #     position=1,
        #     leave=False,
        # )
        prob_map = tile_inference(
            model,
            image,
            args.window_size,
            args.stride,
            device,
            args.batch_size,
            # batch_progress=batch_bar,
        )
        save_outputs(prob_map, out_root, image_path.stem, args.threshold)
        # batch_bar.close()

    print(f"Saved outputs to {out_root}")


if __name__ == "__main__":
    main()
