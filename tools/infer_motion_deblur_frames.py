#!/usr/bin/env python3
"""Run a Turtle video-restoration model on one ordered frame directory."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basicsr.utils.options import parse


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--arch",
        choices=("t0", "t1"),
        help="Turtle architecture; defaults to the architecture named by the config.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("full", "tiled"), default="full")
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_frame(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return pil_to_tensor(image).float().div_(255.0)


def save_frame_atomic(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .clamp_(0, 1)
        .mul_(255)
        .round_()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def run_full(
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    device: torch.device,
    k_cache,
    v_cache,
):
    height, width = current.shape[-2:]
    model_input = torch.stack((previous, current), dim=0).unsqueeze(0).to(device)
    output, k_cache, v_cache = model(model_input, k_cache, v_cache)
    output = output.squeeze(0)[..., :height, :width].cpu()
    return output, k_cache, v_cache


def patch_indices(length: int, tile: int, stride: int) -> list[int]:
    if tile >= length:
        return [0]
    indices = list(range(0, length - tile, stride))
    if not indices or indices[-1] != length - tile:
        indices.append(length - tile)
    return indices


def run_tiled(
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    device: torch.device,
    tile: int,
    overlap: int,
    k_cache,
    v_cache,
):
    if tile <= overlap:
        raise ValueError("tile must be greater than tile-overlap")
    if tile % 32:
        raise ValueError("tile must be divisible by 32")

    _, height, width = current.shape
    tile = min(tile, height, width)
    tile -= tile % 32
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("effective tile must be greater than tile-overlap")

    h_indices = patch_indices(height, tile, stride)
    w_indices = patch_indices(width, tile, stride)
    accumulator = torch.zeros(1, 3, height, width)
    weights = torch.zeros_like(accumulator)
    next_k_cache: dict[str, list[torch.Tensor | None]] = {}
    next_v_cache: dict[str, list[torch.Tensor | None]] = {}

    for h_index in h_indices:
        for w_index in w_indices:
            key = f"{h_index}-{w_index}"
            previous_patch = previous[:, h_index : h_index + tile, w_index : w_index + tile]
            current_patch = current[:, h_index : h_index + tile, w_index : w_index + tile]
            model_input = torch.stack((previous_patch, current_patch), dim=0).unsqueeze(0).to(device)

            old_k = None
            old_v = None
            if k_cache is not None and key in k_cache:
                old_k = [value.to(device) if value is not None else None for value in k_cache[key]]
                old_v = [value.to(device) if value is not None else None for value in v_cache[key]]

            output, new_k, new_v = model(model_input, old_k, old_v)
            output = output.detach().cpu()
            next_k_cache[key] = [value.detach().cpu() if value is not None else None for value in new_k]
            next_v_cache[key] = [value.detach().cpu() if value is not None else None for value in new_v]
            accumulator[..., h_index : h_index + tile, w_index : w_index + tile].add_(output)
            weights[..., h_index : h_index + tile, w_index : w_index + tile].add_(1)

    if torch.any(weights == 0):
        raise RuntimeError("tiled inference left uncovered pixels")
    return accumulator.div_(weights).squeeze(0), next_k_cache, next_v_cache


def main() -> None:
    args = parse_args()
    files = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"No input images found in {args.input_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    options = parse(str(args.config), is_train=True)
    model_name = str(options.get("model", "Turtle_t1_arch")).lower()
    if args.arch == "t0" or (args.arch is None and model_name == "turtle_arch"):
        architecture = "turtle_arch"
    elif args.arch == "t1" or (args.arch is None and model_name == "turtle_t1_arch"):
        architecture = "turtle_t1_arch"
    else:
        raise ValueError(
            f"Cannot infer Turtle architecture from config model={options.get('model')!r}; "
            "pass --arch t0 or --arch t1"
        )
    model = import_module(f"basicsr.models.archs.{architecture}").make_model(options)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["params"], strict=True)
    model = model.to(device).eval()
    del checkpoint

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    previous = None
    k_cache = None
    v_cache = None
    with torch.inference_mode():
        for frame_path in tqdm(files, desc=f"Turtle {args.mode}"):
            current = load_frame(frame_path)
            if previous is None:
                previous = current

            if args.mode == "full":
                restored, k_cache, v_cache = run_full(
                    model, previous, current, device, k_cache, v_cache
                )
            else:
                restored, k_cache, v_cache = run_tiled(
                    model,
                    previous,
                    current,
                    device,
                    args.tile,
                    args.tile_overlap,
                    k_cache,
                    v_cache,
                )

            output_path = args.output_dir / f"{frame_path.stem}.png"
            if args.overwrite or not output_path.exists():
                save_frame_atomic(restored, output_path)
            previous = current

    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"peak_cuda_memory_gib={peak_gib:.3f}")
    print(f"processed_frames={len(files)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
