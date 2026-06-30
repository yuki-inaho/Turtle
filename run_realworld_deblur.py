#!/usr/bin/env python
"""Run Turtle's Real-World Deblurring (BSD) model on a folder of video frames.

This is a clean, self-contained entry point (no Docker / absolute paths needed)
for the real-world deblurring task described in the README.

Real-world deblurring uses:
  * config      : options/Turtle_Derain_VRDS.yml   (network architecture)
  * checkpoint  : trained_models/BSD.pth           (download from the GDrive link)
  * model_type  : t0

Example
-------
    uv run python run_realworld_deblur.py \
        --input  datasets/demo_blur \
        --output outputs/demo_deblur \
        --model  trained_models/BSD.pth

`--input` must be a directory containing the ordered frames of ONE video
(e.g. Frame_0001.png, Frame_0002.png, ...). Frames are processed sequentially so
the model's truncated causal history (K/V cache) carries across the sequence.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# --- make the in-tree `basicsr` namespace package importable ---------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))            # for `import basicsr.*`
sys.path.insert(0, str(REPO_ROOT / "basicsr"))  # for `from utils import ...`

from basicsr.utils.options import parse
from basicsr.inference_no_ground_truth import run_inference_patched
from importlib import import_module


def create_video_model(opt, model_type="t0"):
    modules = {
        "t0": "basicsr.models.archs.turtle_arch",
        "t1": "basicsr.models.archs.turtle_t1_arch",
        "SR": "basicsr.models.archs.turtlesuper_t1_arch",
    }
    if model_type not in modules:
        raise ValueError(f"Unknown model_type: {model_type}")
    return import_module(modules[model_type]).make_model(opt)


def load_frames(input_dir):
    files = sorted(
        glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
    )
    if not files:
        raise FileNotFoundError(f"No .png/.jpg frames found in: {input_dir}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Turtle Real-World Deblurring inference")
    ap.add_argument("--input", required=True, help="Folder of ordered video frames")
    ap.add_argument("--output", default="outputs/realworld_deblur", help="Output folder")
    ap.add_argument("--model", default="trained_models/BSD.pth", help="Path to BSD.pth checkpoint")
    ap.add_argument("--config", default="options/Turtle_Derain_VRDS.yml", help="Option/architecture file")
    ap.add_argument("--model-type", default="t0", choices=["t0", "t1", "SR"])
    ap.add_argument("--tile", type=int, default=320, help="Tile size (multiple of 8)")
    ap.add_argument("--tile-overlap", type=int, default=128, help="Tile overlap")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for p in (args.config, args.model):
        if not os.path.exists(p):
            sys.exit(f"[error] file not found: {p}")

    print(f"> device       : {device}")
    print(f"> config       : {args.config}")
    print(f"> checkpoint   : {args.model}")
    print(f"> input frames : {args.input}")
    print(f"> output       : {args.output}")

    opt = parse(args.config, is_train=True)
    model = create_video_model(opt, args.model_type)
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"] if "params" in state else state)
    model = model.to(device).eval()
    print("> model loaded.")

    files = load_frames(args.input)
    print(f"> {len(files)} frames found.")
    os.makedirs(args.output, exist_ok=True)

    to_tensor = transforms.ToTensor()
    previous_frame = None
    k_cache, v_cache = None, None

    st = time.time()
    with torch.no_grad():
        for ix, fpath in enumerate(tqdm(files, desc="deblurring")):
            img = np.array(Image.open(fpath).convert("RGB"))
            current_frame = to_tensor(img).type(torch.FloatTensor)
            c, h, w = current_frame.shape
            if previous_frame is None:
                previous_frame = current_frame

            restored, k_cache, v_cache = run_inference_patched(
                previous_frame.unsqueeze(0),
                current_frame.unsqueeze(0),
                model, device,
                tile=args.tile,
                tile_overlap=args.tile_overlap,
                prev_patch_dict_k=k_cache,
                prev_patch_dict_v=v_cache,
                model_type=args.model_type,
            )
            restored = restored.squeeze(0)[:, :h, :w]

            out = (restored.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            out_path = os.path.join(args.output, f"{Path(fpath).stem}_deblur.png")
            cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            previous_frame = current_frame

    print(f"> done in {time.time() - st:.1f}s. results -> {args.output}")


if __name__ == "__main__":
    main()
