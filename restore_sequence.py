#!/usr/bin/env python
"""Restore a sequence of numbered images with Turtle.

A single entry point for every Turtle restoration task (deblurring, desnowing,
deraining, raindrop removal, super-resolution). Point it at a folder of ordered
frames (`0001.png`, `0002.png`, ... — natural-sorted) and it writes the restored
frames to an output folder. Optionally it also stitches them into an mp4.

Frames are processed **sequentially** so Turtle's truncated causal history
(the K/V cache) propagates across the whole sequence, exactly as in evaluation.

Examples
--------
    # Real-world deblurring (BSD model):
    uv run python restore_sequence.py --task deblur-realworld \
        --input datasets/my_frames --output outputs/deblur

    # Desnowing, and also build a video:
    uv run python restore_sequence.py --task desnow \
        --input datasets/snow_frames --output outputs/desnow --video --fps 24

    # Use an explicit checkpoint / override the tile size:
    uv run python restore_sequence.py --task raindrop \
        --input frames --output out --model trained_models/RainDrop.pth --tile 256

Run `uv run python restore_sequence.py --list` to see all tasks.
"""
import argparse
import glob
import os
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# --- make the in-tree `basicsr` namespace package importable ---------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))               # for `import basicsr.*`
sys.path.insert(0, str(REPO_ROOT / "basicsr"))   # for `from utils import ...`

from basicsr.utils.options import parse  # noqa: E402


def run_inference_patched(img_lq_prev, img_lq_curr, model, device, tile,
                          tile_overlap, prev_patch_dict_k=None,
                          prev_patch_dict_v=None, img_multiple_of=8,
                          model_type="t0"):
    """Tile-based forward pass with K/V cache carried across frames.

    Pads each frame to a multiple of 8, runs the model tile-by-tile, and blends
    overlapping tiles. The per-tile K/V caches are threaded between frames.
    """
    height, width = img_lq_curr.shape[2], img_lq_curr.shape[3]
    H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    img_lq_curr = torch.nn.functional.pad(img_lq_curr, (0, padw, 0, padh), "reflect")
    img_lq_prev = torch.nn.functional.pad(img_lq_prev, (0, padw, 0, padh), "reflect")

    b, c, h, w = img_lq_curr.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be a multiple of 8"
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w).type_as(img_lq_curr)
    Wt = torch.zeros_like(E)

    patch_dict_k, patch_dict_v = {}, {}
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch_curr = img_lq_curr[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            in_patch_prev = img_lq_prev[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            if model_type == "SR":
                in_patch_prev = torch.nn.functional.interpolate(in_patch_prev, scale_factor=1 / 4, mode="bicubic")
                in_patch_curr = torch.nn.functional.interpolate(in_patch_curr, scale_factor=1 / 4, mode="bicubic")

            x = torch.concat((in_patch_prev.unsqueeze(0), in_patch_curr.unsqueeze(0)), dim=1).to(device)
            if prev_patch_dict_k is not None and prev_patch_dict_v is not None:
                old_k = [t.to(device) if t is not None else None for t in prev_patch_dict_k[f"{h_idx}-{w_idx}"]]
                old_v = [t.to(device) if t is not None else None for t in prev_patch_dict_v[f"{h_idx}-{w_idx}"]]
            else:
                old_k = old_v = None

            out_patch, k_c, v_c = model(x.float(), old_k, old_v)
            patch_dict_k[f"{h_idx}-{w_idx}"] = [t.detach().cpu() if t is not None else None for t in k_c]
            patch_dict_v[f"{h_idx}-{w_idx}"] = [t.detach().cpu() if t is not None else None for t in v_c]
            out_patch = out_patch.detach().cpu()
            E[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch)
            Wt[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(torch.ones_like(out_patch))

    restored = torch.clamp(E.div_(Wt), 0, 1)
    return restored, patch_dict_k, patch_dict_v


def run_sr_upscale_patched(img_lq_prev, img_lq_curr, model, device, tile,
                           tile_overlap, prev_patch_dict_k=None,
                           prev_patch_dict_v=None, img_multiple_of=8, scale=4):
    """Tile-based TRUE super-resolution: each input tile is upscaled `scale`x.

    Unlike `run_inference_patched`'s SR path (which downscales the input first so
    the output matches the input size), this feeds tiles at full resolution and
    accumulates them into a `scale`x output canvas. This keeps VRAM bounded while
    producing a genuine 4x upscale.
    """
    height, width = img_lq_curr.shape[2], img_lq_curr.shape[3]
    H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    img_lq_curr = torch.nn.functional.pad(img_lq_curr, (0, padw, 0, padh), "reflect")
    img_lq_prev = torch.nn.functional.pad(img_lq_prev, (0, padw, 0, padh), "reflect")

    b, c, h, w = img_lq_curr.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be a multiple of 8"
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * scale, w * scale).type_as(img_lq_curr)
    Wt = torch.zeros_like(E)

    patch_dict_k, patch_dict_v = {}, {}
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch_curr = img_lq_curr[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            in_patch_prev = img_lq_prev[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            x = torch.concat((in_patch_prev.unsqueeze(0), in_patch_curr.unsqueeze(0)), dim=1).to(device)
            if prev_patch_dict_k is not None and prev_patch_dict_v is not None:
                old_k = [t.to(device) if t is not None else None for t in prev_patch_dict_k[f"{h_idx}-{w_idx}"]]
                old_v = [t.to(device) if t is not None else None for t in prev_patch_dict_v[f"{h_idx}-{w_idx}"]]
            else:
                old_k = old_v = None

            out_patch, k_c, v_c = model(x.float(), old_k, old_v)
            patch_dict_k[f"{h_idx}-{w_idx}"] = [t.detach().cpu() if t is not None else None for t in k_c]
            patch_dict_v[f"{h_idx}-{w_idx}"] = [t.detach().cpu() if t is not None else None for t in v_c]
            out_patch = out_patch.detach().cpu()
            hs, ws = h_idx * scale, w_idx * scale
            ph, pw = out_patch.shape[-2], out_patch.shape[-1]
            E[..., hs:hs + ph, ws:ws + pw].add_(out_patch)
            Wt[..., hs:hs + ph, ws:ws + pw].add_(torch.ones_like(out_patch))

    restored = torch.clamp(E.div_(Wt), 0, 1)
    return restored, patch_dict_k, patch_dict_v

# task -> preset. `model` is the default checkpoint filename in trained_models/.
TASKS = {
    "deblur-realworld": dict(config="options/Turtle_Derain_VRDS.yml",  model="BSD_Deblur.pth",      model_type="t0", tile=320, overlap=128),
    "deblur-gopro":     dict(config="options/Turtle_Deblur_Gopro.yml", model="GoPro_Deblur.pth",    model_type="t1", tile=320, overlap=192),
    "desnow":           dict(config="options/Turtle_Desnow.yml",       model="Desnow.pth",          model_type="t0", tile=320, overlap=128),
    "derain-night":     dict(config="options/Turtle_Derain.yml",       model="NightRain.pth",       model_type="t0", tile=320, overlap=128),
    "raindrop":         dict(config="options/Turtle_Derain_VRDS.yml",  model="RainDrop.pth",        model_type="t1", tile=320, overlap=128),
    "sr":               dict(config="options/Turtle_SR_MVSR.yml",      model="SuperResolution.pth", model_type="SR", tile=256, overlap=64),
}

ARCH_MODULES = {
    "t0": "basicsr.models.archs.turtle_arch",
    "t1": "basicsr.models.archs.turtle_t1_arch",
    "SR": "basicsr.models.archs.turtlesuper_t1_arch",
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def natural_key(path):
    """Sort like a human: frame2 < frame10. Falls back to plain string order."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(input_dir):
    files = [p for p in glob.glob(os.path.join(input_dir, "*"))
             if p.lower().endswith(IMG_EXTS)]
    if not files:
        raise FileNotFoundError(f"No images ({', '.join(IMG_EXTS)}) found in: {input_dir}")
    return sorted(files, key=natural_key)


def build_model(opt, model_type, ckpt_path, device):
    module = import_module(ARCH_MODULES[model_type])
    model = module.make_model(opt)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"] if "params" in state else state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(
        description="Restore a sequence of numbered images with Turtle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--task", choices=list(TASKS), help="Restoration task preset")
    ap.add_argument("--input", help="Folder with the ordered input frames")
    ap.add_argument("--output", default="outputs/restored", help="Output folder")
    ap.add_argument("--model", default=None, help="Checkpoint path (overrides task default)")
    ap.add_argument("--config", default=None, help="Option file (overrides task default)")
    ap.add_argument("--model-type", default=None, choices=list(ARCH_MODULES),
                    help="Turtle variant (overrides task default)")
    ap.add_argument("--models-dir", default="trained_models",
                    help="Folder holding the downloaded checkpoints")
    ap.add_argument("--tile", type=int, default=None, help="Tile size (multiple of 8)")
    ap.add_argument("--tile-overlap", type=int, default=None, help="Overlap between tiles")
    ap.add_argument("--no-patches", action="store_true",
                    help="Process each whole frame at once instead of tiling (needs more VRAM)")
    ap.add_argument("--sr-upscale", action="store_true",
                    help="SR only: feed frames at full resolution (skip the internal 1/4 "
                         "downscale) so the output is a true 4x upscale. Implies --no-patches.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--video", action="store_true", help="Also write an mp4 of the restored frames")
    ap.add_argument("--fps", type=float, default=24.0, help="FPS for --video")
    ap.add_argument("--list", action="store_true", help="List available tasks and exit")
    args = ap.parse_args()

    if args.list:
        print("Available tasks:")
        for name, p in TASKS.items():
            print(f"  {name:18s} model={p['model']:20s} type={p['model_type']} "
                  f"config={p['config']}")
        return
    if not args.task or not args.input:
        ap.error("--task and --input are required (use --list to see tasks)")

    preset = TASKS[args.task]
    config = args.config or preset["config"]
    model_type = args.model_type or preset["model_type"]
    tile = args.tile if args.tile is not None else preset["tile"]
    overlap = args.tile_overlap if args.tile_overlap is not None else preset["overlap"]
    model_path = args.model or os.path.join(args.models_dir, preset["model"])

    sr_upscale = args.sr_upscale and model_type == "SR"
    if args.sr_upscale and model_type != "SR":
        print("[warn] --sr-upscale only applies to SR models; ignoring.")
    no_patches = args.no_patches and not sr_upscale  # SR true-upscale uses its own tiled path

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    for label, p in (("config", config), ("checkpoint", model_path)):
        if not os.path.exists(p):
            sys.exit(f"[error] {label} not found: {p}\n"
                     f"        download checkpoints with:\n"
                     f"        uv run gdown --folder "
                     f"\"https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA\" "
                     f"-O {args.models_dir}")

    print(f"> task        : {args.task}  (model_type={model_type})")
    print(f"> device      : {device}")
    print(f"> config      : {config}")
    print(f"> checkpoint  : {model_path}")
    print(f"> tiling      : {'off' if no_patches else f'tile={tile}, overlap={overlap}'}")
    if sr_upscale:
        print("> sr-upscale  : on (true 4x output)")

    opt = parse(config, is_train=True)
    model = build_model(opt, model_type, model_path, device)
    print("> model loaded.")

    files = list_frames(args.input)
    print(f"> {len(files)} frames found in {args.input}")
    os.makedirs(args.output, exist_ok=True)

    to_tensor = transforms.ToTensor()
    previous_frame = None
    k_cache, v_cache = None, None
    writer = None
    st = time.time()

    with torch.no_grad():
        for fpath in tqdm(files, desc=f"{args.task}"):
            img = np.array(Image.open(fpath).convert("RGB"))
            current = to_tensor(img).type(torch.FloatTensor)
            c, h, w = current.shape
            if previous_frame is None:
                previous_frame = current

            out_h, out_w = h, w
            if sr_upscale:
                # true 4x super-resolution, tiled to keep VRAM bounded
                restored, k_cache, v_cache = run_sr_upscale_patched(
                    previous_frame.unsqueeze(0), current.unsqueeze(0),
                    model, device, tile=tile, tile_overlap=overlap,
                    prev_patch_dict_k=k_cache, prev_patch_dict_v=v_cache,
                )
                restored = restored.squeeze(0)
                out_h, out_w = h * 4, w * 4
            elif no_patches:
                if model_type == "SR":
                    # evaluation protocol: simulate LR by downscaling, model upscales 4x back
                    prev_in = torch.nn.functional.interpolate(previous_frame.unsqueeze(0), scale_factor=1 / 4, mode="bicubic")
                    curr_in = torch.nn.functional.interpolate(current.unsqueeze(0), scale_factor=1 / 4, mode="bicubic")
                else:
                    prev_in, curr_in = previous_frame.unsqueeze(0), current.unsqueeze(0)
                x = torch.concat((prev_in, curr_in), dim=0).unsqueeze(0).to(device)
                restored, k_cache, v_cache = model(x, k_cache, v_cache)
                restored = torch.clamp(restored, 0, 1).squeeze(0)
            else:
                restored, k_cache, v_cache = run_inference_patched(
                    previous_frame.unsqueeze(0), current.unsqueeze(0),
                    model, device, tile=tile, tile_overlap=overlap,
                    prev_patch_dict_k=k_cache, prev_patch_dict_v=v_cache,
                    model_type=model_type,
                )
                restored = restored.squeeze(0)

            restored = restored[:, :out_h, :out_w]
            out = (restored.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(args.output, f"{Path(fpath).stem}_restored.png")
            cv2.imwrite(out_path, out_bgr)

            if args.video:
                if writer is None:
                    vh, vw = out_bgr.shape[:2]
                    vpath = os.path.join(args.output, f"{args.task}.mp4")
                    writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (vw, vh))
                writer.write(out_bgr)

            previous_frame = current

    if writer is not None:
        writer.release()
        print(f"> video       : {os.path.join(args.output, f'{args.task}.mp4')}")
    print(f"> done in {time.time() - st:.1f}s. {len(files)} frames -> {args.output}")


if __name__ == "__main__":
    main()
