#!/usr/bin/env python
"""Build a side-by-side comparison video from two folders of frames.

Left = the frames in `--left`, right = the frames in `--right` (e.g. original vs.
restored). Frames are paired by natural sort order. A generic tool: it takes only
folder/file paths, so nothing about a specific dataset is baked in.

Example
-------
    uv run python make_comparison_video.py \
        --left  input_frames \
        --right restored_frames \
        --output comparison.mp4 --fps 24 --labels "Original,Restored"
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def natural_key(path):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(folder):
    files = [p for p in glob.glob(os.path.join(folder, "*"))
             if p.lower().endswith(IMG_EXTS)]
    if not files:
        sys.exit(f"[error] no images found in: {folder}")
    return sorted(files, key=natural_key)


def label_bar(width, text, height=34, bg=(0, 0, 0), fg=(255, 255, 255)):
    bar = np.full((height, width, 3), bg, np.uint8)
    if text:
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(bar, text, ((width - tw) // 2, (height + th) // 2),
                    font, scale, fg, thick, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(
        description="Side-by-side comparison video from two frame folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--left", required=True, help="Folder of left frames")
    ap.add_argument("--right", required=True, help="Folder of right frames")
    ap.add_argument("--output", default="comparison.mp4", help="Output video file")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--labels", default="Original,Restored",
                    help="Comma-separated labels for left,right (empty to disable)")
    ap.add_argument("--gap", type=int, default=6, help="Pixel gap between the two panes")
    args = ap.parse_args()

    left_files = list_frames(args.left)
    right_files = list_frames(args.right)
    n = min(len(left_files), len(right_files))
    if len(left_files) != len(right_files):
        print(f"[warn] frame count differs (left={len(left_files)}, "
              f"right={len(right_files)}); using first {n}.")

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else ["", ""]
    while len(labels) < 2:
        labels.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = None
    st = __import__("time").time()

    for i in tqdm(range(n), desc="composing"):
        lf = cv2.imread(left_files[i])
        rf = cv2.imread(right_files[i])
        if lf is None or rf is None:
            print(f"[warn] skipping unreadable pair at index {i}")
            continue
        # match right height to left height
        if rf.shape[0] != lf.shape[0]:
            scale = lf.shape[0] / rf.shape[0]
            rf = cv2.resize(rf, (int(round(rf.shape[1] * scale)), lf.shape[0]))

        h = lf.shape[0]
        gap = np.full((h, args.gap, 3), 255, np.uint8) if args.gap > 0 else None
        panes = [lf, rf] if gap is None else [lf, gap, rf]
        frame = np.hstack(panes)

        if any(labels[:2]):
            bar = np.hstack([
                label_bar(lf.shape[1], labels[0]),
                np.full((34, args.gap, 3), 0, np.uint8) if gap is not None else np.empty((34, 0, 3), np.uint8),
                label_bar(rf.shape[1], labels[1]),
            ])
            frame = np.vstack([bar, frame])

        if writer is None:
            vh, vw = frame.shape[:2]
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (vw, vh))
            if not writer.isOpened():
                sys.exit(f"[error] could not open VideoWriter for {args.output}")
        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"> done in {__import__('time').time() - st:.1f}s. "
          f"{n} frames -> {args.output}")


if __name__ == "__main__":
    main()
