#!/usr/bin/env python3
"""Create CCW90 Raw/Turtle side-by-side comparison frames and a video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--restored-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restored-label", default="Turtle cache=3")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=450)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def add_label(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    labeled = image.convert("RGBA")
    draw = ImageDraw.Draw(labeled, "RGBA")
    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    bar_height = max(52, bbox[3] - bbox[1] + 24)
    draw.rectangle((0, 0, labeled.width, bar_height), fill=(0, 0, 0, 175))
    draw.text(
        (16, 10),
        label,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )
    return labeled.convert("RGB")


def rotate_and_label(path: Path, label: str, font: ImageFont.ImageFont) -> Image.Image:
    with Image.open(path) as image:
        # ROTATE_90 is counter-clockwise in PIL. Labels are added afterwards.
        rotated = image.convert("RGB").transpose(Image.Transpose.ROTATE_90)
    return add_label(rotated, label, font)


def main() -> None:
    args = parse_args()
    input_files = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.limit]
    if len(input_files) != args.limit:
        raise ValueError(f"expected {args.limit} input frames, found {len(input_files)}")

    restored_files = {path.stem: path for path in args.restored_dir.iterdir() if path.is_file()}
    missing = [path.stem for path in input_files if path.stem not in restored_files]
    if missing:
        raise ValueError(f"missing restored frames: {missing[:5]}")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(30)

    for index, input_path in enumerate(input_files, start=1):
        output_path = frames_dir / f"{index:08d}.png"
        if output_path.exists() and not args.overwrite:
            continue
        raw = rotate_and_label(input_path, "Raw", font)
        restored = rotate_and_label(restored_files[input_path.stem], args.restored_label, font)
        if raw.size != restored.size:
            raise ValueError(f"rotated size mismatch for {input_path.name}: {raw.size} vs {restored.size}")
        comparison = Image.new("RGB", (raw.width + restored.width, raw.height))
        comparison.paste(raw, (0, 0))
        comparison.paste(restored, (raw.width, 0))
        comparison.save(output_path, format="PNG")

    video_path = args.output_dir / "raw_vs_turtle_cache3_ccw90.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-start_number",
            "1",
            "-i",
            str(frames_dir / "%08d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
    print(f"comparison_frames={len(input_files)} output_dir={args.output_dir}")
    print(f"video={video_path}")


if __name__ == "__main__":
    main()
