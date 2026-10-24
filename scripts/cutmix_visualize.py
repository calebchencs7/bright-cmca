#!/usr/bin/env python3
"""Visualize a spatially aligned CutMix example for optical/SAR/label PNGs.

Expected input layout:
  data/target/{optical,sar,y}.png
  data/doner/{optical,sar,y}.png

By default reads ``cutmix-example`` in the repository root and writes the
results to ``outputs/cutmix_visualization``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MODALITIES = ("optical", "sar", "y")


def has_cutmix_data(directory: Path) -> bool:
    return all((directory / f"data/{group}/{name}.png").is_file()
               for group in ("target", "doner") for name in MODALITIES)


def find_input_dir(script_path: Path) -> Path | None:
    """Locate the example data from the launch directory or script location."""
    starts = [Path.cwd().resolve(), script_path.resolve().parent]
    visited: set[Path] = set()
    for start in starts:
        for ancestor in (start, *start.parents):
            if ancestor in visited:
                continue
            visited.add(ancestor)
            for candidate in (ancestor / "cutmix-example", ancestor):
                if has_cutmix_data(candidate):
                    return candidate
    return None


def center_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def rgba(image: Image.Image) -> Image.Image:
    return image.convert("RGBA")


def dashed_rectangle(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                     color=(255, 238, 90, 255), width=2, dash=7) -> None:
    x1, y1, x2, y2 = box
    for x in range(x1, x2, dash * 2):
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
    for y in range(y1, y2, dash * 2):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Data folder containing data/target and data/doner; can be anywhere")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output folder (default: <input-dir>/cutmix_output)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5,
                        help="Target area fraction retained, between 0 and 1")
    parser.add_argument("--min-side", type=float, default=0.35,
                        help="Minimum CutMix rectangle side as a fraction of image side")
    args = parser.parse_args()
    if not 0 <= args.lam <= 1:
        parser.error("--lambda must be between 0 and 1")
    if not 0 < args.min_side <= 1:
        parser.error("--min-side must be in (0, 1]")

    root = args.input_dir or find_input_dir(Path(__file__))
    if root is None:
        parser.error("Could not locate the data automatically. Pass --input-dir with the folder containing data/target and data/doner.")
    root = root.expanduser().resolve()
    output = args.output_dir or (root.parent / "outputs/cutmix_visualization"
                                 if root.name == "cutmix-example" else root / "cutmix_output")
    target_dir, donor_dir = root / "data/target", root / "data/doner"
    for directory in (target_dir, donor_dir):
        for name in MODALITIES:
            path = directory / f"{name}.png"
            if not path.is_file():
                parser.error(f"Missing required input: {path}")

    images = {group: {name: Image.open((target_dir if group == "target" else donor_dir) / f"{name}.png")
                      for name in MODALITIES} for group in ("target", "doner")}
    width = min(im.width for group in images.values() for im in group.values())
    height = min(im.height for group in images.values() for im in group.values())
    images = {group: {name: center_crop(im, width, height)
                      for name, im in group_images.items()}
              for group, group_images in images.items()}

    # Derive a rectangle whose area approximates (1-lambda) of the image.
    side = max(args.min_side, min(1.0, (1.0 - args.lam) ** 0.5))
    box_w, box_h = max(1, round(width * side)), max(1, round(height * side))
    # Place the donor patch slightly left of center and below center.
    center_x, center_y = round(width * 0.40), round(height * 0.68)
    x1 = min(max(0, center_x - box_w // 2), width - box_w)
    y1 = min(max(0, center_y - box_h // 2), height - box_h)
    box = (x1, y1, x1 + box_w, y1 + box_h)
    actual_lambda = 1.0 - box_w * box_h / (width * height)

    output.mkdir(parents=True, exist_ok=True)
    crop_dir, mixed_dir = output / "doner_cut_regions", output / "mixed"
    crop_dir.mkdir(exist_ok=True)
    mixed_dir.mkdir(exist_ok=True)
    tiles: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for name in MODALITIES:
        target, donor = rgba(images["target"][name]), rgba(images["doner"][name])
        marked = donor.copy()
        draw = ImageDraw.Draw(marked)
        if name == "y":
            # Lightly tint the selected label patch so the donor region is visible.
            overlay = Image.new("RGBA", donor.size, (255, 238, 90, 0))
            overlay.putalpha(Image.new("L", donor.size, 0))
            patch_alpha = Image.new("L", donor.size, 0)
            ImageDraw.Draw(patch_alpha).rectangle(box, fill=72)
            overlay.putalpha(patch_alpha)
            marked = Image.alpha_composite(marked, overlay)
            draw = ImageDraw.Draw(marked)
        dashed_rectangle(draw, box)
        mixed = target.copy()
        mixed.paste(donor.crop(box), (x1, y1))
        dashed_rectangle(ImageDraw.Draw(mixed), box)
        marked.save(crop_dir / f"{name}_doner_region.png")
        mixed.save(mixed_dir / f"{name}.png")
        tiles.append((name, target, marked, mixed))

    metadata = {
        "input_alignment": "center-cropped all six inputs to common dimensions",
        "size": {"width": width, "height": height},
        "doner_box_xyxy_exclusive": list(box),
        "requested_target_lambda": args.lam,
        "actual_target_lambda": round(actual_lambda, 6),
        "seed": args.seed,
        "outputs": {"doner_region": "doner_cut_regions/", "mixed": "mixed/"},
    }
    (output / "cutmix_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # One overview image makes target, selected donor patch, and result easy to compare.
    pad, label_h, gap, header_h = 18, 30, 16, 42
    thumb_w = max(320, width)
    thumb_h = round(height * thumb_w / width)
    overview = Image.new("RGB", (pad * 2 + thumb_w * 3 + gap * 2,
                                  header_h + pad + (thumb_h + label_h) * 3 + gap * 2), "#f4f2e9")
    draw = ImageDraw.Draw(overview)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    headings = ("Target", "Doner: clipped area", "CutMix result")
    for col, title in enumerate(headings):
        draw.text((pad + col * (thumb_w + gap), 7), title, fill="#171717", font=font)
    for row, (name, target, marked, mixed) in enumerate(tiles):
        for col, im in enumerate((target, marked, mixed)):
            x = pad + col * (thumb_w + gap)
            y = header_h + pad + row * (height + label_h + gap)
            tile = im.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
            overview.paste(tile, (x, y))
            if col == 0:
                draw.text((x, y + thumb_h + 4), {"optical": "Optical", "sar": "SAR", "y": "Damage label"}[name], fill="#171717", font=font)
    overview.save(output / "cutmix_overview.png")
    print(f"CutMix output written to: {output}")
    print(f"Doner region (x1, y1, x2, y2): {box}; actual lambda={actual_lambda:.4f}")


if __name__ == "__main__":
    main()
