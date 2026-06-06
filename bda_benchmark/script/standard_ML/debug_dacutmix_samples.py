import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.make_data_loader import MultimodalDamageAssessmentDatset


MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
PALETTE = {
    0: (0, 0, 0),
    1: (80, 190, 120),
    2: (245, 170, 45),
    3: (220, 60, 60),
    255: (255, 255, 255),
}


def denormalize_chw(x):
    x = np.asarray(x, dtype=np.float32).transpose(1, 2, 0)
    x = x * STD + MEAN
    return np.clip(x, 0, 255).astype(np.uint8)


def colorize_label(label):
    label = np.asarray(label).astype(np.uint8)
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls_id, color in PALETTE.items():
        out[label == cls_id] = color
    return out


def draw_box(img, box, color=(255, 255, 0), width=3):
    if box is None:
        return img
    y0, y1, x0, x1 = box
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    for i in range(width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)
    return np.asarray(out)


def make_panel(pre_img, post_img, label, box=None):
    pre = denormalize_chw(pre_img)
    post = denormalize_chw(post_img)
    lab = colorize_label(label)
    pre = draw_box(pre, box)
    post = draw_box(post, box)
    lab = draw_box(lab, box)
    gap = np.ones((pre.shape[0], 8, 3), dtype=np.uint8) * 255
    return np.concatenate([pre, gap, post, gap, lab], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Save visual DACutMix samples.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--damage_class_ids", type=str, default="2,3")
    parser.add_argument("--dacutmix_min_damage_pixels", type=int, default=200)
    parser.add_argument("--dacutmix_min_damage_ratio", type=float, default=0.05)
    parser.add_argument("--dacutmix_patch_min_ratio", type=float, default=0.12)
    parser.add_argument("--dacutmix_patch_max_ratio", type=float, default=0.35)
    parser.add_argument("--dacutmix_box_tries", type=int, default=10)
    parser.add_argument("--dacutmix_donor_tries", type=int, default=10)
    parser.add_argument("--suffix", type=str, default=".tif")
    args = parser.parse_args()

    with open(args.data_list_path) as f:
        data_list = [line.strip() for line in f if line.strip()]

    damage_class_ids = tuple(int(x.strip()) for x in args.damage_class_ids.split(",") if x.strip())
    dataset = MultimodalDamageAssessmentDatset(
        dataset_path=args.dataset_path,
        data_list=data_list,
        crop_size=args.crop_size,
        type="train",
        suffix=args.suffix,
        use_dacutmix=True,
        dacutmix_prob=1.0,
        damage_class_ids=damage_class_ids,
        dacutmix_min_damage_pixels=args.dacutmix_min_damage_pixels,
        dacutmix_min_damage_ratio=args.dacutmix_min_damage_ratio,
        dacutmix_patch_min_ratio=args.dacutmix_patch_min_ratio,
        dacutmix_patch_max_ratio=args.dacutmix_patch_max_ratio,
        dacutmix_box_tries=args.dacutmix_box_tries,
        dacutmix_donor_tries=args.dacutmix_donor_tries,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    n = min(args.num_samples, len(dataset))
    for i in range(n):
        pre_img, post_img, _, label, data_idx = dataset[i]
        info = getattr(dataset, "last_dacutmix_info", None) or {}
        panel = make_panel(pre_img, post_img, label, box=info.get("box"))
        safe_name = str(data_idx).replace("/", "_").replace("\\", "_")
        out_path = os.path.join(args.output_dir, f"{i:02d}_{safe_name}.png")
        Image.fromarray(panel).save(out_path)
        if info:
            print(
                f"{out_path} | {info['base_event']} <- {info['donor_event']} "
                f"box={info['box']}"
            )
        else:
            print(f"{out_path} | no DACutMix patch found")


if __name__ == "__main__":
    main()
