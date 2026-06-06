import argparse
import os
import random

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    0: (0, 0, 0),
    1: (80, 190, 120),
    2: (245, 170, 45),
    3: (220, 60, 60),
    4: (150, 60, 200),
    255: (255, 255, 255),
}


def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.asarray(imageio.imread(path))


def to_uint8_rgb(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    img = img.astype(np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def sar_to_uint8(img, p_low=2.0, p_high=98.0):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    img = img.astype(np.float32)
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros((*img.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(valid, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        gray = np.zeros_like(img, dtype=np.uint8)
    else:
        gray = ((img - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def colorize_label(label):
    label = np.asarray(label)
    if label.ndim == 3:
        label = label[:, :, 0]
    label = label.astype(np.uint8)
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls_id, color in PALETTE.items():
        out[label == cls_id] = color
    return out


def resize_to_height(img, height):
    if img.shape[0] == height:
        return img
    width = max(1, int(round(img.shape[1] * height / img.shape[0])))
    return np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def add_header(panel, title, labels):
    header_h = 44
    out = Image.new("RGB", (panel.shape[1], panel.shape[0] + header_h), (255, 255, 255))
    out.paste(Image.fromarray(panel), (0, header_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((8, 5), title, fill=(0, 0, 0), font=font)
    for text, x in labels:
        draw.text((x + 8, 25), text, fill=(70, 70, 70), font=small_font)
    return np.asarray(out)


def make_panel(pre_img, post_sar, label, title):
    pre = to_uint8_rgb(pre_img)
    sar = sar_to_uint8(post_sar)
    lab = colorize_label(label)

    target_h = max(pre.shape[0], sar.shape[0], lab.shape[0])
    pre = resize_to_height(pre, target_h)
    sar = resize_to_height(sar, target_h)
    lab = resize_to_height(lab, target_h)

    gap = np.ones((target_h, 8, 3), dtype=np.uint8) * 255
    x_pre = 0
    x_sar = pre.shape[1] + gap.shape[1]
    x_lab = x_sar + sar.shape[1] + gap.shape[1]
    panel = np.concatenate([pre, gap, sar, gap, lab], axis=1)
    return add_header(
        panel,
        title,
        [
            ("pre-event optical", x_pre),
            ("post-event SAR", x_sar),
            ("label", x_lab),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Save raw pre-event optical, post-event SAR, and label panels."
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", type=str, default=".tif")
    args = parser.parse_args()

    with open(args.data_list_path) as f:
        data_list = [line.strip() for line in f if line.strip()]

    if args.random:
        random.seed(args.seed)
        data_list = random.sample(data_list, k=min(args.num_samples, len(data_list)))
    else:
        data_list = data_list[:args.num_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    for i, name in enumerate(data_list):
        pre_path = os.path.join(
            args.dataset_path, "pre-event", name + "_pre_disaster" + args.suffix
        )
        post_path = os.path.join(
            args.dataset_path, "post-event", name + "_post_disaster" + args.suffix
        )
        label_path = os.path.join(
            args.dataset_path, "target", name + "_building_damage" + args.suffix
        )

        panel = make_panel(
            read_image(pre_path),
            read_image(post_path),
            read_image(label_path),
            title=name,
        )
        safe_name = name.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(args.output_dir, f"{i:02d}_{safe_name}.png")
        Image.fromarray(panel).save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
