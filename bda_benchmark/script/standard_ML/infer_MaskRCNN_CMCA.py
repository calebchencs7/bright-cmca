"""
Inference for Mask R-CNN + CMCA on BRIGHT.

Outputs:
  - Pixel maps: original label PNG + colored PNG (compatible with current scripts)
  - Optional COCO-style prediction JSON for challenge submission workflow
"""

import argparse
import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.mask_rcnn_cmca import build_model_cmca

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def load_id_list(path_or_csv: str) -> List[str]:
    if os.path.isfile(path_or_csv):
        with open(path_or_csv, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = [x.strip() for x in path_or_csv.split(",") if x.strip()]
    if not ids:
        raise ValueError(f"No ids found from: {path_or_csv}")
    return ids


def robust_sar_normalize(sar: np.ndarray) -> np.ndarray:
    sar = sar.astype(np.float32)
    if sar.ndim == 3:
        sar = sar[:, :, 0]

    if np.max(sar) <= 1.0:
        return np.clip(sar, 0.0, 1.0)

    p2, p98 = np.percentile(sar, [2.0, 98.0])
    if p98 - p2 < 1e-6:
        out = sar / 255.0
    else:
        out = (np.clip(sar, p2, p98) - p2) / (p98 - p2)
    return np.clip(out, 0.0, 1.0)


def rgb_normalize_01(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[-1] > 3:
        rgb = rgb[:, :, :3]

    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


class BrightMaskRCNNInferDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        id_list: List[str],
        suffix: str = ".tif",
        with_label: bool = False,
    ):
        self.dataset_path = dataset_path
        self.id_list = list(id_list)
        self.suffix = suffix
        self.with_label = with_label

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index: int):
        tile_id = self.id_list[index]

        pre_path = os.path.join(self.dataset_path, "pre-event", f"{tile_id}_pre_disaster{self.suffix}")
        post_path = os.path.join(self.dataset_path, "post-event", f"{tile_id}_post_disaster{self.suffix}")

        rgb = rgb_normalize_01(np.asarray(imageio.imread(pre_path)))
        sar = robust_sar_normalize(np.asarray(imageio.imread(post_path)))

        image_4ch = np.concatenate([rgb, sar[..., None]], axis=-1)
        image_4ch = torch.from_numpy(np.transpose(image_4ch, (2, 0, 1))).float()

        if self.with_label:
            tgt_path = os.path.join(self.dataset_path, "target", f"{tile_id}_building_damage{self.suffix}")
            label = np.asarray(imageio.imread(tgt_path))
            if label.ndim == 3:
                label = label[:, :, 0]
            label = torch.from_numpy(label.astype(np.uint8))
        else:
            label = torch.zeros((image_4ch.shape[1], image_4ch.shape[2]), dtype=torch.uint8)

        return image_4ch, tile_id, label


def collate_fn(batch):
    images, names, labels = zip(*batch)
    return list(images), list(names), list(labels)


def predictions_to_dense_map(
    pred: Dict[str, torch.Tensor],
    h: int,
    w: int,
    score_thr: float = 0.3,
    mask_thr: float = 0.5,
) -> np.ndarray:
    dense = np.zeros((h, w), dtype=np.uint8)
    if pred is None or len(pred.get("scores", [])) == 0:
        return dense

    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()  # [N,1,H,W]

    order = np.argsort(scores)
    for idx in order:
        score = float(scores[idx])
        if score < score_thr:
            continue
        cls = int(labels[idx])
        if cls < 1 or cls > 3:
            continue

        m = masks[idx, 0] >= mask_thr
        dense[m] = cls

    return dense


def fast_miou(gt: np.ndarray, pred: np.ndarray, num_classes: int = 4) -> float:
    valid = gt != 255
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return 0.0

    cm = np.bincount(
        num_classes * gt.astype(np.int64) + pred.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    iou_list = []
    for c in range(num_classes):
        inter = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - inter
        if union > 0:
            iou_list.append(inter / union)
    return float(np.mean(iou_list)) if iou_list else 0.0


def load_checkpoint(model: torch.nn.Module, model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            return ckpt
        if "backbone" in ckpt:
            model.load_state_dict(ckpt["backbone"], strict=True)
            return ckpt
    model.load_state_dict(ckpt, strict=True)
    return ckpt


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Mask R-CNN + CMCA on BRIGHT")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--test_data_list_path", type=str, required=True)
    parser.add_argument("--suffix", type=str, default=".tif")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--mask_thr", type=float, default=0.5)

    parser.add_argument("--cmca_num_heads", type=int, default=4)
    parser.add_argument("--cmca_sr_ratio", type=int, default=2)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--with_label", action="store_true",
                        help="If set, read target maps and report quick mIoU.")

    parser.add_argument("--save_coco_json", type=str, default=None,
                        help="Path to save COCO-style prediction JSON.")
    parser.add_argument("--image_id_map_json", type=str, default=None,
                        help="Optional JSON mapping: {tile_id: coco_image_id}.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    ids = load_id_list(args.test_data_list_path)
    ds = BrightMaskRCNNInferDataset(
        dataset_path=args.test_dataset_path,
        id_list=ids,
        suffix=args.suffix,
        with_label=args.with_label,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_cmca(
        num_classes=4,
        pretrained=False,
        cmca_num_heads=args.cmca_num_heads,
        cmca_sr_ratio=args.cmca_sr_ratio,
    ).to(device)
    _ = load_checkpoint(model, args.model_path, device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    orig_dir = os.path.join(args.output_dir, "original")
    color_dir = os.path.join(args.output_dir, "colored")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    color_map = {
        0: (255, 255, 255),
        1: (70, 181, 121),
        2: (228, 189, 139),
        3: (182, 70, 69),
    }

    image_id_map = None
    if args.image_id_map_json:
        with open(args.image_id_map_json, "r", encoding="utf-8") as f:
            image_id_map = json.load(f)

    coco_results = []
    miou_list = []

    with torch.no_grad():
        for images, names, labels in tqdm(loader, desc="Infer"):
            image = images[0].to(device)
            name = names[0]
            gt_label = labels[0].numpy()

            pred = model([image])[0]

            h, w = image.shape[-2:]
            dense = predictions_to_dense_map(
                pred=pred,
                h=h,
                w=w,
                score_thr=args.score_thr,
                mask_thr=args.mask_thr,
            )

            Image.fromarray(dense).save(os.path.join(orig_dir, f"{name}_building_damage.png"))

            color_img = np.zeros((h, w, 3), dtype=np.uint8)
            for cls, color in color_map.items():
                color_img[dense == cls] = color
            Image.fromarray(color_img).save(os.path.join(color_dir, f"{name}_building_damage.png"))

            if args.with_label:
                miou_list.append(fast_miou(gt_label, dense, num_classes=4))

            if args.save_coco_json is not None:
                scores = pred["scores"].detach().cpu().numpy()
                labels_np = pred["labels"].detach().cpu().numpy()
                boxes = pred["boxes"].detach().cpu().numpy()
                masks = pred["masks"].detach().cpu().numpy()[:, 0]

                coco_img_id = image_id_map.get(name, None) if image_id_map else None
                if coco_img_id is None:
                    coco_img_id = name

                for i in range(len(scores)):
                    score = float(scores[i])
                    cls = int(labels_np[i])
                    if score < args.score_thr or cls < 1 or cls > 3:
                        continue

                    x1, y1, x2, y2 = boxes[i].tolist()
                    bbox = [
                        float(x1),
                        float(y1),
                        float(max(0.0, x2 - x1)),
                        float(max(0.0, y2 - y1)),
                    ]

                    item = {
                        "image_id": coco_img_id,
                        "category_id": cls,
                        "score": score,
                        "bbox": bbox,
                    }

                    if mask_utils is not None:
                        bin_mask = (masks[i] >= args.mask_thr).astype(np.uint8)
                        rle = mask_utils.encode(np.asfortranarray(bin_mask))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        item["segmentation"] = rle

                    coco_results.append(item)

    if args.with_label and miou_list:
        print(f"Quick pixel mIoU: {np.mean(miou_list) * 100:.2f}%")

    if args.save_coco_json is not None:
        with open(args.save_coco_json, "w", encoding="utf-8") as f:
            json.dump(coco_results, f)
        if mask_utils is None:
            print("Saved COCO JSON without segmentation RLE (pycocotools not installed).")
        print(f"COCO predictions saved to: {args.save_coco_json}")

    print(f"Inference done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
