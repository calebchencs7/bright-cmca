"""
Train Mask R-CNN + CMCA for BRIGHT.

This script keeps the existing UNet-style scripts untouched and adds a
self-contained instance training pipeline for 4-channel input [RGB + SAR].

Expected dataset layout (same as existing BRIGHT scripts):
  <dataset_path>/pre-event/<tile>_pre_disaster.tif
  <dataset_path>/post-event/<tile>_post_disaster.tif
  <dataset_path>/target/<tile>_building_damage.tif

Damage labels in target raster:
  0: background
  1: intact
  2: damaged
  3: destroyed
  255: ignore

Instance targets are built on-the-fly by connected-components per class.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.mask_rcnn_cmca import build_model_cmca
from util_func.training_curve import TrainingCurveRecorder


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


def random_crop_pair(
    rgb: np.ndarray,
    sar: np.ndarray,
    label: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = label.shape

    out_h = max(crop_size, h)
    out_w = max(crop_size, w)

    pad_rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
    pad_sar = np.zeros((out_h, out_w), dtype=np.float32)
    pad_label = np.ones((out_h, out_w), dtype=label.dtype) * 255

    h_off = random.randint(0, out_h - h)
    w_off = random.randint(0, out_w - w)

    pad_rgb[h_off:h_off + h, w_off:w_off + w] = rgb
    pad_sar[h_off:h_off + h, w_off:w_off + w] = sar
    pad_label[h_off:h_off + h, w_off:w_off + w] = label

    ch = random.randint(0, out_h - crop_size)
    cw = random.randint(0, out_w - crop_size)

    rgb_c = pad_rgb[ch:ch + crop_size, cw:cw + crop_size]
    sar_c = pad_sar[ch:ch + crop_size, cw:cw + crop_size]
    lbl_c = pad_label[ch:ch + crop_size, cw:cw + crop_size]
    return rgb_c, sar_c, lbl_c


def build_instance_target(
    label: np.ndarray,
    image_id: int,
    min_instance_area: int = 16,
) -> Dict[str, torch.Tensor]:
    h, w = label.shape

    boxes = []
    labels = []
    masks = []
    areas = []

    for cls in (1, 2, 3):
        bin_mask = (label == cls).astype(np.uint8)
        if bin_mask.max() == 0:
            continue

        num_cc, cc_map = cv2.connectedComponents(bin_mask)
        for cc_id in range(1, num_cc):
            inst = (cc_map == cc_id)
            area = int(inst.sum())
            if area < min_instance_area:
                continue

            ys, xs = np.where(inst)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 <= x0 or y1 <= y0:
                continue

            boxes.append([x0, y0, x1, y1])
            labels.append(cls)
            masks.append(inst.astype(np.uint8))
            areas.append(float((x1 - x0) * (y1 - y0)))

    if len(boxes) == 0:
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, h, w), dtype=torch.uint8),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
    else:
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

    return target


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

    order = np.argsort(scores)  # low -> high, so high score overwrites
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


def fast_oa_miou(gt: np.ndarray, pred: np.ndarray, num_classes: int = 4) -> Tuple[float, float]:
    valid = gt != 255
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return 0.0, 0.0

    cm = np.bincount(
        num_classes * gt.astype(np.int64) + pred.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    total = float(cm.sum())
    oa = float(np.trace(cm) / total) if total > 0 else 0.0

    iou_list = []
    for c in range(num_classes):
        inter = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - inter
        if union > 0:
            iou_list.append(inter / union)
    miou = float(np.mean(iou_list)) if iou_list else 0.0
    return oa, miou


class BrightMaskRCNNDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        id_list: List[str],
        crop_size: int,
        mode: str = "train",
        suffix: str = ".tif",
        max_iters: int = None,
        min_instance_area: int = 16,
    ):
        self.dataset_path = dataset_path
        self.id_list = list(id_list)
        self.crop_size = int(crop_size)
        self.mode = mode
        self.suffix = suffix
        self.min_instance_area = int(min_instance_area)

        if max_iters is not None:
            rep = int(np.ceil(float(max_iters) / len(self.id_list)))
            self.id_list = (self.id_list * rep)[:max_iters]

    def __len__(self) -> int:
        return len(self.id_list)

    def _paths(self, tile_id: str) -> Tuple[str, str, str]:
        pre = os.path.join(self.dataset_path, "pre-event", f"{tile_id}_pre_disaster{self.suffix}")
        post = os.path.join(self.dataset_path, "post-event", f"{tile_id}_post_disaster{self.suffix}")
        tgt = os.path.join(self.dataset_path, "target", f"{tile_id}_building_damage{self.suffix}")
        return pre, post, tgt

    def __getitem__(self, index: int):
        tile_id = self.id_list[index]
        pre_path, post_path, tgt_path = self._paths(tile_id)

        rgb = imageio.imread(pre_path)
        sar = imageio.imread(post_path)
        label = imageio.imread(tgt_path)

        if label.ndim == 3:
            label = label[:, :, 0]

        rgb = rgb_normalize_01(np.asarray(rgb))
        sar = robust_sar_normalize(np.asarray(sar))
        label = np.asarray(label, dtype=np.uint8)

        if "train" in self.mode:
            rgb, sar, label = random_crop_pair(rgb, sar, label, self.crop_size)

            if random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()
                sar = np.fliplr(sar).copy()
                label = np.fliplr(label).copy()
            if random.random() > 0.5:
                rgb = np.flipud(rgb).copy()
                sar = np.flipud(sar).copy()
                label = np.flipud(label).copy()
            k = random.randint(0, 3)
            if k > 0:
                rgb = np.rot90(rgb, k).copy()
                sar = np.rot90(sar, k).copy()
                label = np.rot90(label, k).copy()

        image_4ch = np.concatenate([rgb, sar[..., None]], axis=-1)
        image_4ch = torch.from_numpy(np.transpose(image_4ch, (2, 0, 1))).float()

        target = build_instance_target(
            label=label,
            image_id=index,
            min_instance_area=self.min_instance_area,
        )

        return image_4ch, target, tile_id, torch.from_numpy(label.astype(np.uint8))


def collate_fn(batch):
    images, targets, names, labels = zip(*batch)
    return list(images), list(targets), list(names), list(labels)


def evaluate_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_thr: float,
    mask_thr: float,
) -> Dict[str, float]:
    model.eval()
    oa_list = []
    miou_list = []

    with torch.no_grad():
        for images, _, _, gt_labels in tqdm(data_loader, desc="Eval", leave=False):
            images = [img.to(device) for img in images]
            preds = model(images)

            for pred, gt in zip(preds, gt_labels):
                gt_np = gt.numpy()
                h, w = gt_np.shape
                dense = predictions_to_dense_map(pred, h, w, score_thr=score_thr, mask_thr=mask_thr)
                oa, miou = fast_oa_miou(gt_np, dense, num_classes=4)
                oa_list.append(oa)
                miou_list.append(miou)

    model.train()
    return {
        "oa": float(np.mean(oa_list)) if oa_list else 0.0,
        "miou": float(np.mean(miou_list)) if miou_list else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN + CMCA on BRIGHT")

    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, default=None)
    parser.add_argument("--test_dataset_path", type=str, default=None)

    parser.add_argument("--train_data_list_path", type=str, required=True)
    parser.add_argument("--val_data_list_path", type=str, default=None)
    parser.add_argument("--test_data_list_path", type=str, default=None)

    parser.add_argument("--suffix", type=str, default=".tif")
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--max_iters", type=int, default=240000)
    parser.add_argument("--min_instance_area", type=int, default=16)

    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument(
        "--lr_policy",
        type=str,
        default="constant",
        choices=["constant"],
        help="Compatibility option. Mask R-CNN trainer currently uses constant LR.",
    )

    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--mask_thr", type=float, default=0.5)

    parser.add_argument("--cmca_num_heads", type=int, default=4)
    parser.add_argument("--cmca_sr_ratio", type=int, default=2)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--model_param_path", type=str, default="./checkpoints")
    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--curve_log_interval", type=int, default=10)
    parser.add_argument("--curve_save_interval", type=int, default=500)

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_ids = load_id_list(args.train_data_list_path)
    val_ids = load_id_list(args.val_data_list_path) if args.val_data_list_path else None
    test_ids = load_id_list(args.test_data_list_path) if args.test_data_list_path else None

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    persistent_workers = bool(args.persistent_workers and args.num_workers and args.num_workers > 0)
    prefetch_factor = max(1, int(args.prefetch_factor))

    train_set = BrightMaskRCNNDataset(
        dataset_path=args.train_dataset_path,
        id_list=train_ids,
        crop_size=args.crop_size,
        mode="train",
        suffix=args.suffix,
        max_iters=args.max_iters,
        min_instance_area=args.min_instance_area,
    )
    train_loader_kwargs = {
        "batch_size": args.train_batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if args.num_workers and args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_set, **train_loader_kwargs)

    eval_num_workers = max(1, args.num_workers // 2)
    eval_persistent_workers = bool(args.persistent_workers and eval_num_workers > 0)

    val_loader = None
    if val_ids is not None and args.val_dataset_path is not None:
        val_set = BrightMaskRCNNDataset(
            dataset_path=args.val_dataset_path,
            id_list=val_ids,
            crop_size=args.crop_size,
            mode="test",
            suffix=args.suffix,
            max_iters=None,
            min_instance_area=args.min_instance_area,
        )
        val_loader_kwargs = {
            "batch_size": args.eval_batch_size,
            "shuffle": False,
            "num_workers": eval_num_workers,
            "drop_last": False,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
        }
        if eval_num_workers > 0:
            val_loader_kwargs["persistent_workers"] = eval_persistent_workers
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader = DataLoader(val_set, **val_loader_kwargs)

    test_loader = None
    if test_ids is not None and args.test_dataset_path is not None:
        test_set = BrightMaskRCNNDataset(
            dataset_path=args.test_dataset_path,
            id_list=test_ids,
            crop_size=args.crop_size,
            mode="test",
            suffix=args.suffix,
            max_iters=None,
            min_instance_area=args.min_instance_area,
        )
        test_loader_kwargs = {
            "batch_size": args.eval_batch_size,
            "shuffle": False,
            "num_workers": eval_num_workers,
            "drop_last": False,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
        }
        if eval_num_workers > 0:
            test_loader_kwargs["persistent_workers"] = eval_persistent_workers
            test_loader_kwargs["prefetch_factor"] = prefetch_factor
        test_loader = DataLoader(test_set, **test_loader_kwargs)

    model = build_model_cmca(
        num_classes=4,
        pretrained=args.pretrained,
        cmca_num_heads=args.cmca_num_heads,
        cmca_sr_ratio=args.cmca_sr_ratio,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    amp_enabled = bool(args.use_amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.model_param_path,
        args.dataset,
        f"MaskRCNN_CMCA_{now_str}",
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    eval_interval = max(1, int(args.eval_interval))
    log_interval = max(1, int(args.curve_log_interval))
    save_interval = max(1, int(args.curve_save_interval))
    curve_recorder = TrainingCurveRecorder(save_dir)

    best_score = -1.0
    best_step = -1
    global_step = 0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

            running_loss = 0.0
            running_steps = 0

            for images, targets, _, _ in pbar:
                global_step += 1
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad(set_to_none=True)

                autocast_kwargs = {
                    "device_type": device.type,
                    "enabled": amp_enabled,
                }
                if amp_enabled:
                    autocast_kwargs["dtype"] = amp_dtype
                with torch.autocast(**autocast_kwargs):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses):
                    print(f"Non-finite loss encountered: {losses.item():.6f}, skip step")
                    continue

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                step_loss = float(losses.item())
                running_loss += step_loss
                running_steps += 1
                pbar.set_postfix(loss=f"{running_loss / max(1, running_steps):.4f}")
                curve_recorder.add_train_loss(global_step, step_loss)

                if global_step % log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    terms = " ".join(
                        f"{name}={value.item():.4f}" for name, value in sorted(loss_dict.items())
                    )
                    print(f"iter {global_step} | total={step_loss:.4f} {terms} lr={lr:.2e}")

                if global_step % eval_interval == 0:
                    val_metrics = None
                    test_metrics = None

                    if val_loader is not None:
                        val_metrics = evaluate_metrics(
                            model=model,
                            data_loader=val_loader,
                            device=device,
                            score_thr=args.score_thr,
                            mask_thr=args.mask_thr,
                        )
                        curve_recorder.add_eval_metrics(
                            global_step, "val", val_metrics["oa"] * 100.0, val_metrics["miou"] * 100.0
                        )
                        print(
                            f"[Iter {global_step}] VAL OA={val_metrics['oa'] * 100.0:.2f}% "
                            f"mIoU={val_metrics['miou'] * 100.0:.2f}%"
                        )

                    if test_loader is not None:
                        test_metrics = evaluate_metrics(
                            model=model,
                            data_loader=test_loader,
                            device=device,
                            score_thr=args.score_thr,
                            mask_thr=args.mask_thr,
                        )
                        curve_recorder.add_eval_metrics(
                            global_step, "test", test_metrics["oa"] * 100.0, test_metrics["miou"] * 100.0
                        )
                        print(
                            f"[Iter {global_step}] TEST OA={test_metrics['oa'] * 100.0:.2f}% "
                            f"mIoU={test_metrics['miou'] * 100.0:.2f}%"
                        )

                    score_for_best = None
                    if val_metrics is not None:
                        score_for_best = val_metrics["miou"]
                    elif test_metrics is not None:
                        score_for_best = test_metrics["miou"]

                    if score_for_best is not None and score_for_best > best_score:
                        best_score = score_for_best
                        best_step = global_step
                        torch.save(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "args": vars(args),
                                "best_score": best_score,
                            },
                            os.path.join(save_dir, "best_model.pth"),
                        )
                        print(f"[Iter {global_step}] new best mIoU={best_score * 100.0:.2f}%")

                    curve_recorder.save()
                elif global_step % save_interval == 0:
                    curve_recorder.save()

            avg_train_loss = running_loss / max(1, running_steps)
            print(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f}")

            if args.save_every_epoch:
                torch.save(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "best_score": best_score,
                    },
                    os.path.join(save_dir, f"epoch_{epoch:03d}.pth"),
                )
    finally:
        torch.save(
            {
                "epoch": args.epochs,
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_score": best_score,
                "best_step": best_step,
            },
            os.path.join(save_dir, "last_model.pth"),
        )
        curve_recorder.save()
        print(f"Training curves saved to {save_dir}")
        if best_step > 0:
            print(f"Best checkpoint at iter {best_step}, mIoU={best_score * 100.0:.2f}%")
        else:
            print("Best checkpoint not updated (no eval metrics available).")
        print(f"Training done. Checkpoints in: {save_dir}")


if __name__ == "__main__":
    main()
