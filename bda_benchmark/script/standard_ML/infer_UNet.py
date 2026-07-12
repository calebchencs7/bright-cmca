"""
Inference script for building damage assessment.

Supports:
    - Vanilla backbone (UNet, DeepLabV3+, etc.) — loads flat state_dict
    - Per-disaster-event and per-disaster-type metrics
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset.make_data_loader import (
    MultimodalDamageAssessmentDatset,
)
from model.UNet import UNet
from util_func.metrics import Evaluator


# ---------------------------------------------------------------------------
# Build backbone (shared with train_UNet.py)
# ---------------------------------------------------------------------------

def build_backbone(model_type, in_channels, num_classes):
    mt = model_type.lower()
    if mt == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)
    if mt == "unetwithfeatures":
        # DPCL-friendly subclass of UNet. State dict is identical to UNet,
        # so checkpoints are cross-compatible. At inference we just call
        # backbone(x) (return_features=False is the default) and the
        # forward path is byte-identical to vanilla UNet.
        from model.UNetDPCL import UNetWithFeatures
        return UNetWithFeatures(in_channels=in_channels, num_classes=num_classes)
    if mt == "unetcmca":
        from model.UNetCMCA import UNetCMCA
        return UNetCMCA(in_channels=in_channels, num_classes=num_classes)
    if mt in ("siamattnunetcmca", "siamattncmca"):
        from model.SiamAttnUNetCMCA import SiamAttnUNetCMCA
        return SiamAttnUNetCMCA(in_channels=3, num_classes=num_classes)
    if mt == "damageformercmca":
        from model.DamageFormerCMCA import DamageFormerCMCA
        return DamageFormerCMCA(num_classes=num_classes)
    if mt == "damageformer":
        from model.DamageFormer import DamageFormer
        return DamageFormer(num_classes=num_classes)
    if mt in ("deeplabv3plus", "deeplabv3+"):
        from model.DeepLabV3Plus import DeepLabV3Plus
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes)
    if mt in ("deeplabv3pluscmca", "deeplabv3+cmca"):
        from model.DeepLabV3PlusCMCA import DeepLabV3PlusCMCA
        return DeepLabV3PlusCMCA(in_channels=in_channels, num_classes=num_classes)
    if mt == "siamattnunet":
        from model.SiamAttnUNet import SiamAttnUNet
        return SiamAttnUNet(in_channels=3, num_classes=num_classes)
    if mt == "siamcrnncmca":
        from model.SiamCRNNCMCA import SiamCRNNCMCA
        return SiamCRNNCMCA(num_classes=num_classes)
    if mt == "siamcrnn":
        from model.SiamCRNN import SiamCRNN
        return SiamCRNN(num_classes=num_classes)
    raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class Inference:
    DISASTER_EVENTS = [
        "turkey-earthquake", "hawaii-wildfire", "morocco-earthquake",
        "haiti-earthquake", "la_palma-volcano", "congo-volcano",
        "beirut-explosion", "bata-explosion", "libya-flood",
        "noto-earthquake", "marshall-wildfire", "ukraine-conflict",
        "myanmar-hurricane", "mexico-hurricane",
    ]
    DISASTER_TYPES = [
        "earthquake", "wildfire", "volcano", "explosion",
        "flood", "conflict", "hurricane",
    ]
    COLOR_MAP = {
        0: (255, 255, 255),   # Background - white
        1: (70, 181, 121),    # Intact - green
        2: (228, 189, 139),   # Damaged - yellow
        3: (182, 70, 69),     # Destroyed - red
    }

    def __init__(self, args):
        self.device = self._resolve_device(args.device)
        print(f"Using device: {self.device}")

        self.output_dir = args.output_dir

        # ---- Dataset ----
        dataset = MultimodalDamageAssessmentDatset(
            args.test_dataset_path, args.test_data_list,
            1024, None, "test", suffix=".tif",
        )
        self.test_loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False)

        # ---- Build backbone ----
        self.backbone = build_backbone(args.model_type, in_channels=6, num_classes=4)

        # ---- Load checkpoint ----
        ckpt = torch.load(args.model_path, map_location=self.device)

        if "backbone" in ckpt:
            # New-style nested checkpoint
            self.backbone.load_state_dict(ckpt["backbone"], strict=True)
            print(f"Loaded nested checkpoint (step={ckpt.get('step', '?')})")
        else:
            # Legacy flat state_dict
            self.backbone.load_state_dict(ckpt, strict=True)
            print("Loaded legacy flat checkpoint")

        self.backbone = self.backbone.to(self.device).eval()

        # ---- Evaluators ----
        self.evaluator = Evaluator(num_class=4)
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.single_evaluator = Evaluator(num_class=4)
        self.disaster_type_evals = {t: Evaluator(num_class=4) for t in self.DISASTER_TYPES}
        self.disaster_event_evals = {e: Evaluator(num_class=4) for e in self.DISASTER_EVENTS}

        # Output dirs
        os.makedirs(os.path.join(self.output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "colored"), exist_ok=True)

    @staticmethod
    def _resolve_device(device_arg):
        if device_arg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_arg)

    def _forward(self, input_data):
        return self.backbone(input_data)

    def run_inference(self):
        print("Starting inference...")
        self.evaluator.reset()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                self.single_evaluator.reset()

                pre, post, labels_loc, labels_clf, file_name = data

                pre = pre.to(self.device)
                post = post.to(self.device)
                file_name = file_name[0]

                input_data = torch.cat([pre, post], dim=1)
                logits = self._forward(input_data)

                output = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                output = output.squeeze(0)

                self.save_colored_map(output, file_name)
                self.save_original_map(output, file_name)

                labels_clf_np = labels_clf.squeeze().cpu().numpy()
                labels_loc_np = labels_loc.squeeze().cpu().numpy()

                # Localization F1 (binary: building vs background)
                output_loc = output.copy()
                output_loc[output_loc > 0] = 1
                self.evaluator_loc.add_batch(labels_loc_np, output_loc)

                # Per-building damage evaluation
                damage_mask = labels_loc_np > 0
                if damage_mask.any():
                    self.evaluator_clf.add_batch(
                        labels_clf_np[damage_mask], output[damage_mask]
                    )

                self.evaluator.add_batch(labels_clf_np, output)
                self.single_evaluator.add_batch(labels_clf_np, output)

                for dtype in self.DISASTER_TYPES:
                    if dtype in file_name:
                        self.disaster_type_evals[dtype].add_batch(labels_clf_np, output)
                        break

                for event in self.DISASTER_EVENTS:
                    if event in file_name:
                        self.disaster_event_evals[event].add_batch(labels_clf_np, output)
                        break

        self._print_overall_metrics()
        self._print_event_metrics()
        self._print_type_metrics()

    def save_original_map(self, prediction, file_name):
        path = os.path.join(self.output_dir, "original", file_name + "_building_damage.png")
        Image.fromarray(prediction).save(path)

    def save_colored_map(self, prediction, file_name):
        color_img = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for cls, color in self.COLOR_MAP.items():
            color_img[prediction == cls] = color
        path = os.path.join(self.output_dir, "colored", file_name + "_building_damage.png")
        Image.fromarray(color_img).save(path)

    def _print_overall_metrics(self):
        OA = self.evaluator.Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        loc_f1 = self.evaluator_loc.Pixel_F1_score() * 100
        damage_f1 = self.evaluator_clf.Damage_F1_score()
        hmean_f1 = _safe_hmean(damage_f1) * 100

        print("\n=== Overall Metrics ===")
        print(f"Pixel Accuracy: {OA*100:.2f}%")
        print(f"Mean IoU: {mIoU*100:.2f}%")
        print(f"IoU per class: {IoU*100}")
        print(f"F1 Score (loc): {loc_f1:.2f}%")
        print(f"F1 Score (clf): {hmean_f1:.2f}%")

    def _print_event_metrics(self):
        print("\n=== Per-Event Metrics ===")
        total = 0.0
        for event, ev in self.disaster_event_evals.items():
            miou = ev.Mean_Intersection_over_Union()
            iou = ev.Intersection_over_Union()
            total += miou
            print(f"{event}: mIoU={miou*100:.2f}%, IoU={iou*100}")
        print(f"Average mIoU={total/len(self.DISASTER_EVENTS)*100:.2f}%")

    def _print_type_metrics(self):
        print("\n=== Per-Disaster-Type Metrics ===")
        total = 0.0
        for dtype, ev in self.disaster_type_evals.items():
            miou = ev.Mean_Intersection_over_Union()
            iou = ev.Intersection_over_Union()
            total += miou
            print(f"{dtype}: mIoU={miou*100:.2f}%, IoU={iou*100}")
        print(f"Average mIoU={total/len(self.DISASTER_TYPES)*100:.2f}%")


def _safe_hmean(scores, eps=1e-6):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    scores = np.where(scores <= 0, eps, scores)
    return len(scores) / np.sum(1.0 / scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on BRIGHT")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="UNet")
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--test_data_list_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cuda:0, cuda:1, mps, or cpu",
    )

    args = parser.parse_args()

    with open(args.test_data_list_path) as f:
        args.test_data_list = [line.strip() for line in f]

    inference = Inference(args)
    inference.run_inference()
