"""
Inference script for building damage assessment.

Supports:
    - Vanilla backbone (UNet, DeepLabV3+, etc.) — loads flat state_dict
    - Per-disaster-event and per-disaster-type metrics
"""

import os
import sys
import argparse
import csv
import json
import time

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
    if mt == "unetca":
        from model.UNetCA import UNetCA
        return UNetCA(in_channels=in_channels, num_classes=num_classes)
    if mt in ("siamattnunetcmca", "siamattncmca"):
        from model.SiamAttnUNetCMCA import SiamAttnUNetCMCA
        return SiamAttnUNetCMCA(in_channels=3, num_classes=num_classes)
    if mt in ("siamattnunetca", "siamattnca"):
        from model.SiamAttnUNetCA import SiamAttnUNetCA
        return SiamAttnUNetCA(in_channels=3, num_classes=num_classes)
    if mt == "damageformercmcadeep":
        from model.DamageFormerCMCADeep import DamageFormerCMCADeep
        return DamageFormerCMCADeep(num_classes=num_classes)
    if mt == "damageformercmca":
        from model.DamageFormerCMCA import DamageFormerCMCA
        return DamageFormerCMCA(num_classes=num_classes)
    if mt == "damageformerca":
        from model.DamageFormerCA import DamageFormerCA
        return DamageFormerCA(num_classes=num_classes)
    if mt == "damageformer":
        from model.DamageFormer import DamageFormer
        return DamageFormer(num_classes=num_classes)
    if mt in ("changemamba", "mambabda", "mambabda_tiny", "mambabdatiny"):
        from model.ChangeMamba import ChangeMamba
        return ChangeMamba(in_channels=in_channels, num_classes=num_classes)
    if mt in ("changemambacmca", "mambabdacmca", "mambabda_tiny_cmca"):
        from model.ChangeMamba import ChangeMambaCMCA
        return ChangeMambaCMCA(in_channels=in_channels, num_classes=num_classes)
    if mt in ("changemambaca", "mambabdaca", "mambabda_tiny_ca"):
        from model.ChangeMamba import ChangeMambaCA
        return ChangeMambaCA(in_channels=in_channels, num_classes=num_classes)
    if mt in ("deeplabv3plus", "deeplabv3+"):
        from model.DeepLabV3Plus import DeepLabV3Plus
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes)
    if mt in ("deeplabv3pluscmca", "deeplabv3+cmca"):
        from model.DeepLabV3PlusCMCA import DeepLabV3PlusCMCA
        return DeepLabV3PlusCMCA(in_channels=in_channels, num_classes=num_classes)
    if mt in ("deeplabv3plusca", "deeplabv3+ca"):
        from model.DeepLabV3PlusCA import DeepLabV3PlusCA
        return DeepLabV3PlusCA(in_channels=in_channels, num_classes=num_classes)
    if mt == "siamattnunet":
        from model.SiamAttnUNet import SiamAttnUNet
        return SiamAttnUNet(in_channels=3, num_classes=num_classes)
    if mt == "siamcrnncmca":
        from model.SiamCRNNCMCA import SiamCRNNCMCA
        return SiamCRNNCMCA(num_classes=num_classes)
    if mt == "siamcrnnca":
        from model.SiamCRNNCA import SiamCRNNCA
        return SiamCRNNCA(num_classes=num_classes)
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
        self.model_path = args.model_path
        self.model_type = args.model_type
        self.timing_warmup = max(0, int(args.timing_warmup))
        self.compute_flops = bool(args.compute_flops)
        self.forward_latencies_ms = []
        self.input_shapes = set()

        # ---- Dataset ----
        dataset = MultimodalDamageAssessmentDatset(
            args.test_dataset_path, args.test_data_list,
            1024, None, "test", suffix=".tif",
            modality_mode=args.modality_mode,
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
        self._print_model_size()

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
        if getattr(self.backbone, "supports_loc_head", False):
            return self.backbone(input_data, return_loc=True)
        return None, self.backbone(input_data)

    def _synchronize_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _model_size_stats(self):
        parameters = list(self.backbone.parameters())
        buffers = list(self.backbone.buffers())
        total_params = sum(p.numel() for p in parameters)
        trainable_params = sum(p.numel() for p in parameters if p.requires_grad)
        state_bytes = sum(x.numel() * x.element_size() for x in parameters + buffers)
        return total_params, trainable_params, state_bytes / (1024 ** 2)

    def _print_model_size(self):
        total, trainable, state_mb = self._model_size_stats()
        print("\n=== Model Complexity ===")
        print(f"Total parameters: {total:,} ({total / 1e6:.3f} M)")
        print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.3f} M)")
        print(f"Model state size: {state_mb:.2f} MB")

    def _device_name(self):
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        if self.device.type == "mps":
            return "Apple Metal Performance Shaders"
        return "CPU"

    def _compute_flop_stats(self):
        stats = {
            "flops": None,
            "gflops": None,
            "flops_input_shape": None,
            "flops_tool": "fvcore",
            "flops_convention": "one fused multiply-add counts as one FLOP",
            "unsupported_flop_ops": {},
            "flops_error": None,
        }
        if not self.compute_flops:
            stats["flops_error"] = "disabled"
            return stats

        input_shapes = sorted(self.input_shapes)
        if len(input_shapes) != 1:
            stats["flops_error"] = (
                "FLOPs require exactly one input shape; "
                f"observed {len(input_shapes)} shapes"
            )
            return stats

        shape = input_shapes[0]
        stats["flops_input_shape"] = list(shape)

        try:
            from fvcore.nn import FlopCountAnalysis

            dtype = next(self.backbone.parameters()).dtype
            example = torch.zeros(shape, device=self.device, dtype=dtype)
            analysis = FlopCountAnalysis(self.backbone, (example,))

            if "changemamba" in self.model_type.lower() or "mambabda" in self.model_type.lower():
                from MambaCD.classification.models.vmamba import selective_scan_flop_jit

                analysis = analysis.set_op_handle(
                    **{
                        "aten::silu": None,
                        "aten::neg": None,
                        "aten::exp": None,
                        "aten::flip": None,
                        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
                        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
                        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
                        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
                        "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
                        "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
                    }
                )

            with torch.no_grad():
                total_flops = float(analysis.total())
            unsupported = dict(analysis.unsupported_ops())

            stats["flops"] = total_flops
            stats["gflops"] = total_flops / 1e9
            stats["unsupported_flop_ops"] = unsupported
            del analysis, example
        except Exception as exc:
            stats["flops_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._synchronize_device()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return stats

    def _save_efficiency_metrics(self):
        total, trainable, state_mb = self._model_size_stats()
        warmup = min(self.timing_warmup, max(0, len(self.forward_latencies_ms) - 1))
        measured = self.forward_latencies_ms[warmup:]

        if measured:
            latency_mean = float(np.mean(measured))
            latency_median = float(np.median(measured))
            latency_p95 = float(np.percentile(measured, 95))
            fps = 1000.0 / latency_mean
        else:
            latency_mean = latency_median = latency_p95 = fps = None

        peak_allocated_mb = None
        peak_reserved_mb = None
        if self.device.type == "cuda":
            peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

        flop_stats = self._compute_flop_stats()
        checkpoint_mb = os.path.getsize(self.model_path) / (1024 ** 2)
        input_shapes = [list(shape) for shape in sorted(self.input_shapes)]
        metrics = {
            "model_type": self.model_type,
            "device": str(self.device),
            "device_name": self._device_name(),
            "precision": "fp32",
            "batch_size": 1,
            "input_shapes": input_shapes,
            "total_parameters": total,
            "total_parameters_m": total / 1e6,
            "trainable_parameters": trainable,
            "trainable_parameters_m": trainable / 1e6,
            "model_state_mb": state_mb,
            "checkpoint_mb": checkpoint_mb,
            **flop_stats,
            "timing_warmup_samples": warmup,
            "timed_samples": len(measured),
            "latency_mean_ms_per_image": latency_mean,
            "latency_median_ms_per_image": latency_median,
            "latency_p95_ms_per_image": latency_p95,
            "throughput_fps": fps,
            "peak_cuda_allocated_mb": peak_allocated_mb,
            "peak_cuda_reserved_mb": peak_reserved_mb,
            "timing_scope": "model_forward_only",
        }

        json_path = os.path.join(self.output_dir, "efficiency_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=True)

        csv_path = os.path.join(self.output_dir, "efficiency_metrics.csv")
        csv_row = dict(metrics)
        csv_row["input_shapes"] = ";".join("x".join(map(str, s)) for s in input_shapes)
        flop_shape = csv_row.get("flops_input_shape")
        csv_row["flops_input_shape"] = (
            "x".join(map(str, flop_shape)) if flop_shape else ""
        )
        csv_row["unsupported_flop_ops"] = json.dumps(
            csv_row.get("unsupported_flop_ops", {}), sort_keys=True
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
            writer.writeheader()
            writer.writerow(csv_row)

        print("\n=== Efficiency Metrics ===")
        print(f"Benchmark device: {metrics['device_name']} ({self.device})")
        print(f"Input shape(s): {csv_row['input_shapes']}")
        print(f"Checkpoint size: {checkpoint_mb:.2f} MB")
        if flop_stats["gflops"] is not None:
            print(
                f"Computational cost: {flop_stats['gflops']:.3f} GFLOPs "
                f"(input {csv_row['flops_input_shape']}, fvcore)"
            )
            if flop_stats["unsupported_flop_ops"]:
                print(
                    "WARNING: unsupported FLOP operators: "
                    f"{flop_stats['unsupported_flop_ops']}"
                )
            else:
                print("Unsupported FLOP operators: none")
        elif flop_stats["flops_error"] != "disabled":
            print(f"FLOPs unavailable: {flop_stats['flops_error']}")
        if measured:
            print(
                f"Forward latency: mean={latency_mean:.3f} ms/image, "
                f"median={latency_median:.3f} ms/image, p95={latency_p95:.3f} ms/image"
            )
            print(f"Throughput: {fps:.2f} FPS ({len(measured)} timed samples)")
        if peak_allocated_mb is not None:
            print(f"Peak CUDA memory allocated: {peak_allocated_mb:.2f} MB")
            print(f"Peak CUDA memory reserved: {peak_reserved_mb:.2f} MB")
        print(f"Saved efficiency metrics: {json_path}")
        print(f"Saved efficiency table row: {csv_path}")

    def run_inference(self):
        print("Starting inference...")
        self.evaluator.reset()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                self.single_evaluator.reset()

                pre, post, labels_loc, labels_clf, file_name = data

                pre = pre.to(self.device)
                post = post.to(self.device)
                file_name = file_name[0]

                input_data = torch.cat([pre, post], dim=1)
                self.input_shapes.add(tuple(input_data.shape))
                self._synchronize_device()
                forward_start = time.perf_counter()
                loc_logits, logits = self._forward(input_data)
                self._synchronize_device()
                self.forward_latencies_ms.append(
                    (time.perf_counter() - forward_start) * 1000.0
                )

                output = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                output = output.squeeze(0)

                self.save_colored_map(output, file_name)
                self.save_original_map(output, file_name)

                labels_clf_np = labels_clf.squeeze().cpu().numpy()
                labels_loc_np = labels_loc.squeeze().cpu().numpy()

                # Localization F1 (binary: building vs background)
                if loc_logits is not None:
                    output_loc = torch.argmax(loc_logits, dim=1).cpu().numpy().astype(np.uint8)
                    output_loc = output_loc.squeeze(0)
                else:
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
        self._save_efficiency_metrics()

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
    parser.add_argument("--modality_mode", type=str, default="both",
                        choices=["both", "optical_only", "sar_only", "no_optical", "no_sar"],
                        help="Modality ablation mode for inference.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cuda:0, cuda:1, mps, or cpu",
    )
    parser.add_argument(
        "--timing_warmup",
        type=int,
        default=20,
        help="Number of initial forward passes excluded from latency statistics.",
    )
    flops_group = parser.add_mutually_exclusive_group()
    flops_group.add_argument(
        "--compute_flops",
        dest="compute_flops",
        action="store_true",
        help="Compute FLOPs once for the observed input shape (default).",
    )
    flops_group.add_argument(
        "--skip_flops",
        dest="compute_flops",
        action="store_false",
        help="Skip FLOP analysis.",
    )
    parser.set_defaults(compute_flops=True)

    args = parser.parse_args()

    with open(args.test_data_list_path) as f:
        args.test_data_list = [line.strip() for line in f]

    inference = Inference(args)
    inference.run_inference()
