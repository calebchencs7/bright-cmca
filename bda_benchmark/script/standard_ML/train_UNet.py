"""
Unified training script for building damage assessment.

Default hyper-parameters reproduce the BRIGHT paper baselines exactly.

Extensions (all opt-in, zero impact on baseline when unused):
    - DACutMix: event-aware, damage-aware CutMix across disaster events
    - Ordinal Damage Loss (ODL): exploits ordinal structure of damage levels
    - Damage Prototype Contrastive Learning (DPCL): organises pixel features
        into class-discriminative clusters via prototype InfoNCE on a 128-d
        projection of the dec3 decoder feature. Targets the persistently low
        Damaged-class IoU. Supports SP-DPCL (single proto/class) and MP-DPCL
        (multi proto/class) via --dpcl_num_prototypes.
    - PolyLR / CosineAnnealing schedule (default: constant LR, same as paper)
    - AMP mixed precision on CUDA
    - Class-weighted CE loss

Architecture when DPCL is enabled:
    backbone(input, return_features=True) -> (logits, dec3)
    main_loss(logits, labels) + lambda_dpcl(t) * DPCL(dec3, labels)

"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset.make_data_loader import (
    MultimodalDamageAssessmentDatset,
    parse_disaster_event,
)
from model.UNet import UNet
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L
from util_func.training_curve import TrainingCurveRecorder


# ---------------------------------------------------------------------------
# Helper: build backbone by name
# ---------------------------------------------------------------------------

def build_backbone(model_type, in_channels, num_classes):
    """Instantiate a segmentation backbone by name."""
    mt = model_type.lower()

    if mt == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)

    if mt == "unetwithfeatures":
        # DPCL-friendly subclass of UNet. Same parameters and state dict as
        # the baseline UNet; only forward() additionally exposes dec3.
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
# Ordinal Damage Loss (ODL) — exploits rank order of damage levels
# ---------------------------------------------------------------------------

def ordinal_damage_loss(logits, labels):
    """
    Ordinal rank-BCE loss for building damage classes 1/2/3.

    Motivation: damage levels have natural ordering (Intact < Damaged < Destroyed).
    Standard CE treats all misclassifications equally, but confusing Intact with
    Destroyed (2-rank error) should be penalized more than Intact with Damaged
    (1-rank error). This loss encodes that ordinal structure.

    Method: decompose the 3-class ordinal problem into 2 binary thresholds:
        - P(label > 1): is damage at least "Damaged"?
        - P(label > 2): is damage "Destroyed"?
    Then apply BCE on each threshold independently.

    Only computed on building pixels (label ∈ {1, 2, 3}). Background (label=0)
    and ignore (label=255) pixels are excluded.

    Args:
        logits: (B, 4, H, W) raw model output (4 classes: bg, intact, damaged, destroyed)
        labels: (B, H, W) ground truth labels

    Returns:
        Scalar loss, or 0 if no valid building pixels exist.
    """
    valid = (labels > 0) & (labels != 255)
    if not torch.any(valid):
        return logits.new_tensor(0.0)

    # Softmax over damage sub-classes only: P(intact), P(damaged), P(destroyed)
    damage_probs = F.softmax(logits[:, 1:4], dim=1)  # (B, 3, H, W)

    # Extract valid building pixels: (N, 3)
    flat_probs = damage_probs.permute(0, 2, 3, 1)[valid]
    flat_labels = labels[valid]

    # Cumulative probabilities for ordinal thresholds
    p_gt1 = (flat_probs[:, 1] + flat_probs[:, 2]).clamp(1e-6, 1 - 1e-6)  # P(≥Damaged)
    p_gt2 = flat_probs[:, 2].clamp(1e-6, 1 - 1e-6)                       # P(=Destroyed)

    # Binary targets
    t_gt1 = (flat_labels > 1).float()  # 1 if Damaged or Destroyed
    t_gt2 = (flat_labels > 2).float()  # 1 if Destroyed

    # BCE in fp32 for numerical stability (not AMP-safe otherwise)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        loss_gt1 = F.binary_cross_entropy(p_gt1.float(), t_gt1, reduction="mean")
        loss_gt2 = F.binary_cross_entropy(p_gt2.float(), t_gt2, reduction="mean")

    return loss_gt1 + loss_gt2


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Clean, modular trainer for building damage assessment.

    Default configuration reproduces the BRIGHT paper baselines:
        - AdamW, lr=1e-4, weight_decay=5e-3
        - Constant LR (no scheduler)
        - CE + 0.75 * Lovász softmax
        - max_iters controls dataset length (sample-level budget)
    """

    def __init__(self, args):
        self.args = args
        self.device = self._resolve_device(args.device)
        print(f"Using device: {self.device}")

        # Evaluators
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_total = Evaluator(num_class=4)

        self.damage_class_ids = _parse_int_csv(args.damage_class_ids, "damage_class_ids")
        print(f"Damage class ids: {self.damage_class_ids}")
        if args.use_dacutmix:
            print(
                "DACutMix enabled "
                f"(p={args.dacutmix_prob}, damage_ids={self.damage_class_ids}, "
                f"min_pixels={args.dacutmix_min_damage_pixels}, "
                f"min_ratio={args.dacutmix_min_damage_ratio})"
            )
        # Intervals
        self.eval_interval = max(1, int(args.eval_interval))
        self.log_interval = max(1, int(args.curve_log_interval))
        self.save_interval = max(1, int(args.curve_save_interval))

        # DataLoader options
        self.pin_memory = bool(args.pin_memory and self.device.type == "cuda")
        self.persistent_workers = bool(
            args.persistent_workers and args.num_workers and args.num_workers > 0
        )
        self.prefetch_factor = max(1, int(args.prefetch_factor))

        # AMP
        self.use_amp = bool(args.use_amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        # CUDA optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # ---- Build backbone ----
        in_channels = 6  # pre(3) + post(3), always 6
        self.backbone = build_backbone(args.model_type, in_channels, num_classes=4)
        self.backbone = self.backbone.to(self.device)

        self.grad_clip_norm = max(0.0, float(getattr(args, "grad_clip_norm", 0.0)))

        # ---- Ordinal Damage Loss (optional) ----
        self.use_ordinal = bool(args.use_ordinal_loss)
        self.ordinal_weight = max(0.0, float(args.ordinal_weight))
        self.ordinal_warmup_iters = max(0, int(args.ordinal_warmup_iters))
        if self.use_ordinal:
            if self.ordinal_warmup_iters > 0:
                print(f"Ordinal Damage Loss enabled (weight={self.ordinal_weight}, "
                      f"warmup={self.ordinal_warmup_iters} iters linear ramp)")
            else:
                print(f"Ordinal Damage Loss enabled (weight={self.ordinal_weight}, no warmup)")

        # ---- Damage Prototype Contrastive Learning (optional) ----
        self.use_dpcl = bool(args.use_dpcl)
        self.dpcl_weight = max(0.0, float(args.dpcl_weight))
        self.dpcl = None
        if self.use_dpcl:
            # Sanity-check: DPCL needs the backbone to expose decoder features.
            # The baseline `UNet` does NOT — use `UNetWithFeatures` instead.
            mt_lower = args.model_type.lower()
            dpcl_compatible = ("unetwithfeatures", "unetcmca")
            if mt_lower not in dpcl_compatible:
                raise ValueError(
                    f"--use_dpcl requires a backbone that exposes decoder "
                    f"features via forward(x, return_features=True). "
                    f"Got --model_type {args.model_type!r}. "
                    f"Use one of: {dpcl_compatible}. "
                    f"(The vanilla 'UNet' is the baseline reference and does "
                    f"not return features; use 'UNetWithFeatures' for DPCL.)"
                )

            from model.DPCL import DamagePrototypeContrastiveLoss

            # Parse --dpcl_num_prototypes "1,1,1" → {1:1, 2:1, 3:1}
            kk = [int(x) for x in str(args.dpcl_num_prototypes).split(",")]
            if len(kk) != 3:
                raise ValueError(
                    "--dpcl_num_prototypes must be 3 comma-separated ints "
                    "(intact, damaged, destroyed), e.g. '1,1,1' or '1,3,2'."
                )
            num_proto = {1: kk[0], 2: kk[1], 3: kk[2]}

            # Parse --dpcl_class_loss_weights "1.0,2.0,1.0" → {1:1.0, 2:2.0, 3:1.0}
            ww = [float(x) for x in str(args.dpcl_class_loss_weights).split(",")]
            if len(ww) != 3:
                raise ValueError(
                    "--dpcl_class_loss_weights must be 3 floats "
                    "(intact, damaged, destroyed)."
                )
            cls_w = {1: ww[0], 2: ww[1], 3: ww[2]}

            # dec3 channel count is 256 for both UNet and UNetCMCA
            feat_dim = int(args.dpcl_feat_dim)

            self.dpcl = DamagePrototypeContrastiveLoss(
                feat_dim=feat_dim,
                proj_dim=int(args.dpcl_proj_dim),
                num_prototypes_per_class=num_proto,
                samples_per_class=int(args.dpcl_samples_per_class),
                warmup_iters=int(args.dpcl_warmup_iters),
                ramp_iters=int(args.dpcl_ramp_iters),
                momentum=float(args.dpcl_momentum),
                temperature=float(args.dpcl_temperature),
                class_loss_weights=cls_w,
                ortho_weight=float(args.dpcl_ortho_weight),
            ).to(self.device)

            mode = "SP-DPCL" if all(k == 1 for k in kk) else "MP-DPCL"
            print(f"Damage Prototype Contrastive Learning enabled ({mode})")
            print(f"  num_protos={num_proto}, weight={self.dpcl_weight}, "
                  f"τ={args.dpcl_temperature}, m={args.dpcl_momentum}, "
                  f"warmup={args.dpcl_warmup_iters}+ramp={args.dpcl_ramp_iters} iters")

        # ---- Save path ----
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts = []
        if self.use_ordinal:
            suffix_parts.append("ODL")
        if self.use_dpcl:
            dpcl_suffix = (
                "MPDPCL" if (self.dpcl is not None and not self.dpcl.is_single_proto)
                else "SPDPCL"
            )
            suffix_parts.append(dpcl_suffix)
        if args.use_dacutmix:
            suffix_parts.append("DACutMix")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        self.model_save_path = os.path.join(
            args.model_param_path, args.dataset,
            args.model_type + suffix + "_" + now_str
        )
        os.makedirs(self.model_save_path, exist_ok=True)
        self.curve_recorder = TrainingCurveRecorder(self.model_save_path)

        # ---- Resume ----
        if args.resume:
            self._load_checkpoint(args.resume)

        # ---- Optimizer: joint params ----
        params = list(self.backbone.parameters())
        if self.dpcl is not None:
            # Only the projection head has trainable parameters; prototypes
            # are buffers updated via EMA, not SGD.
            params += list(self.dpcl.proj.parameters())
        self.trainable_params = params
        self.optim = optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # ---- Class weights ----
        self.class_weights = None
        if args.class_weights:
            weights = [float(x) for x in args.class_weights.split(",")]
            if len(weights) != 4:
                raise ValueError("class_weights must have 4 comma-separated values.")
            self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # ---- LR scheduler ----
        self.scheduler = self._build_scheduler(args)

    # -----------------------------------------------------------------------
    # Device / schedule
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_arg):
        if device_arg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if device_arg == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS is not available.")
        return torch.device(device_arg)

    def _build_scheduler(self, args):
        if args.lr_policy == "constant":
            return None

        # Compute total steps for schedule
        total_steps = max(1, int(np.ceil(
            float(args.max_iters) / float(max(1, args.train_batch_size))
        )))

        if args.lr_policy == "poly":
            warmup = max(0, int(args.warmup_iters))
            power = args.lr_power

            def lr_lambda(step):
                if step < warmup:
                    return max(1e-6, float(step + 1) / float(max(1, warmup)))
                progress = float(step - warmup) / float(max(1, total_steps - warmup))
                return max(1e-6, (1.0 - progress) ** power)

            return optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)

        if args.lr_policy == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=total_steps, eta_min=1e-7
            )

        return None

    def _load_checkpoint(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        # Handle both old-style (flat state_dict) and new-style (nested dict)
        if "backbone" in ckpt:
            self.backbone.load_state_dict(ckpt["backbone"], strict=False)
            if self.dpcl is not None and "dpcl" in ckpt:
                self.dpcl.load_state_dict(ckpt["dpcl"], strict=False)
        else:
            # Legacy format: flat state_dict
            backbone_state = {k: v for k, v in ckpt.items()
                              if k in self.backbone.state_dict()}
            self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded checkpoint from {path}")

    # -----------------------------------------------------------------------
    # Dataset factory
    # -----------------------------------------------------------------------

    def _make_dataset(self, dataset_path, data_list, crop_size, max_iters, split):
        return MultimodalDamageAssessmentDatset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            type=split,
            suffix=".tif",
            use_dacutmix=(split == "train" and self.args.use_dacutmix),
            dacutmix_prob=self.args.dacutmix_prob,
            damage_class_ids=self.damage_class_ids,
            dacutmix_min_damage_pixels=self.args.dacutmix_min_damage_pixels,
            dacutmix_min_damage_ratio=self.args.dacutmix_min_damage_ratio,
            dacutmix_patch_min_ratio=self.args.dacutmix_patch_min_ratio,
            dacutmix_patch_max_ratio=self.args.dacutmix_patch_max_ratio,
            dacutmix_box_tries=self.args.dacutmix_box_tries,
            dacutmix_donor_tries=self.args.dacutmix_donor_tries,
            return_dacutmix_stats=(split == "train" and self.args.use_dacutmix),
        )

    # -----------------------------------------------------------------------
    # Batch preparation
    # -----------------------------------------------------------------------

    def _prepare_batch(self, data, return_ids=False):
        """Unpack batch. Returns (input, labels_loc, labels_clf)."""
        pre, post, labels_loc, labels_clf, data_idx = data[:5]

        pre = pre.to(self.device)
        post = post.to(self.device)
        labels_loc = labels_loc.to(self.device).long()
        labels_clf = labels_clf.to(self.device).long()

        input_data = torch.cat([pre, post], dim=1)
        if return_ids:
            return input_data, labels_loc, labels_clf, data_idx
        return input_data, labels_loc, labels_clf

    @staticmethod
    def _dacutmix_batch_counts(data):
        if len(data) < 7:
            return 0, 0
        attempted = data[5]
        applied = data[6]
        if torch.is_tensor(attempted):
            attempted = int(attempted.sum().item())
        else:
            attempted = int(np.asarray(attempted).sum())
        if torch.is_tensor(applied):
            applied = int(applied.sum().item())
        else:
            applied = int(np.asarray(applied).sum())
        return attempted, applied

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def _forward(
        self,
        input_data,
        return_features=False,
    ):
        """
        If `return_features` is True, also returns the dec3 mid-decoder
        feature map for use by DPCL or other auxiliary losses.

        Returns:
            logits: (B, num_classes, H, W)
            feat_dec3: (B, 256, H/4, W/4)   only if return_features=True
        """
        feat = None

        if return_features:
            logits, feat = self.backbone(input_data, return_features=True)
        else:
            logits = self.backbone(input_data)

        if return_features:
            return logits, feat
        return logits

    # -----------------------------------------------------------------------
    # Loss — baseline: CE + 0.75 * Lovász; optional: + ordinal
    # -----------------------------------------------------------------------

    def _effective_ordinal_weight(self, current_iter):
        """
        Linear warmup schedule for ordinal loss weight.

        Phase 1 [0, warmup):          weight = 0   (ODL silent, main loss stabilises)
        Phase 2 [warmup, 2*warmup):   weight ramps linearly 0 → ordinal_weight
        Phase 3 [2*warmup, ∞):        weight = ordinal_weight (full strength)

        If ordinal_warmup_iters == 0, always returns ordinal_weight (no warmup).
        """
        if self.ordinal_warmup_iters <= 0:
            return self.ordinal_weight
        if current_iter < self.ordinal_warmup_iters:
            return 0.0
        ramp = min(1.0, (current_iter - self.ordinal_warmup_iters)
                   / float(self.ordinal_warmup_iters))
        return self.ordinal_weight * ramp

    def _compute_loss(
        self,
        logits,
        labels_clf,
        current_iter=0,
        feat_dec3=None,
    ):
        """
        CE + 0.75 * Lovász softmax + optional ordinal damage loss + optional DPCL.

        current_iter is used to compute the warmup-adjusted ordinal / DPCL weights.
        feat_dec3 is the dec3 decoder feature; required when self.dpcl is enabled.

        NOTE: The original BRIGHT paper code does NOT use ignore_index=255 in
        cross_entropy. We keep that behaviour as default. Lovász already handles
        ignore=255 via its own masking.
        """
        loss_logits = logits.float()
        if self.class_weights is not None:
            ce = F.cross_entropy(loss_logits, labels_clf, weight=self.class_weights)
        else:
            ce = F.cross_entropy(loss_logits, labels_clf)

        lovasz = L.lovasz_softmax(
            F.softmax(loss_logits, dim=1), labels_clf, ignore=255
        )

        total = ce + 0.75 * lovasz

        # Optional: ordinal damage loss with warmup
        ord_loss = loss_logits.new_tensor(0.0)
        if self.use_ordinal:
            ord_loss = ordinal_damage_loss(loss_logits, labels_clf)
            eff_w = self._effective_ordinal_weight(current_iter)
            total = total + eff_w * ord_loss

        # Optional: damage prototype contrastive loss with warmup
        dpcl_loss = loss_logits.new_tensor(0.0)
        if self.dpcl is not None and feat_dec3 is not None:
            # DPCL is computed in fp32 internally; safe under AMP autocast
            # because the projection head is a small MLP and prototype math
            # is forced to fp32 inside DamagePrototypeContrastiveLoss.
            dpcl_loss = self.dpcl(feat_dec3, labels_clf, current_iter=current_iter)
            eff_w_dpcl = self.dpcl.effective_weight(current_iter, self.dpcl_weight)
            if eff_w_dpcl > 0.0:
                total = total + eff_w_dpcl * dpcl_loss

        return total, ce, lovasz, ord_loss, dpcl_loss

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def training(self):
        best_mIoU = 0.0
        best_round = {}

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Paper convention: max_iters = number of samples to see
        train_dataset = self._make_dataset(
            dataset_path=self.args.train_dataset_path,
            data_list=self.args.train_data_name_list,
            crop_size=self.args.crop_size,
            max_iters=self.args.max_iters,
            split="train",
        )

        loader_kwargs = {
            "batch_size": self.args.train_batch_size,
            "shuffle": True,
            "num_workers": self.args.num_workers,
            "drop_last": False,
            "pin_memory": self.pin_memory,
        }
        if self.args.num_workers and self.args.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        train_loader = DataLoader(train_dataset, **loader_kwargs)
        elem_num = len(train_loader)

        self.backbone.train()
        if self.dpcl is not None:
            self.dpcl.train()

        train_enumerator = enumerate(train_loader)
        pbar = tqdm(total=elem_num, desc="Training")
        dacutmix_attempted_window = 0
        dacutmix_applied_window = 0
        dacutmix_seen_window = 0
        dacutmix_attempted_total = 0
        dacutmix_applied_total = 0
        dacutmix_seen_total = 0

        try:
            for itera, data in train_enumerator:
                pbar.update(1)
                dacutmix_attempted, dacutmix_applied = self._dacutmix_batch_counts(data)
                batch_size_seen = int(data[0].shape[0]) if hasattr(data[0], "shape") else 0
                dacutmix_attempted_window += dacutmix_attempted
                dacutmix_applied_window += dacutmix_applied
                dacutmix_seen_window += batch_size_seen
                dacutmix_attempted_total += dacutmix_attempted
                dacutmix_applied_total += dacutmix_applied
                dacutmix_seen_total += batch_size_seen

                input_data, labels_loc, labels_clf = self._prepare_batch(data)

                # Skip empty label batches
                if not (labels_clf != 255).any().item():
                    continue

                self.optim.zero_grad(set_to_none=True)

                amp_ctx = torch.autocast(
                    device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                )
                with amp_ctx:
                    feat_dec3 = None
                    need_dec3 = self.dpcl is not None

                    if need_dec3:
                        logits, feat_dec3 = self._forward(
                            input_data, return_features=True
                        )
                    else:
                        logits = self._forward(input_data)

                (
                    total_loss, ce_loss, lovasz_loss, ord_loss,
                    dpcl_loss,
                ) = (
                    self._compute_loss(
                        logits, labels_clf,
                        current_iter=itera,
                        feat_dec3=feat_dec3,
                    )
                )

                if not torch.isfinite(total_loss).item():
                    print(
                        f"[WARN] non-finite loss at iter {itera+1}; "
                        "skipping optimizer step to avoid corrupting weights."
                    )
                    self.optim.zero_grad(set_to_none=True)
                    continue

                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    if self.grad_clip_norm > 0.0:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params, self.grad_clip_norm
                        )
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if self.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params, self.grad_clip_norm
                        )
                    self.optim.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.curve_recorder.add_train_loss(itera + 1, total_loss.item())

                # Logging
                if (itera + 1) % self.log_interval == 0:
                    lr = self.optim.param_groups[0]["lr"]
                    msg = (f"iter {itera+1} | total={total_loss.item():.4f} "
                           f"ce={ce_loss.item():.4f} lovasz={lovasz_loss.item():.4f}")
                    if self.use_ordinal:
                        eff_w = self._effective_ordinal_weight(itera)
                        msg += f" ord={ord_loss.item():.4f}(w={eff_w:.4f})"
                    if self.dpcl is not None:
                        eff_w_dpcl = self.dpcl.effective_weight(itera, self.dpcl_weight)
                        msg += f" dpcl={dpcl_loss.item():.4f}(w={eff_w_dpcl:.4f})"
                    msg += f" lr={lr:.2e}"
                    if self.args.use_dacutmix and dacutmix_seen_window > 0:
                        attempt_rate = 100.0 * dacutmix_attempted_window / max(1, dacutmix_seen_window)
                        success_rate = 100.0 * dacutmix_applied_window / max(1, dacutmix_seen_window)
                        accept_rate = 100.0 * dacutmix_applied_window / max(1, dacutmix_attempted_window)
                        total_success = 100.0 * dacutmix_applied_total / max(1, dacutmix_seen_total)
                        msg += (
                            f" dacutmix={dacutmix_applied_window}/{dacutmix_attempted_window}"
                            f"/{dacutmix_seen_window}"
                            f"(succ={success_rate:.1f}%,"
                            f" try={attempt_rate:.1f}%,"
                            f" acc={accept_rate:.1f}%,"
                            f" total={total_success:.1f}%)"
                        )
                        dacutmix_attempted_window = 0
                        dacutmix_applied_window = 0
                        dacutmix_seen_window = 0
                    print(msg)

                # Evaluation
                if (itera + 1) % self.eval_interval == 0:
                    self.backbone.eval()

                    val_metrics = self._evaluate("val")
                    test_metrics = self._evaluate("test")

                    self.curve_recorder.add_eval_metrics(
                        itera + 1, "val",
                        val_metrics["OA"] * 100, val_metrics["mIoU"] * 100,
                        val_metrics["event_mIoU_std"] * 100,
                        val_metrics["event_mIoU_min"] * 100,
                        val_metrics["event_mIoU_p25"] * 100,
                    )
                    self.curve_recorder.add_eval_metrics(
                        itera + 1, "test",
                        test_metrics["OA"] * 100, test_metrics["mIoU"] * 100,
                        test_metrics["event_mIoU_std"] * 100,
                        test_metrics["event_mIoU_min"] * 100,
                        test_metrics["event_mIoU_p25"] * 100,
                    )

                    if val_metrics["mIoU"] > best_mIoU:
                        best_mIoU = val_metrics["mIoU"]
                        self._save_checkpoint(itera + 1)
                        best_round = {
                            "best iter": itera + 1,
                            "loc f1 (val)": val_metrics["loc_f1"] * 100,
                            "clf f1 (val)": val_metrics["clf_f1"] * 100,
                            "OA (val)": val_metrics["OA"] * 100,
                            "mIoU (val)": val_metrics["mIoU"] * 100,
                            "sub class IoU (val)": val_metrics["IoU_per_class"] * 100,
                            "event mIoU std (val)": val_metrics["event_mIoU_std"] * 100,
                            "worst event mIoU (val)": val_metrics["event_mIoU_min"] * 100,
                            "event mIoU p25 (val)": val_metrics["event_mIoU_p25"] * 100,
                            "loc f1 (test)": test_metrics["loc_f1"] * 100,
                            "clf f1 (test)": test_metrics["clf_f1"] * 100,
                            "OA (test)": test_metrics["OA"] * 100,
                            "mIoU (test)": test_metrics["mIoU"] * 100,
                            "sub class IoU (test)": test_metrics["IoU_per_class"] * 100,
                            "event mIoU std (test)": test_metrics["event_mIoU_std"] * 100,
                            "worst event mIoU (test)": test_metrics["event_mIoU_min"] * 100,
                            "event mIoU p25 (test)": test_metrics["event_mIoU_p25"] * 100,
                        }

                    self.backbone.train()
                    self.curve_recorder.save()

                elif (itera + 1) % self.save_interval == 0:
                    self.curve_recorder.save()

        finally:
            pbar.close()
            self.curve_recorder.save()
            print(f"Training curves saved to {self.model_save_path}")
            print(f"The accuracy of the best round is {best_round}")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def _evaluate(self, split="val"):
        """Run evaluation on val or test set."""
        self.evaluator_total.reset()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        event_evaluators = {}

        if split == "val":
            dataset_path = self.args.val_dataset_path
            data_list = self.args.val_data_name_list
            print("---------starting validation-----------")
        else:
            dataset_path = self.args.test_dataset_path
            data_list = self.args.test_data_name_list
            print("---------starting testing-----------")

        eval_dataset = self._make_dataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=1024,
            max_iters=None,
            split="test",
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=self.args.eval_batch_size,
            num_workers=1, drop_last=False
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for data in eval_loader:
                input_data, labels_loc, labels_clf, data_idx = self._prepare_batch(
                    data, return_ids=True
                )
                logits = self._forward(input_data)

                labels_loc_np = labels_loc.cpu().numpy()
                labels_clf_np = labels_clf.cpu().numpy()
                pred_clf = logits.data.cpu().numpy().argmax(axis=1)

                pred_loc = pred_clf.copy()
                pred_loc[pred_loc > 0] = 1

                self.evaluator_loc.add_batch(labels_loc_np, pred_loc)
                self.evaluator_clf.add_batch(
                    labels_clf_np[labels_loc_np > 0],
                    pred_clf[labels_loc_np > 0],
                )
                self.evaluator_total.add_batch(labels_clf_np, pred_clf)

                for b, name in enumerate(data_idx):
                    event = parse_disaster_event(name)
                    if event not in event_evaluators:
                        event_evaluators[event] = Evaluator(num_class=4)
                    event_evaluators[event].add_batch(labels_clf_np[b], pred_clf[b])

        loc_f1 = self.evaluator_loc.Pixel_F1_score()
        damage_f1 = self.evaluator_clf.Damage_F1_score()
        harmonic_f1 = _safe_hmean(damage_f1)
        OA = self.evaluator_total.Pixel_Accuracy()
        IoU_per_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        per_event_miou = {
            event: evaluator.Mean_Intersection_over_Union()
            for event, evaluator in event_evaluators.items()
        }
        per_event_values = np.asarray(list(per_event_miou.values()), dtype=np.float32)
        if per_event_values.size > 0:
            event_miou_std = float(np.std(per_event_values))
            event_miou_min = float(np.min(per_event_values))
            event_miou_p25 = float(np.percentile(per_event_values, 25))
        else:
            event_miou_std = 0.0
            event_miou_min = 0.0
            event_miou_p25 = 0.0

        tag = "VAL" if split == "val" else "TEST"
        print(f"[{tag}] OA={100*OA:.4f}, mIoU={100*mIoU:.4f}, IoU={100*IoU_per_class}")
        print(
            f"[{tag}] event_mIoU std={100*event_miou_std:.4f}, "
            f"worst={100*event_miou_min:.4f}, p25={100*event_miou_p25:.4f}"
        )

        return {
            "loc_f1": loc_f1,
            "clf_f1": harmonic_f1,
            "OA": OA,
            "mIoU": mIoU,
            "IoU_per_class": IoU_per_class,
            "per_event_mIoU": per_event_miou,
            "event_mIoU_std": event_miou_std,
            "event_mIoU_min": event_miou_min,
            "event_mIoU_p25": event_miou_p25,
        }

    def _save_checkpoint(self, step):
        """Save backbone + optional DPCL state dict."""
        state = {"backbone": self.backbone.state_dict(), "step": step}
        if self.dpcl is not None:
            # Save the full DPCL state (projection head weights + prototype
            # buffers + assignment counters), in case we want to inspect or
            # resume contrastive training.
            state["dpcl"] = self.dpcl.state_dict()
        torch.save(state, os.path.join(self.model_save_path, "best_model.pth"))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_int_csv(value, name):
    try:
        out = tuple(int(x.strip()) for x in str(value).split(",") if x.strip())
    except Exception as exc:
        raise ValueError(f"{name} must be a comma-separated list of ints.") from exc
    if not out:
        raise ValueError(f"{name} must contain at least one class id.")
    return out


def _safe_hmean(scores, eps=1e-6):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    scores = np.where(scores <= 0, eps, scores)
    return len(scores) / np.sum(1.0 / scores)


# ---------------------------------------------------------------------------
# CLI — defaults match the original BRIGHT paper code exactly
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")

    # Data
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--train_data_list_path", type=str)
    parser.add_argument("--val_dataset_path", type=str)
    parser.add_argument("--val_data_list_path", type=str)
    parser.add_argument("--test_dataset_path", type=str)
    parser.add_argument("--test_data_list_path", type=str)

    # Training — paper defaults
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=800000)

    # Model
    parser.add_argument("--model_type", type=str, default="UNet")
    parser.add_argument("--model_param_path", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Optimizer — paper defaults: lr=1e-4, wd=5e-3, constant LR
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--lr_policy", type=str, default="constant",
                        choices=["constant", "poly", "cosine"],
                        help="LR schedule. Paper default: constant.")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="Poly LR decay power (only for poly schedule).")
    parser.add_argument("--warmup_iters", type=int, default=1000,
                        help="Linear warmup steps (only for poly/cosine).")

    # Loss
    parser.add_argument("--class_weights", type=str, default=None,
                        help="Comma-separated weights for 4 classes, e.g. 1,1,2,2")
    parser.add_argument("--damage_class_ids", type=str, default="2,3",
                        help="Comma-separated class ids treated as damaged for "
                             "DACutMix and damage-balanced sampling.")

    # DACutMix (Damage-Aware CutMix) — all opt-in.
    parser.add_argument("--use_dacutmix", action="store_true",
                        help="Enable event-aware, damage-aware CutMix "
                             "(DACutMix) in the training dataset.")
    parser.add_argument("--dacutmix_prob", type=float, default=0.5,
                        help="Probability of applying DACutMix to a training sample.")
    parser.add_argument("--dacutmix_min_damage_pixels", type=int, default=200,
                        help="Minimum damaged pixels required in a donor patch.")
    parser.add_argument("--dacutmix_min_damage_ratio", type=float, default=0.05,
                        help="Minimum donor patch damage ratio required for CutMix.")
    parser.add_argument("--dacutmix_patch_min_ratio", type=float, default=0.20,
                        help="Minimum patch side length as a fraction of crop size.")
    parser.add_argument("--dacutmix_patch_max_ratio", type=float, default=0.50,
                        help="Maximum patch side length as a fraction of crop size.")
    parser.add_argument("--dacutmix_box_tries", type=int, default=10,
                        help="Random boxes to try per donor sample.")
    parser.add_argument("--dacutmix_donor_tries", type=int, default=10,
                        help="Cross-event donor samples to try before falling back "
                             "to the un-mixed image.")

    # Ordinal Damage Loss (ODL) — opt-in
    parser.add_argument("--use_ordinal_loss", action="store_true",
                        help="Enable ordinal damage loss (rank-BCE on damage levels).")
    parser.add_argument("--ordinal_weight", type=float, default=0.15,
                        help="Target weight for ordinal damage loss term.")
    parser.add_argument("--ordinal_warmup_iters", type=int, default=0,
                        help="Linear warmup iters for ODL weight. "
                             "0 = no warmup (apply full weight from iter 0). "
                             "N > 0: ODL is silent for first N iters, then ramps "
                             "linearly to ordinal_weight over the next N iters.")

    # Damage Prototype Contrastive Learning (DPCL) — opt-in
    parser.add_argument("--use_dpcl", action="store_true",
                        help="Enable Damage Prototype Contrastive Learning.")
    parser.add_argument("--dpcl_weight", type=float, default=0.1,
                        help="Target weight for DPCL loss term after warmup.")
    parser.add_argument("--dpcl_num_prototypes", type=str, default="1,1,1",
                        help="Number of prototypes per building class as "
                             "'intact,damaged,destroyed'. SP-DPCL: '1,1,1'. "
                             "MP-DPCL example: '1,3,2'.")
    parser.add_argument("--dpcl_class_loss_weights", type=str, default="1.0,2.0,1.0",
                        help="Per-class loss weight for DPCL InfoNCE, "
                             "'intact,damaged,destroyed'. Damaged is upweighted "
                             "by default to address its lowest IoU.")
    parser.add_argument("--dpcl_feat_dim", type=int, default=256,
                        help="Channel count of the hooked decoder feature. "
                             "256 for dec3 in UNet/UNetCMCA at crop=640.")
    parser.add_argument("--dpcl_proj_dim", type=int, default=128,
                        help="Embedding dimension after DPCL projection head.")
    parser.add_argument("--dpcl_samples_per_class", type=int, default=512,
                        help="Max pixels sampled per class per batch.")
    parser.add_argument("--dpcl_warmup_iters", type=int, default=3000,
                        help="Iterations during which DPCL only updates "
                             "prototypes; loss is masked out.")
    parser.add_argument("--dpcl_ramp_iters", type=int, default=2000,
                        help="Iterations of linear ramp from 0 to dpcl_weight "
                             "after warmup completes.")
    parser.add_argument("--dpcl_momentum", type=float, default=0.99,
                        help="EMA momentum for prototype updates. "
                             "Smaller -> faster adaptation. 0.99 recommended.")
    parser.add_argument("--dpcl_temperature", type=float, default=0.1,
                        help="Softmax temperature τ for InfoNCE.")
    parser.add_argument("--dpcl_ortho_weight", type=float, default=0.0,
                        help="Weight on orthogonality regulariser between "
                             "sub-prototypes of the same class. Only relevant "
                             "for MP-DPCL (K>1 for some class). 0.01 suggested.")
    # Performance
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda","cuda:0", "cuda:1", "mps", "cpu"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"])
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0,
                        help="Clip global gradient norm when > 0. Useful for AMP stability.")

    # Logging
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--curve_log_interval", type=int, default=10)
    parser.add_argument("--curve_save_interval", type=int, default=500)

    # Internal (backward compat)
    parser.add_argument("--train_data_name_list", type=list, default=None)
    parser.add_argument("--val_data_name_list", type=list, default=None)
    parser.add_argument("--test_data_name_list", type=list, default=None)

    args = parser.parse_args()

    # Load split files
    with open(args.train_data_list_path) as f:
        args.train_data_name_list = [line.strip() for line in f]
    with open(args.val_data_list_path) as f:
        args.val_data_name_list = [line.strip() for line in f]
    with open(args.test_data_list_path) as f:
        args.test_data_name_list = [line.strip() for line in f]

    if args.val_data_name_list == args.test_data_name_list:
        print("[WARN] val and test lists are identical.")

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
