"""
DPCL: Damage Prototype Contrastive Learning
============================================

A drop-in auxiliary loss module that organises pixel features into
class-discriminative clusters in a learned embedding space, addressing the
core failure mode of building damage assessment: the "Damaged" class is
visually ambiguous and overlaps with both "Intact" and "Destroyed" in feature
space, leading to its persistently low IoU (35–48% across all baselines on
BRIGHT, vs 88–90% for Intact and 55–65% for Destroyed).

Why DPCL works where ODL fails
------------------------------
Ordinal Damage Loss (ODL) supervises softmax probabilities with rank-BCE,
which is information-theoretically redundant with cross-entropy: both push
the same softmax outputs around. ODL adds no NEW representational constraint.

DPCL operates on a SEPARATE embedding space (a 128-d projection of decoder
features) and explicitly shapes the geometry of that space:
    - Pull each pixel feature toward its class prototype (intra-class compact).
    - Push each pixel feature away from other classes' prototypes (inter-class).
    - Up-weight the loss for the Damaged class so its cluster becomes tighter.

Architectural design
--------------------
1. Projection head: Conv1×1 -> BN -> ReLU -> Conv1×1 (per-pixel MLP), taking
   decoder mid-layer features (e.g., dec3 at 160x160, 256 ch in UNet/UNetCMCA
   at crop=640) and mapping them to 128-d L2-normalised vectors.

2. Class prototypes: K_total × proj_dim L2-normalised vectors, registered as
   buffers and updated by EMA with momentum 0.99. Background (class 0) is NOT
   used as an anchor — it dominates pixels and is too heterogeneous.

3. Class-balanced pixel sampling: from each batch, sample at most N pixels per
   building-class (Intact / Damaged / Destroyed). Pixels with label == 255
   (ignore) are excluded. This prevents Intact (the dominant building class)
   from drowning out Damaged in the contrastive signal.

4. InfoNCE loss: for each sampled feature, compute scaled dot-product
   similarity to all prototypes and apply softmax cross-entropy with the true
   class as target. Class-level loss weights up-weight Damaged.

5. Warmup: phase 1 (iter < warmup_iters) only updates prototypes, no loss
   gradient. Phase 2 (warmup_iters → warmup_iters + ramp_iters) linearly
   ramps the loss weight from 0 to dpcl_weight. Phase 3: full weight.

Single-prototype (SP-DPCL, this v1) vs multi-prototype (MP-DPCL, future v2)
--------------------------------------------------------------------------
SP-DPCL: 1 prototype per building class (3 prototypes total). Suitable as the
first ablation. Supported here with `num_prototypes_per_class={1:1,2:1,3:1}`.

MP-DPCL: multiple prototypes per class (e.g., intact=1, damaged=3, destroyed=2),
with logsumexp soft-positive InfoNCE for loss and hard-nearest assignment for
EMA updates. Also supported by this code (just pass a different K dict). Add
orthogonality reg by setting `ortho_weight > 0`.

Usage
-----
    from model.DPCL import DamagePrototypeContrastiveLoss

    dpcl = DamagePrototypeContrastiveLoss(
        feat_dim=256,           # dec3 channel count for UNet/UNetCMCA
        proj_dim=128,
        num_prototypes_per_class={1: 1, 2: 1, 3: 1},  # SP-DPCL
        samples_per_class=512,
        warmup_iters=3000,
        ramp_iters=2000,
        momentum=0.99,
        temperature=0.1,
        class_loss_weights={1: 1.0, 2: 2.0, 3: 1.0},
    ).to(device)

    # During training:
    logits, feat_dec3 = backbone(x, return_features=True)
    main_loss = ce + 0.75 * lovasz
    dpcl_loss = dpcl(feat_dec3, labels_clf, current_iter=itera)
    total = main_loss + dpcl.effective_weight(itera, base_weight=0.1) * dpcl_loss
    total.backward()

Notes
-----
- Prototype updates are done under torch.no_grad() (EMA, not SGD).
- All projected features are L2-normalised so similarity is cosine.
- AMP-safe: prototype updates and InfoNCE are computed in fp32 to avoid
  numerical issues with normalised vectors at fp16 precision.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _normalize_class_weight_dict(d, classes=(1, 2, 3), default=1.0):
    """Return a dict {c: float} for the building classes, filling missing keys."""
    out = {}
    for c in classes:
        out[c] = float(default if d is None else d.get(c, default))
    return out


# --------------------------------------------------------------------------
# DPCL module
# --------------------------------------------------------------------------

class DamagePrototypeContrastiveLoss(nn.Module):
    """
    Damage Prototype Contrastive Learning.

    SP-DPCL (v1): single prototype per building class.
    MP-DPCL (v2): multiple prototypes per class.

    See module docstring for details and usage.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        proj_dim: int = 128,
        num_prototypes_per_class: Optional[Dict[int, int]] = None,
        samples_per_class: int = 512,
        warmup_iters: int = 3000,
        ramp_iters: int = 2000,
        momentum: float = 0.99,
        temperature: float = 0.1,
        class_loss_weights: Optional[Dict[int, float]] = None,
        ortho_weight: float = 0.0,
    ):
        super().__init__()

        # ---------- Class / prototype layout ----------
        if num_prototypes_per_class is None:
            num_prototypes_per_class = {1: 1, 2: 1, 3: 1}
        self.classes = tuple(sorted(num_prototypes_per_class.keys()))   # e.g., (1,2,3)
        self.K_per_class = {c: int(num_prototypes_per_class[c]) for c in self.classes}
        assert all(K >= 1 for K in self.K_per_class.values()), \
            "Each class must have at least 1 prototype."

        # Flattened prototype index ↔ (class, sub-index)
        proto_class = []         # length K_total, value ∈ self.classes
        proto_class_idx = []     # length K_total, position of class in self.classes
        for ci, c in enumerate(self.classes):
            for _ in range(self.K_per_class[c]):
                proto_class.append(c)
                proto_class_idx.append(ci)
        self.K_total = len(proto_class)
        self.is_single_proto = all(K == 1 for K in self.K_per_class.values())

        # ---------- Buffers ----------
        protos = torch.randn(self.K_total, proj_dim)
        protos = F.normalize(protos, dim=1)
        self.register_buffer("prototypes", protos)
        self.register_buffer(
            "proto_class", torch.tensor(proto_class, dtype=torch.long)
        )
        self.register_buffer(
            "proto_class_idx", torch.tensor(proto_class_idx, dtype=torch.long)
        )
        # Running assignment counter (used by MP-DPCL for dead-prototype reset)
        self.register_buffer(
            "assign_count", torch.zeros(self.K_total, dtype=torch.long)
        )

        # ---------- Projection head ----------
        # Conv1x1 -> BN -> ReLU -> Conv1x1 (this IS the per-pixel MLP)
        hidden = max(proj_dim * 2, feat_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, proj_dim, kernel_size=1, bias=False),
        )

        # ---------- Hyper-parameters ----------
        self.proj_dim = int(proj_dim)
        self.samples_per_class = int(samples_per_class)
        self.warmup_iters = int(warmup_iters)
        self.ramp_iters = max(0, int(ramp_iters))
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.ortho_weight = float(ortho_weight) if not self.is_single_proto else 0.0

        # Class loss weights — register as buffer for AMP compatibility
        cw = _normalize_class_weight_dict(
            class_loss_weights, classes=self.classes, default=1.0
        )
        weight_tensor = torch.tensor(
            [cw[c] for c in self.classes], dtype=torch.float32
        )
        self.register_buffer("class_loss_weights", weight_tensor)

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    def effective_weight(self, current_iter: int, base_weight: float) -> float:
        """
        Schedule for the DPCL loss weight.

        Phase 1 [0, warmup_iters):          0           (prototypes learn, no loss)
        Phase 2 [warmup, warmup+ramp):      0 → base    (linear ramp)
        Phase 3 [warmup+ramp, ∞):           base
        """
        if current_iter < self.warmup_iters:
            return 0.0
        if self.ramp_iters <= 0:
            return float(base_weight)
        ramp = (current_iter - self.warmup_iters) / float(self.ramp_iters)
        ramp = max(0.0, min(1.0, ramp))
        return float(base_weight) * ramp

    def is_in_warmup(self, current_iter: int) -> bool:
        """True if we are still in the prototype-only warmup phase."""
        return current_iter < self.warmup_iters

    # ------------------------------------------------------------------
    # Internal: sampling, prototype update, loss
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _resize_labels(self, labels: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Downsample labels to feature resolution with nearest neighbour."""
        if labels.shape[-2:] == (H, W):
            return labels
        labels_f = labels.float().unsqueeze(1)
        labels_d = F.interpolate(labels_f, size=(H, W), mode="nearest")
        return labels_d.squeeze(1).long()

    @torch.no_grad()
    def _sample_indices(self, labels_lr: torch.Tensor):
        """
        Class-balanced pixel index sampling on the flattened (B*H*W) axis.

        Returns:
            flat_idx:        (N,) long, indices into the flattened pixel axis
            class_idx:       (N,) long, indices into self.classes (0..|classes|-1)
        """
        labels_flat = labels_lr.reshape(-1)
        idx_lists = []
        cls_lists = []
        for ci, c in enumerate(self.classes):
            mask = (labels_flat == c)
            n = int(mask.sum().item())
            if n == 0:
                continue
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if n > self.samples_per_class:
                perm = torch.randperm(n, device=idx.device)[: self.samples_per_class]
                idx = idx[perm]
            idx_lists.append(idx)
            cls_lists.append(
                torch.full((idx.numel(),), ci, dtype=torch.long, device=idx.device)
            )

        if len(idx_lists) == 0:
            return None, None
        return torch.cat(idx_lists, dim=0), torch.cat(cls_lists, dim=0)

    @torch.no_grad()
    def _update_prototypes(
        self,
        sampled_feats: torch.Tensor,
        sampled_class_idx: torch.Tensor,
    ):
        """
        EMA prototype update.

        SP (K=1 per class): mean of class features → EMA into the single proto.
        MP (K>1 for some class): hard-nearest same-class assignment first, then
        per-prototype EMA on its assigned subset.
        """
        if sampled_feats.numel() == 0:
            return

        feats_fp32 = sampled_feats.detach().float()                 # (N, D)

        if self.is_single_proto:
            for ci, _ in enumerate(self.classes):
                mask = (sampled_class_idx == ci)
                if not mask.any():
                    continue
                feats_c = feats_fp32[mask]                           # (n_c, D)
                mean_c = feats_c.mean(dim=0)
                k = (self.proto_class_idx == ci).nonzero(as_tuple=True)[0].item()
                new_proto = (
                    self.momentum * self.prototypes[k]
                    + (1.0 - self.momentum) * mean_c
                )
                self.prototypes[k] = F.normalize(new_proto, dim=0)
                self.assign_count[k] += int(mask.sum().item())
            return

        # MP path: hard-nearest assignment within same class
        for ci, _ in enumerate(self.classes):
            mask = (sampled_class_idx == ci)
            if not mask.any():
                continue
            feats_c = feats_fp32[mask]                               # (n_c, D)
            ks = (self.proto_class_idx == ci).nonzero(as_tuple=True)[0]
            sub_protos = self.prototypes[ks]                         # (K_c, D)
            sim = feats_c @ sub_protos.t()                           # (n_c, K_c)
            assign = sim.argmax(dim=1)                               # (n_c,)
            for j, k in enumerate(ks.tolist()):
                m = (assign == j)
                if not m.any():
                    continue
                mean_kj = feats_c[m].mean(dim=0)
                new_proto = (
                    self.momentum * self.prototypes[k]
                    + (1.0 - self.momentum) * mean_kj
                )
                self.prototypes[k] = F.normalize(new_proto, dim=0)
                self.assign_count[k] += int(m.sum().item())

    def _infonce_loss(
        self,
        sampled_feats: torch.Tensor,
        sampled_class_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE / softmax cross-entropy against prototypes.

        SP path: each pixel has exactly one positive prototype (its class proto).
        MP path: each pixel has K_c positives (all same-class sub-protos),
        combined via logsumexp (soft-positive InfoNCE).
        """
        # Similarities: (N, K_total). Compute in fp32 for numerical stability.
        sim = sampled_feats.float() @ self.prototypes.float().t()
        sim = sim / max(self.temperature, 1e-6)

        proto_class_idx = self.proto_class_idx                       # (K_total,)

        if self.is_single_proto:
            pos_k = torch.empty(
                len(self.classes), dtype=torch.long, device=sampled_feats.device
            )
            for ci in range(len(self.classes)):
                pos_k[ci] = (proto_class_idx == ci).nonzero(as_tuple=True)[0].item()
            target = pos_k[sampled_class_idx]                        # (N,)
            weights = self.class_loss_weights[sampled_class_idx]     # (N,)
            ce = F.cross_entropy(sim, target, reduction="none")      # (N,)
            return (ce * weights).sum() / weights.sum().clamp_min(1.0)

        # MP path: logsumexp soft-positive InfoNCE
        all_log_norm = torch.logsumexp(sim, dim=1)                   # (N,)
        pos_mask = (proto_class_idx.unsqueeze(0) == sampled_class_idx.unsqueeze(1))
        sim_pos = sim.masked_fill(~pos_mask, float("-inf"))
        pos_log_num = torch.logsumexp(sim_pos, dim=1)                # (N,)
        per_sample = -(pos_log_num - all_log_norm)
        weights = self.class_loss_weights[sampled_class_idx]
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    def _ortho_loss(self) -> torch.Tensor:
        """
        Orthogonality regulariser between sub-prototypes of the same class.

        For each class with K_c > 1, penalise pairwise dot products
        between its sub-prototypes (off-diagonal of K_c × K_c Gram matrix).
        Returns 0 for SP-DPCL.
        """
        if self.ortho_weight <= 0.0 or self.is_single_proto:
            return self.prototypes.new_tensor(0.0)

        loss = self.prototypes.new_tensor(0.0)
        n_terms = 0
        for ci in range(len(self.classes)):
            ks = (self.proto_class_idx == ci).nonzero(as_tuple=True)[0]
            if ks.numel() < 2:
                continue
            sub = self.prototypes[ks].float()                        # (K_c, D)
            gram = sub @ sub.t()                                     # (K_c, K_c)
            off_diag = gram - torch.eye(
                gram.size(0), device=gram.device, dtype=gram.dtype
            )
            loss = loss + (off_diag ** 2).sum()
            n_terms += ks.numel() * (ks.numel() - 1)
        if n_terms > 0:
            loss = loss / float(n_terms)
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        current_iter: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            feats:        (B, C_in, H_feat, W_feat) decoder mid-layer features
                          (e.g., dec3 at 160x160, 256 ch for crop=640 UNet).
            labels:       (B, H_label, W_label) damage labels at original or
                          feature resolution. 255 = ignore.
            current_iter: training iteration index. Used by the trainer to
                          decide whether to skip applying this loss.

        Returns:
            scalar loss tensor. Caller is responsible for applying the
            `effective_weight(current_iter, base_weight)` schedule.
        """
        # 1) Project + L2-normalise features (gradient flows here).
        # Under AMP the backbone feature may be fp16 while the projection head
        # stays fp32 because DPCL is computed outside autocast. Cast the input
        # explicitly so Conv2d sees matching dtypes.
        feat_proj = self.proj(feats.float())                         # (B, D, H, W)
        feat_proj = F.normalize(feat_proj, dim=1)

        # 2) Resize labels to feature resolution.
        H, W = feat_proj.shape[-2:]
        labels_lr = self._resize_labels(labels, H, W)                # (B, H, W)

        # 3) Class-balanced sampling — get flat indices ONCE, share across
        #    prototype-update path and gradient path.
        flat_idx, class_idx = self._sample_indices(labels_lr)
        if flat_idx is None or flat_idx.numel() == 0:
            return feat_proj.new_tensor(0.0)

        # 4) Gather features at sampled positions (single source of truth)
        B, D = feat_proj.shape[0], feat_proj.shape[1]
        feats_flat = feat_proj.permute(0, 2, 3, 1).reshape(-1, D)    # (B*H*W, D)
        sampled_feats = feats_flat[flat_idx]                         # (N, D), grad ON

        # 5) Update prototypes via EMA (no-grad path on the same samples)
        self._update_prototypes(sampled_feats, class_idx)

        # 6) InfoNCE / soft-positive InfoNCE loss
        info_loss = self._infonce_loss(sampled_feats, class_idx)

        # 7) Optional orthogonality regulariser (MP only)
        ortho = self._ortho_loss() * self.ortho_weight

        return info_loss + ortho
