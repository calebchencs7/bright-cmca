"""
UNetWithFeatures: a thin DPCL-friendly wrapper around the BRIGHT baseline UNet.

Why this file exists
--------------------
The vanilla `UNet` defined in `model/UNet.py` is the reference implementation
shipped with the BRIGHT paper [Chen et al., ESSD 2025]. We deliberately keep
that file byte-equivalent to the official baseline so that:
    1. Reproducing BRIGHT baselines uses the *exact* upstream code.
    2. Future rebases against the upstream BRIGHT repo do not conflict with
       our auxiliary-loss extensions.
    3. Reviewers can verify that "baseline UNet" in our paper matches the
       upstream paper's UNet without checking diff history.

Auxiliary losses (DPCL, future BOCL, etc.) need access to a decoder mid-layer
feature (`dec3`, 256ch at H/4 × W/4) for their projection head. Rather than
modify the baseline UNet's `forward` signature or rely on PyTorch forward
hooks (which complicate AMP / gradient checkpointing / DDP debugging), we
expose this feature through a thin subclass that overrides only `forward`.

Design contract
---------------
- Inherits all parameters and buffers from `UNet`. State dicts are
  cross-compatible: a state dict saved from `UNet` loads cleanly into
  `UNetWithFeatures` and vice versa.
- `forward(x)` matches the parent's behaviour exactly: returns the logits.
- `forward(x, return_features=True)` additionally returns the dec3 feature
  used by DPCL: a `(B, 256, H/4, W/4)` tensor (e.g. 160×160 at crop=640).
- No new layers, no new parameters, no new buffers.

Paper-language template
-----------------------
> "We use the unmodified UNet from Chen et al. [BRIGHT, 2025] as our
> segmentation backbone for baseline experiments. For our proposed
> DPCL-augmented model, we wrap UNet in a thin subclass `UNetWithFeatures`
> that exposes the dec3 mid-decoder feature for the contrastive head, with
> no other architectural change."
"""

from __future__ import annotations

import torch

from model.UNet import UNet


class UNetWithFeatures(UNet):
    """
    DPCL/auxiliary-loss-friendly UNet.

    Identical to the BRIGHT-baseline `UNet` in every respect except that
    `forward` accepts a `return_features` flag. When set, it additionally
    returns the dec3 feature map (256ch at H/4 × W/4 resolution) so that
    auxiliary heads (e.g. DPCL's projection head) can attach without
    modifying the baseline file or using forward hooks.

    Usage:
        backbone = UNetWithFeatures(in_channels=6, num_classes=4)

        # Baseline-style call: returns just logits, behaviour unchanged
        logits = backbone(x)

        # DPCL-style call: also returns dec3 for the contrastive head
        logits, feat_dec3 = backbone(x, return_features=True)
    """

    def forward(self, x, return_features: bool = False):
        # ---- Encoder path (identical to parent) ----
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        # ---- Decoder path (identical to parent) ----
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        if return_features:
            return out, dec3
        return out
