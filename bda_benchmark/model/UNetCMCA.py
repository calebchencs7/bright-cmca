"""
UNet-CMCA: UNet with Cross-Modal Change Attention.

Architecture overview
=====================
  Pre-event optical (3ch)  ─── Optical Encoder ───┐
                                                   ├── CMCA ── Bottleneck ── Decoder ── Output
  Post-event SAR   (3ch)  ─── SAR Encoder    ───┘

Key difference from vanilla UNet:
  1. Dual-branch encoder: separate pathways preserve modality-specific
     features (optical = building structure, SAR = post-disaster state).
  2. Cross-Modal Change Attention (CMCA) at the bottleneck: optical
     features query SAR features via cross-attention to highlight WHERE
     and HOW MUCH structural change has occurred.
  3. Shared decoder with fused skip connections.

Motivation (from BRIGHT paper, Table 8):
  Optical-only DamageFormer mIoU = 69.76%, Optical+SAR = 70.79%.
  The mere +1% gap shows current concat-based fusion fails to exploit
  SAR's structural change information. CMCA addresses this by learning
  explicit cross-modal correspondences.

Interface:
  Same as UNet — forward(x) where x is (B, 6, H, W).
  Internally splits into optical[:3] and SAR[3:].
  Drop-in replacement: change --model_type UNet to UNetCMCA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.UNet import ConvBlock


# --------------------------------------------------------------------------
# Cross-Modal Change Attention (CMCA)
# --------------------------------------------------------------------------

class CrossModalChangeAttention(nn.Module):
    """
    CMCA — the core innovation module.

    Given bottleneck features from optical encoder and SAR encoder:
      - Q from optical:  "for each position, what building structure exists?"
      - K/V from SAR:    "what does the post-disaster scene look like here?"

    Cross-attention aligns SAR features to optical positions, then a learned
    projection extracts the change signal by comparing the aligned features.

    Uses spatial reduction on K/V for memory efficiency (PVT-style).
    At bottleneck resolution 40×40 (crop_size=640), attention cost is modest.

    Args:
        dim:       channel dimension of input features (e.g. 512)
        num_heads: number of attention heads
        sr_ratio:  spatial reduction ratio for K/V (2 = halve each side)
    """

    def __init__(self, dim, num_heads=4, sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)

        # Spatial reduction for K/V (depthwise conv + BN)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim),
                nn.BatchNorm2d(dim),
            )

        # Learned change extraction: [aligned_sar ; optical] → change
        self.change_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, opt_feat, sar_feat):
        """
        Args:
            opt_feat: (B, C, H, W) optical encoder bottleneck features
            sar_feat: (B, C, H, W) SAR encoder bottleneck features
        Returns:
            change:   (B, C, H, W) change-enhanced features
        """
        B, C, H, W = opt_feat.shape
        h, d = self.num_heads, self.head_dim

        # Q from optical (full resolution)
        Q = self.q_proj(opt_feat)
        Q = Q.reshape(B, h, d, H * W).permute(0, 1, 3, 2)       # (B, h, N, d)

        # K, V from SAR (spatially reduced for efficiency)
        sar_kv = self.sr(sar_feat) if self.sr_ratio > 1 else sar_feat
        Hs, Ws = sar_kv.shape[2], sar_kv.shape[3]
        Ns = Hs * Ws

        K = self.k_proj(sar_kv).reshape(B, h, d, Ns).permute(0, 1, 3, 2)  # (B, h, Ns, d)
        V = self.v_proj(sar_kv).reshape(B, h, d, Ns).permute(0, 1, 3, 2)  # (B, h, Ns, d)

        # Scaled dot-product cross-attention
        attn = (Q @ K.transpose(-1, -2)) * self.scale              # (B, h, N, Ns)
        attn = F.softmax(attn, dim=-1)

        aligned_sar = (attn @ V)                                   # (B, h, N, d)
        aligned_sar = aligned_sar.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Change = learned comparison between aligned SAR and optical
        # High response = large structural difference = severe damage
        change = self.change_proj(torch.cat([aligned_sar, opt_feat], dim=1))
        return change


# --------------------------------------------------------------------------
# UNet-CMCA model
# --------------------------------------------------------------------------

class UNetCMCA(nn.Module):
    """
    Dual-branch UNet with Cross-Modal Change Attention.

    Encoder: two independent branches (optical / SAR), each 5 ConvBlocks.
    Bottleneck: CMCA fuses the two branches into a change-aware feature.
    Decoder: standard UNet decoder (identical structure to vanilla UNet).
    Skip connections: optical + SAR features fused via 1×1 conv at each level.

    Parameter overhead vs vanilla UNet:
      - Encoder: each branch ends at 512ch (vs single branch at 1024ch),
        so total encoder params are comparable (~18M vs ~19M).
      - CMCA + fusion layers add ~4M params.
      - Decoder: identical (~18M).
      - Total: ~41M vs ~37M (≈+11%).

    Args:
        in_channels: total input channels (default 6 = 3 optical + 3 SAR)
        num_classes: output classes (default 4 = bg/intact/damaged/destroyed)
        num_heads:   attention heads in CMCA (default 4)
        sr_ratio:    spatial reduction ratio for CMCA K/V (default 2)
    """

    def __init__(self, in_channels=6, num_classes=4, num_heads=4, sr_ratio=2,
                 use_checkpoint=True):
        super().__init__()
        ch = in_channels // 2   # channels per modality
        self.use_checkpoint = use_checkpoint

        # ---- Optical encoder (pre-event) ----
        self.opt_enc1 = ConvBlock(ch, 64)
        self.opt_enc2 = ConvBlock(64, 128)
        self.opt_enc3 = ConvBlock(128, 256)
        self.opt_enc4 = ConvBlock(256, 512)
        self.opt_enc5 = ConvBlock(512, 512)

        # ---- SAR encoder (post-event) ----
        self.sar_enc1 = ConvBlock(ch, 64)
        self.sar_enc2 = ConvBlock(64, 128)
        self.sar_enc3 = ConvBlock(128, 256)
        self.sar_enc4 = ConvBlock(256, 512)
        self.sar_enc5 = ConvBlock(512, 512)

        # ---- Cross-Modal Change Attention at bottleneck ----
        self.cmca = CrossModalChangeAttention(512, num_heads, sr_ratio)

        # ---- Bottleneck fusion: opt(512) + sar(512) + change(512) → 1024 ----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 * 3, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # ---- Skip connection fusion: concat → 1×1 conv ----
        self.skip_fuse4 = nn.Conv2d(512 * 2, 512, 1)
        self.skip_fuse3 = nn.Conv2d(256 * 2, 256, 1)
        self.skip_fuse2 = nn.Conv2d(128 * 2, 128, 1)
        self.skip_fuse1 = nn.Conv2d(64 * 2, 64, 1)

        # ---- Decoder (same structure as vanilla UNet) ----
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec4 = ConvBlock(512 * 2, 512)
        self.dec3 = ConvBlock(256 * 2, 256)
        self.dec2 = ConvBlock(128 * 2, 128)
        self.dec1 = ConvBlock(64 * 2, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _ckpt(self, module, *inputs):
        """Run module with gradient checkpointing if enabled (saves ~40% peak memory)."""
        if self.use_checkpoint and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 6, H, W) — concatenated [pre_optical, post_SAR].
            return_features: if True, also return the d3 feature map
                (160x160, 256 ch for crop=640). Used by DPCL and other
                auxiliary losses that need decoder mid-layer features.
                Default False keeps the module signature backward-compatible.

        Returns:
            logits: (B, num_classes, H, W)  always
            d3:     (B, 256, H/4, W/4)      only if return_features=True
        """
        mid = x.shape[1] // 2
        opt = x[:, :mid]     # pre-event optical
        sar = x[:, mid:]     # post-event SAR

        # ---- Dual-branch encoding (checkpointed to save memory) ----
        o1 = self._ckpt(self.opt_enc1, opt);       s1 = self._ckpt(self.sar_enc1, sar)
        o2 = self._ckpt(self.opt_enc2, self.pool(o1));  s2 = self._ckpt(self.sar_enc2, self.pool(s1))
        o3 = self._ckpt(self.opt_enc3, self.pool(o2));  s3 = self._ckpt(self.sar_enc3, self.pool(s2))
        o4 = self._ckpt(self.opt_enc4, self.pool(o3));  s4 = self._ckpt(self.sar_enc4, self.pool(s3))
        o5 = self._ckpt(self.opt_enc5, self.pool(o4));  s5 = self._ckpt(self.sar_enc5, self.pool(s4))

        # ---- Cross-Modal Change Attention at bottleneck ----
        change = self.cmca(o5, s5)

        # ---- Bottleneck fusion ----
        neck = self.bottleneck(torch.cat([o5, s5, change], dim=1))

        # ---- Decoder with fused skip connections (also checkpointed) ----
        d4 = self._ckpt(self.dec4, torch.cat([self.upconv4(neck),
                         self.skip_fuse4(torch.cat([o4, s4], dim=1))], dim=1))

        d3 = self._ckpt(self.dec3, torch.cat([self.upconv3(d4),
                         self.skip_fuse3(torch.cat([o3, s3], dim=1))], dim=1))

        d2 = self._ckpt(self.dec2, torch.cat([self.upconv2(d3),
                         self.skip_fuse2(torch.cat([o2, s2], dim=1))], dim=1))

        d1 = self._ckpt(self.dec1, torch.cat([self.upconv1(d2),
                         self.skip_fuse1(torch.cat([o1, s1], dim=1))], dim=1))

        out = self.final_conv(d1)

        if return_features:
            return out, d3
        return out
