"""
SAM-Guided Refinement Module (SGR)
===================================
A lightweight, model-agnostic post-processing module that uses SAM building
masks to refine damage classification logits from ANY segmentation backbone.

Design principles:
    1. Zero intrusion: does NOT modify the backbone's forward() or input channels.
    2. Pluggable: one line to add, one line to remove.
    3. Lightweight: <50K parameters, negligible overhead.
    4. Universal: works with UNet, DeepLabV3+, SiamAttnUNet, DamageFormer, etc.

Architecture:
    Given backbone logits (B, C, H, W) and SAM mask (B, 1, H, W):
    1. Boundary-Aware Context Extraction
       - Derive boundary signal from SAM mask via Laplacian (fixed kernel, no params).
       - Concat [logits, sam_mask, boundary] -> (B, C+2, H, W).
    2. Lightweight Residual Refinement
       - Two depthwise-separable conv blocks with residual connection.
       - Channel-wise SE attention conditioned on SAM mask.
    3. Output: refined_logits = logits + alpha * residual
       - alpha is a learnable scalar initialized near 0 for training stability.

Usage:
    >>> from model.SAMGuidedRefinement import SAMGuidedRefinement
    >>> backbone = UNet(in_channels=6, num_classes=4)
    >>> refiner = SAMGuidedRefinement(num_classes=4)
    >>>
    >>> logits = backbone(x)                        # (B, 4, H, W)
    >>> refined = refiner(logits, sam_mask)          # (B, 4, H, W)
    >>> loss = criterion(refined, labels)
    >>>
    >>> # To ablate: simply remove the refiner call.
    >>> # To transfer: use the same refiner with DeepLabV3+, SiamAttnUNet, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise-separable convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class ChannelSEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.excitation(self.squeeze(x))
        return x * w


class SAMGuidedRefinement(nn.Module):
    """
    Model-agnostic SAM-guided refinement module.

    Args:
        num_classes (int): Number of segmentation classes (must match backbone output).
        hidden_dim (int): Hidden channels in refinement convs. Default 32.
        alpha_init (float): Initial value of learnable residual scale. Default 0.1.
            Starting small ensures the refiner does not destabilize early training.
    """

    def __init__(self, num_classes=4, hidden_dim=32, alpha_init=0.1):
        super().__init__()
        self.num_classes = num_classes

        # Fixed Laplacian kernel for boundary extraction (no learnable params).
        laplacian = torch.tensor(
            [[0., 1., 0.],
             [1., -4., 1.],
             [0., 1., 0.]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian)

        # Input: [logits(C) + sam_mask(1) + boundary(1)] = C + 2
        in_ch = num_classes + 2

        # Lightweight refinement pathway
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(in_ch, hidden_dim, kernel_size=3, padding=1),
            DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ChannelSEBlock(hidden_dim, reduction=4),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True)
        )

        # Learnable residual scale, initialized small for stability.
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self._init_weights()

    def _init_weights(self):
        """Initialize refinement head near identity (small residual at start)."""
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the final 1x1 conv so initial residual is ~0.
        final_conv = self.refine[-1]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def _extract_boundary(self, sam_mask):
        """Extract building boundary signal from SAM mask using fixed Laplacian."""
        # sam_mask: (B, 1, H, W), values in [0, 1]
        boundary = F.conv2d(sam_mask, self.laplacian_kernel, padding=1)
        boundary = torch.abs(boundary)
        # Normalize to [0, 1] per sample
        bmax = boundary.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return boundary / bmax

    def forward(self, logits, sam_mask):
        """
        Refine backbone logits using SAM building mask.

        Args:
            logits: (B, num_classes, H, W) raw backbone output.
            sam_mask: (B, 1, H, W) SAM building probability mask, values in [0, 1].

        Returns:
            refined_logits: (B, num_classes, H, W)
        """
        # Ensure spatial dimensions match
        if sam_mask.shape[2:] != logits.shape[2:]:
            sam_mask = F.interpolate(
                sam_mask, size=logits.shape[2:],
                mode='bilinear', align_corners=False
            )

        # 1. Extract boundary context
        boundary = self._extract_boundary(sam_mask)

        # 2. Concatenate context: [logits, sam_mask, boundary]
        context = torch.cat([logits.detach() if not self.training else logits,
                             sam_mask, boundary], dim=1)

        # 3. Lightweight refinement with residual connection
        residual = self.refine(context)
        refined_logits = logits + self.alpha * residual

        return refined_logits

    def extra_repr(self):
        return f'num_classes={self.num_classes}, alpha_init={self.alpha.item():.4f}'
