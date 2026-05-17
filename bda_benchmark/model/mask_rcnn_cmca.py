"""Mask R-CNN with CMCA stem fusion for BRIGHT RGB+SAR instance mapping.

This file keeps the original baseline in `mask_rcnn.py` untouched.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CrossModalChangeAttention(nn.Module):
    """Cross-attention to align SAR evidence to optical structure."""

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 2) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.sr_ratio = int(sr_ratio)
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=self.sr_ratio,
                    stride=self.sr_ratio,
                    groups=dim,
                    bias=False,
                ),
                nn.BatchNorm2d(dim),
            )
        else:
            self.sr = nn.Identity()

        self.change_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = opt_feat.shape
        heads, head_dim = self.num_heads, self.head_dim

        q = self.q_proj(opt_feat).reshape(b, heads, head_dim, h * w).permute(0, 1, 3, 2)

        sar_reduced = self.sr(sar_feat)
        hs, ws = sar_reduced.shape[-2:]
        ns = hs * ws

        k = self.k_proj(sar_reduced).reshape(b, heads, head_dim, ns).permute(0, 1, 3, 2)
        v = self.v_proj(sar_reduced).reshape(b, heads, head_dim, ns).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        aligned_sar = attn @ v
        aligned_sar = aligned_sar.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.change_proj(torch.cat([aligned_sar, opt_feat], dim=1))


class CMCAResNetBody(nn.Module):
    """ResNet body wrapper: RGB/SAR stems + CMCA before layer1..4 outputs."""

    def __init__(
        self,
        body: nn.Module,
        cmca_num_heads: int = 4,
        cmca_sr_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.body = body
        self.return_layers = dict(body.return_layers)

        conv1 = body.conv1
        out_ch = conv1.out_channels

        self.sar_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_ch,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            dilation=conv1.dilation,
            groups=conv1.groups,
            bias=conv1.bias is not None,
            padding_mode=conv1.padding_mode,
        )

        with torch.no_grad():
            sar_w = conv1.weight.mean(dim=1, keepdim=True)
            self.sar_conv1.weight.copy_(sar_w)
            if conv1.bias is not None:
                self.sar_conv1.bias.copy_(conv1.bias)

        self.cmca = CrossModalChangeAttention(
            dim=out_ch, num_heads=cmca_num_heads, sr_ratio=cmca_sr_ratio
        )
        self.stem_fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        if x.shape[1] < 4:
            raise ValueError(
                f"CMCA Mask R-CNN expects at least 4 channels [R,G,B,SAR], got {x.shape[1]}."
            )

        x_rgb = x[:, :3]
        x_sar = x[:, 3:4]

        rgb_stem = self.body.conv1(x_rgb)
        sar_stem = self.sar_conv1(x_sar)
        change_stem = self.cmca(rgb_stem, sar_stem)

        x = self.stem_fuse(torch.cat([rgb_stem, sar_stem, change_stem], dim=1))
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        out = OrderedDict()
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self.body, layer_name)(x)
            if layer_name in self.return_layers:
                out_name = self.return_layers[layer_name]
                out[out_name] = x

        return out


def build_model_cmca(
    num_classes: int = 4,
    pretrained: bool = True,
    pixel_mean: list = None,
    pixel_std: list = None,
    box_detections_per_img: int = 1500,
    rpn_pre_nms_top_n_test: int = 1500,
    rpn_post_nms_top_n_test: int = 1500,
    cmca_num_heads: int = 4,
    cmca_sr_ratio: int = 2,
) -> nn.Module:
    """Build a 4-channel Mask R-CNN with CMCA stem fusion.

    Input channels are [R, G, B, SAR].
    """
    image_mean = list(pixel_mean) if pixel_mean is not None else [0.485, 0.456, 0.406, 0.5]
    image_std = list(pixel_std) if pixel_std is not None else [0.229, 0.224, 0.225, 0.25]
    if len(image_mean) != 4 or len(image_std) != 4:
        raise ValueError("pixel_mean and pixel_std must each have 4 values [R,G,B,SAR].")

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        image_mean=image_mean,
        image_std=image_std,
        box_detections_per_img=box_detections_per_img,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
    )

    # Replace detection heads to match challenge classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    old_mask_pred = model.roi_heads.mask_predictor
    in_features_mask = old_mask_pred.conv5_mask.in_channels
    dim_reduced = old_mask_pred.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced, num_classes
    )

    # Swap backbone body with CMCA-enhanced stem wrapper.
    model.backbone.body = CMCAResNetBody(
        body=model.backbone.body,
        cmca_num_heads=cmca_num_heads,
        cmca_sr_ratio=cmca_sr_ratio,
    )

    return model
