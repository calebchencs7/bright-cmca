"""
DeepLabV3Plus-CMCA.

The baseline DeepLabV3Plus consumes the concatenated 6-channel input with one
ResNet-50 backbone. This variant uses two modality-specific ResNet-50 branches,
adds Cross-Modal Change Attention at the deepest semantic feature level, then
keeps the standard ASPP + low-level decoder design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.DeepLabV3Plus import ASPP
from model.UNetCMCA import CrossModalChangeAttention


def _resnet50(pretrained=True):
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    except AttributeError:
        return models.resnet50(pretrained=pretrained)


class DeepLabV3PlusCMCA(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=4,
        atrous_rates=[6, 12, 18],
        output_stride=16,
        cmca_heads=8,
        cmca_sr_ratio=2,
        pretrained=True,
    ):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("DeepLabV3PlusCMCA expects paired pre/post channels.")

        branch_channels = in_channels // 2
        self.backbone_pre = _resnet50(pretrained=pretrained)
        self.backbone_post = _resnet50(pretrained=pretrained)

        if branch_channels != 3:
            self.backbone_pre.conv1 = nn.Conv2d(
                branch_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.backbone_post.conv1 = nn.Conv2d(
                branch_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        low_level_channels = 256
        high_level_channels = 2048

        if output_stride == 16:
            for backbone in (self.backbone_pre, self.backbone_post):
                backbone.layer4[0].conv2.stride = (1, 1)
                backbone.layer4[0].downsample[0].stride = (1, 1)
        else:
            raise NotImplementedError

        self.low_level_fuse = nn.Sequential(
            nn.Conv2d(low_level_channels * 2, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.cmca = CrossModalChangeAttention(
            dim=high_level_channels,
            num_heads=cmca_heads,
            sr_ratio=cmca_sr_ratio,
        )
        self.high_level_fuse = nn.Sequential(
            nn.Conv2d(high_level_channels * 3, high_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_level_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(high_level_channels, 256, atrous_rates)

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    @staticmethod
    def _split_inputs(x, post=None):
        if post is not None:
            return x, post
        mid = x.shape[1] // 2
        return x[:, :mid], x[:, mid:]

    @staticmethod
    def _forward_backbone(backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        low = backbone.layer1(x)
        x = backbone.layer2(low)
        x = backbone.layer3(x)
        high = backbone.layer4(x)
        return low, high

    def forward(self, x, post=None):
        size = x.shape[2:]
        pre, post = self._split_inputs(x, post)

        pre_low, pre_high = self._forward_backbone(self.backbone_pre, pre)
        post_low, post_high = self._forward_backbone(self.backbone_post, post)

        low_level_features = self.low_level_fuse(torch.cat([pre_low, post_low], dim=1))
        low_level_features = self.low_level_conv(low_level_features)

        change = self.cmca(pre_high, post_high)
        high_level_features = self.high_level_fuse(
            torch.cat([pre_high, post_high, change], dim=1)
        )
        high_level_features = self.aspp(high_level_features)
        high_level_features = F.interpolate(
            high_level_features,
            size=low_level_features.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.decoder(torch.cat((low_level_features, high_level_features), dim=1))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
