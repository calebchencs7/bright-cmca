"""
DamageFormer-CMCA.

This variant keeps the original dual Swin encoders and pyramid fusion, then
adds Cross-Modal Change Attention between the pre-event and post-event fused
feature maps before the damage head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t

from model.DamageFormer import ResBlock, SwinTransformerFeatureExtractor
from model.UNetCMCA import CrossModalChangeAttention


class DamageFormerCMCA(nn.Module):
    def __init__(
        self,
        num_classes=4,
        cmca_heads=4,
        cmca_sr_ratio=16,
        weights=Swin_T_Weights.DEFAULT,
    ):
        super().__init__()

        self.encoder_1 = SwinTransformerFeatureExtractor(
            swin_transformer=swin_t(weights=weights),
            depths=[2, 2, 6, 2],
        )
        self.encoder_2 = SwinTransformerFeatureExtractor(
            swin_transformer=swin_t(weights=weights),
            depths=[2, 2, 6, 2],
        )

        self.fusion_layer_1 = ResBlock(
            in_channels=1440,
            out_channels=256,
            stride=1,
            downsample=nn.Conv2d(1440, 256, kernel_size=1),
        )
        self.fusion_layer_2 = ResBlock(
            in_channels=1440,
            out_channels=256,
            stride=1,
            downsample=nn.Conv2d(1440, 256, kernel_size=1),
        )

        # The fused Swin stage-1 maps are relatively large. A high sr_ratio
        # keeps attention memory bounded while still providing global alignment.
        self.cmca = CrossModalChangeAttention(
            dim=256,
            num_heads=cmca_heads,
            sr_ratio=cmca_sr_ratio,
        )
        self.fusion_layer_3 = ResBlock(
            in_channels=256 * 3,
            out_channels=256,
            stride=1,
            downsample=nn.Conv2d(256 * 3, 256, kernel_size=1),
        )

        self.clf_1 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
        self.clf_2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)

    @staticmethod
    def _split_inputs(x, post=None):
        if post is not None:
            return x, post, True
        mid = x.shape[1] // 2
        return x[:, :mid], x[:, mid:], False

    @staticmethod
    def _to_nchw(feat):
        return feat.permute(0, 3, 1, 2)

    def _pyramid_feature(self, low1, low2, low3, output):
        output = F.interpolate(output, size=low1.size()[2:], mode="bilinear")
        low3 = F.interpolate(low3, size=low1.size()[2:], mode="bilinear")
        low2 = F.interpolate(low2, size=low1.size()[2:], mode="bilinear")
        return torch.cat([output, low3, low2, low1], dim=1)

    def forward(self, x, post=None):
        pre_data, post_data, return_loc = self._split_inputs(x, post)

        _, pre_low1, pre_low2, pre_low3, pre_output = self.encoder_1(pre_data)
        _, post_low1, post_low2, post_low3, post_output = self.encoder_2(post_data)

        pre_low1 = self._to_nchw(pre_low1)
        post_low1 = self._to_nchw(post_low1)
        pre_low2 = self._to_nchw(pre_low2)
        post_low2 = self._to_nchw(post_low2)
        pre_low3 = self._to_nchw(pre_low3)
        post_low3 = self._to_nchw(post_low3)
        pre_output = self._to_nchw(pre_output)
        post_output = self._to_nchw(post_output)

        pre_feat = self._pyramid_feature(pre_low1, pre_low2, pre_low3, pre_output)
        post_feat = self._pyramid_feature(post_low1, post_low2, post_low3, post_output)

        pre_fused = self.fusion_layer_1(pre_feat)
        post_fused = self.fusion_layer_2(post_feat)
        change = self.cmca(pre_fused, post_fused)

        damage_feat = self.fusion_layer_3(torch.cat([pre_fused, post_fused, change], dim=1))

        output_loc = self.clf_1(pre_fused)
        output_loc = F.interpolate(output_loc, size=pre_data.size()[-2:], mode="bilinear")
        output_dam = self.clf_2(damage_feat)
        output_dam = F.interpolate(output_dam, size=post_data.size()[-2:], mode="bilinear")

        if return_loc:
            return output_loc, output_dam
        return output_dam
