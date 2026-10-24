"""DamageFormer with CMCA on the deepest Swin features before pyramid fusion."""

import torch
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights

from model.DamageFormer import DamageFormer
from model.UNetCMCA import CrossModalChangeAttention


class DamageFormerCMCADeep(DamageFormer):
    def __init__(
        self,
        num_classes=4,
        cmca_heads=8,
        cmca_sr_ratio=1,
        weights=Swin_T_Weights.DEFAULT,
    ):
        super().__init__(num_classes=num_classes, weights=weights)
        self.cmca = CrossModalChangeAttention(
            dim=768, num_heads=cmca_heads, sr_ratio=cmca_sr_ratio
        )

    @staticmethod
    def _pyramid_feature(features):
        low1, low2, low3, deepest = features
        size = low1.shape[-2:]
        return torch.cat([
            F.interpolate(deepest, size=size, mode="bilinear"),
            F.interpolate(low3, size=size, mode="bilinear"),
            F.interpolate(low2, size=size, mode="bilinear"),
            low1,
        ], dim=1)

    def forward(self, x, post=None, return_loc=False):
        pre_data, post_data, paired = self._split_inputs(x, post)
        pre_features = [
            feat.permute(0, 3, 1, 2) for feat in self.encoder_1(pre_data)[1:]
        ]
        post_features = [
            feat.permute(0, 3, 1, 2) for feat in self.encoder_2(post_data)[1:]
        ]

        # Keep shallow features and the original 512 -> 256 fusion intact.
        change = self.cmca(pre_features[-1], post_features[-1])
        post_features[-1] = post_features[-1] + change
        pre_fused = self.fusion_layer_1(self._pyramid_feature(pre_features))
        post_fused = self.fusion_layer_2(self._pyramid_feature(post_features))
        damage_feat = self.fusion_layer_3(
            torch.cat([pre_fused, post_fused], dim=1)
        )

        output_dam = F.interpolate(
            self.clf_2(damage_feat), size=post_data.shape[-2:], mode="bilinear"
        )
        if paired or return_loc:
            output_loc = F.interpolate(
                self.clf_1(pre_fused), size=pre_data.shape[-2:], mode="bilinear"
            )
            return output_loc, output_dam
        return output_dam
