"""DeepLabV3Plus with standard cross-attention for the CMCA ablation."""

from model.DeepLabV3PlusCMCA import DeepLabV3PlusCMCA
from model.UNetCMCA import CrossModalAttention


class DeepLabV3PlusCA(DeepLabV3PlusCMCA):
    """Keep the dual-branch CMCA model and replace only its fusion module."""

    def __init__(
        self,
        in_channels=6,
        num_classes=4,
        atrous_rates=[6, 12, 18],
        output_stride=16,
        ca_heads=8,
        ca_sr_ratio=2,
        pretrained=True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            atrous_rates=atrous_rates,
            output_stride=output_stride,
            cmca_heads=ca_heads,
            cmca_sr_ratio=ca_sr_ratio,
            pretrained=pretrained,
        )
        self.cmca = CrossModalAttention(
            dim=2048,
            num_heads=ca_heads,
            sr_ratio=ca_sr_ratio,
            force_fp32=True,
        )
