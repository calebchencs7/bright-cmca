"""SiamAttnUNet with standard CA for the CMCA control experiment."""

from model.SiamAttnUNetCMCA import SiamAttnUNetCMCA
from model.UNetCMCA import CrossModalAttention


class SiamAttnUNetCA(SiamAttnUNetCMCA):
    """Replace only bottleneck CMCA while preserving the dual-branch model."""

    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        ca_heads=8,
        ca_sr_ratio=2,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            cmca_heads=ca_heads,
            cmca_sr_ratio=ca_sr_ratio,
        )
        self.cmca_5 = CrossModalAttention(
            dim=1024,
            num_heads=ca_heads,
            sr_ratio=ca_sr_ratio,
            force_fp32=True,
        )
