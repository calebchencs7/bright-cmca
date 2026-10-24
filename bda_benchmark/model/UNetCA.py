"""UNet with standard cross-attention for the CMCA control experiment."""

from model.UNetCMCA import CrossModalAttention, UNetCMCA


class UNetCA(UNetCMCA):
    """Keep the UNet-CMCA architecture but replace CMCA with standard CA."""

    def __init__(
        self,
        in_channels=6,
        num_classes=4,
        num_heads=4,
        sr_ratio=2,
        use_checkpoint=True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            use_checkpoint=use_checkpoint,
        )
        self.cmca = CrossModalAttention(
            dim=512,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )
