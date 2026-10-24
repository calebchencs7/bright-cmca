"""SiamCRNN with standard CA for the CMCA control experiment."""

from model.SiamCRNNCMCA import SiamCRNNCMCA
from model.UNetCMCA import CrossModalAttention


class SiamCRNNCA(SiamCRNNCMCA):
    """Replace only the deepest CMCA block and keep ConvLSTM fusion intact."""

    def __init__(self, num_classes=4, ca_heads=8, ca_sr_ratio=1, pretrained=True):
        super().__init__(
            num_classes=num_classes,
            cmca_heads=ca_heads,
            cmca_sr_ratio=ca_sr_ratio,
            pretrained=pretrained,
        )
        self.cmca_4 = CrossModalAttention(
            dim=512,
            num_heads=ca_heads,
            sr_ratio=ca_sr_ratio,
        )
