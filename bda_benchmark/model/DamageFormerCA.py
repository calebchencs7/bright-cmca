"""DamageFormer with standard CA for the CMCA control experiment."""

from torchvision.models import Swin_T_Weights

from model.DamageFormerCMCA import DamageFormerCMCA
from model.UNetCMCA import CrossModalAttention


class DamageFormerCA(DamageFormerCMCA):
    """Replace only CMCA after pyramid fusion with standard cross-attention."""

    def __init__(
        self,
        num_classes=4,
        ca_heads=4,
        ca_sr_ratio=16,
        weights=Swin_T_Weights.DEFAULT,
    ):
        super().__init__(
            num_classes=num_classes,
            cmca_heads=ca_heads,
            cmca_sr_ratio=ca_sr_ratio,
            weights=weights,
        )
        self.cmca = CrossModalAttention(
            dim=256,
            num_heads=ca_heads,
            sr_ratio=ca_sr_ratio,
        )
