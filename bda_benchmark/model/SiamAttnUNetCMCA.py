"""
SiamAttnUNet-CMCA.

This keeps the original SiamAttnUNet decoder and skip-attention design, but
replaces the deepest pre/post fusion with Cross-Modal Change Attention. Shallow
skip features are still fused with the original lightweight 1x1 projections to
keep memory use close to the baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SiamAttnUNet import ChannelAttention, ConvBlock, UpConv
from model.UNetCMCA import CrossModalChangeAttention


class SiamStableCrossModalChangeAttention(CrossModalChangeAttention):
    """CMCA variant used only by SiamAttnUNet-CMCA for fp16 stability."""

    def forward(self, opt_feat, sar_feat):
        B, C, H, W = opt_feat.shape
        h, d = self.num_heads, self.head_dim

        Q = self.q_proj(opt_feat)
        Q = Q.reshape(B, h, d, H * W).permute(0, 1, 3, 2)

        sar_kv = self.sr(sar_feat) if self.sr_ratio > 1 else sar_feat
        Hs, Ws = sar_kv.shape[2], sar_kv.shape[3]
        Ns = Hs * Ws

        K = self.k_proj(sar_kv).reshape(B, h, d, Ns).permute(0, 1, 3, 2)
        V = self.v_proj(sar_kv).reshape(B, h, d, Ns).permute(0, 1, 3, 2)

        with torch.autocast(device_type=opt_feat.device.type, enabled=False):
            attn = (Q.float() @ K.float().transpose(-1, -2)) * self.scale
            attn = F.softmax(attn, dim=-1)
            aligned_sar = attn @ V.float()

        aligned_sar = aligned_sar.to(opt_feat.dtype)
        aligned_sar = aligned_sar.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.change_proj(torch.cat([aligned_sar, opt_feat], dim=1))


class SiamAttnUNetCMCA(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        cmca_heads=8,
        cmca_sr_ratio=2,
    ):
        super().__init__()

        # Pre-disaster encoder
        self.encoder1_pre = ConvBlock(in_channels, 64)
        self.encoder2_pre = ConvBlock(64, 128)
        self.encoder3_pre = ConvBlock(128, 256)
        self.encoder4_pre = ConvBlock(256, 512)
        self.encoder5_pre = ConvBlock(512, 1024)

        # Post-disaster encoder
        self.encoder1_post = ConvBlock(in_channels, 64)
        self.encoder2_post = ConvBlock(64, 128)
        self.encoder3_post = ConvBlock(128, 256)
        self.encoder4_post = ConvBlock(256, 512)
        self.encoder5_post = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fuse_1 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        self.fuse_2 = nn.Conv2d(128 * 2, 128, kernel_size=1)
        self.fuse_3 = nn.Conv2d(256 * 2, 256, kernel_size=1)
        self.fuse_4 = nn.Conv2d(512 * 2, 512, kernel_size=1)

        # CMCA at bottleneck resolution (crop 640 -> 40x40), where full
        # cross-modal attention is expressive without the cost of shallow maps.
        self.cmca_5 = SiamStableCrossModalChangeAttention(
            dim=1024,
            num_heads=cmca_heads,
            sr_ratio=cmca_sr_ratio,
        )

        self.attention1 = ChannelAttention(512)
        self.attention2 = ChannelAttention(256)
        self.attention3 = ChannelAttention(128)
        self.attention4 = ChannelAttention(64)

        self.upconv1 = UpConv(1024, 512)
        self.upconv2 = UpConv(512 * 2, 256)
        self.upconv3 = UpConv(256 * 2, 128)
        self.upconv4 = UpConv(128 * 2, 64)
        self.upconv5 = ConvBlock(64 * 2, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _split_inputs(x, post=None):
        if post is not None:
            return x, post
        mid = x.shape[1] // 2
        return x[:, :mid], x[:, mid:]

    def forward(self, x, post=None):
        x1, x2 = self._split_inputs(x, post)

        enc1_1 = self.encoder1_pre(x1)
        enc2_1 = self.encoder2_pre(self.pool(enc1_1))
        enc3_1 = self.encoder3_pre(self.pool(enc2_1))
        enc4_1 = self.encoder4_pre(self.pool(enc3_1))
        enc5_1 = self.encoder5_pre(self.pool(enc4_1))

        enc1_2 = self.encoder1_post(x2)
        enc2_2 = self.encoder2_post(self.pool(enc1_2))
        enc3_2 = self.encoder3_post(self.pool(enc2_2))
        enc4_2 = self.encoder4_post(self.pool(enc3_2))
        enc5_2 = self.encoder5_post(self.pool(enc4_2))

        enc_1 = self.fuse_1(torch.cat([enc1_1, enc1_2], dim=1))
        enc_2 = self.fuse_2(torch.cat([enc2_1, enc2_2], dim=1))
        enc_3 = self.fuse_3(torch.cat([enc3_1, enc3_2], dim=1))
        enc_4 = self.fuse_4(torch.cat([enc4_1, enc4_2], dim=1))
        enc_5 = self.cmca_5(enc5_1, enc5_2)

        up1 = self.upconv1(enc_5)

        enc_4 = self.attention1(x=enc_4, r=up1)
        up2 = self.upconv2(torch.cat([up1, enc_4], dim=1))

        enc_3 = self.attention2(x=enc_3, r=up2)
        up3 = self.upconv3(torch.cat([up2, enc_3], dim=1))

        enc_2 = self.attention3(x=enc_2, r=up3)
        up4 = self.upconv4(torch.cat([up3, enc_2], dim=1))

        enc_1 = self.attention4(x=enc_1, r=up4)
        up5 = self.upconv5(torch.cat([up4, enc_1], dim=1))

        return self.final(up5)
