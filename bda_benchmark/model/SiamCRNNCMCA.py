"""
SiamCRNN-CMCA.

This variant keeps SiamCRNN's original dual ResNet18 encoders, ConvLSTM temporal
fusion, and FPN-like decoder. CMCA is inserted only at the deepest feature level
and added as a residual change signal to the top-down damage branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SiamCRNN import ConvLSTM, ResBlock
from model.UNetCMCA import CrossModalChangeAttention

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


def _resnet18(pretrained=True):
    try:
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        return torchvision.models.resnet18(weights=weights)
    except AttributeError:
        return torchvision.models.resnet18(pretrained=pretrained)


class SiamCRNNCMCA(nn.Module):
    def __init__(self, num_classes=4, cmca_heads=8, cmca_sr_ratio=1, pretrained=True):
        super().__init__()
        expansion = 1

        self.encoder_1 = _resnet18(pretrained=pretrained)
        self.encoder_2 = _resnet18(pretrained=pretrained)
        return_nodes = {
            "layer1": "feat1",
            "layer2": "feat2",
            "layer3": "feat3",
            "layer4": "feat4",
        }
        self.extractor_1 = create_feature_extractor(self.encoder_1, return_nodes=return_nodes)
        self.extractor_2 = create_feature_extractor(self.encoder_2, return_nodes=return_nodes)

        self.convlstm_4 = ConvLSTM(input_dim=512 * expansion, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_3 = ConvLSTM(input_dim=256 * expansion, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_2 = ConvLSTM(input_dim=128 * expansion, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_1 = ConvLSTM(input_dim=64 * expansion, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)

        self.cmca_4 = CrossModalChangeAttention(
            dim=512 * expansion,
            num_heads=cmca_heads,
            sr_ratio=cmca_sr_ratio,
        )
        self.cmca_4_proj = nn.Sequential(
            nn.Conv2d(512 * expansion, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.trans_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=512 * expansion, out_channels=128, padding=1),
                                           nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=256 * expansion, out_channels=128, padding=1),
                                           nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=128 * expansion, out_channels=128, padding=1),
                                           nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=64 * expansion, out_channels=128, padding=1),
                                           nn.BatchNorm2d(128), nn.ReLU())

        self.smooth_layer_13 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_12 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_11 = ResBlock(in_channels=128, out_channels=128, stride=1)

        self.smooth_layer_23 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_22 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_21 = ResBlock(in_channels=128, out_channels=128, stride=1)

        self.main_clf_loc = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.main_clf_clf = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)

    @staticmethod
    def _split_inputs(x, post=None):
        if post is not None:
            return x, post, True
        mid = x.shape[1] // 2
        return x[:, :mid], x[:, mid:], False

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear") + y

    def forward(self, pre_data, post_data=None):
        pre_data, post_data, return_loc = self._split_inputs(pre_data, post_data)

        pre_features = self.extractor_1(pre_data)
        post_features = self.extractor_2(post_data)
        pre_low_level_feat_1, pre_low_level_feat_2, pre_low_level_feat_3, pre_output = pre_features["feat1"], pre_features["feat2"], pre_features["feat3"], pre_features["feat4"]
        post_low_level_feat_1, post_low_level_feat_2, post_low_level_feat_3, post_output = post_features["feat1"], post_features["feat2"], post_features["feat3"], post_features["feat4"]

        p4_loc = self.trans_layer_4(pre_output)
        combined_4 = torch.stack([pre_output, post_output], dim=1)
        _, last_state_list_4 = self.convlstm_4(combined_4)
        p4 = last_state_list_4[0][0]
        p4 = p4 + self.cmca_4_proj(self.cmca_4(pre_output, post_output))

        p3_loc = self.trans_layer_3(pre_low_level_feat_3)
        p3_loc = self._upsample_add(p4_loc, p3_loc)
        p3_loc = self.smooth_layer_13(p3_loc)
        combined_3 = torch.stack([pre_low_level_feat_3, post_low_level_feat_3], dim=1)
        _, last_state_list_3 = self.convlstm_3(combined_3)
        p3 = last_state_list_3[0][0]
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_23(p3)

        p2_loc = self.trans_layer_2(pre_low_level_feat_2)
        p2_loc = self._upsample_add(p3_loc, p2_loc)
        p2_loc = self.smooth_layer_12(p2_loc)
        combined_2 = torch.stack([pre_low_level_feat_2, post_low_level_feat_2], dim=1)
        _, last_state_list_2 = self.convlstm_2(combined_2)
        p2 = last_state_list_2[0][0]
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_22(p2)

        p1_loc = self.trans_layer_1(pre_low_level_feat_1)
        p1_loc = self._upsample_add(p2_loc, p1_loc)
        p1_loc = self.smooth_layer_11(p1_loc)

        combined_1 = torch.stack([pre_low_level_feat_1, post_low_level_feat_1], dim=1)
        _, last_state_list_1 = self.convlstm_1(combined_1)
        p1 = last_state_list_1[0][0]
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_21(p1)

        output_loc = self.main_clf_loc(p1_loc)
        output_loc = F.interpolate(output_loc, size=pre_data.size()[-2:], mode="bilinear")
        output_clf = self.main_clf_clf(p1)
        output_clf = F.interpolate(output_clf, size=pre_data.size()[-2:], mode="bilinear")

        if return_loc:
            return output_loc, output_clf
        return output_clf
