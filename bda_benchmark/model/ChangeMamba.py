"""
ChangeMamba adapter for the BRIGHT benchmark.

The original ChangeMamba repository keeps the BRIGHT model under
`changedetection/models/ChangeMambaMMBDA.py` and imports it as package
`MambaCD`. In this project we keep ChangeMamba as an external checkout and
provide a small adapter so the unified BRIGHT trainer can call:

    model(concat_6ch) -> damage logits

Old-style calls are also supported:

    model(pre, post) -> (building logits, damage logits)
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.UNetCMCA import CrossModalChangeAttention


def _repo_root_from_env():
    env_root = os.environ.get("CHANGEMAMBA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # bda_benchmark/model/ChangeMamba.py -> bright-cmca -> sibling checkout.
    project_root = Path(__file__).resolve().parents[2]
    return (project_root.parent / "ChangeMamba-master").resolve()


def _install_fvcore_stub_if_needed():
    """VMamba imports fvcore only for FLOP utilities; training does not need it."""
    if "fvcore.nn" in sys.modules:
        return
    if importlib.util.find_spec("fvcore") is not None:
        return

    fvcore_mod = types.ModuleType("fvcore")
    fvcore_nn_mod = types.ModuleType("fvcore.nn")

    def _missing(*_args, **_kwargs):
        raise ImportError(
            "fvcore is required only for ChangeMamba FLOP analysis. "
            "Install fvcore to use that analysis utility."
        )

    fvcore_nn_mod.FlopCountAnalysis = _missing
    fvcore_nn_mod.flop_count_str = _missing
    fvcore_nn_mod.flop_count = _missing
    fvcore_nn_mod.parameter_count = _missing
    fvcore_mod.nn = fvcore_nn_mod

    sys.modules.setdefault("fvcore", fvcore_mod)
    sys.modules.setdefault("fvcore.nn", fvcore_nn_mod)


def _install_csm_triton_stub_if_needed():
    """Allow VMamba import on systems without triton when Triton paths are unused."""
    if importlib.util.find_spec("triton") is not None:
        return
    if "csm_triton" in sys.modules:
        return

    csm_mod = types.ModuleType("csm_triton")

    class _MissingTritonFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *_args, **_kwargs):
            raise ImportError(
                "triton is required for VMamba Triton forward paths. "
                "Install triton or use a non-Triton ChangeMamba forward_type."
            )

    csm_mod.CrossScanTriton = _MissingTritonFn
    csm_mod.CrossMergeTriton = _MissingTritonFn
    csm_mod.CrossScanTriton1b1 = _MissingTritonFn
    sys.modules["csm_triton"] = csm_mod


def _ensure_mambacd_package(root):
    """Expose an arbitrary checkout path as import package `MambaCD`."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            "ChangeMamba source folder not found. Set CHANGEMAMBA_ROOT to the "
            f"ChangeMamba checkout. Tried: {root}"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    pkg = sys.modules.get("MambaCD")
    if pkg is None:
        pkg = types.ModuleType("MambaCD")
        pkg.__path__ = [root_str]
        sys.modules["MambaCD"] = pkg
    elif root_str not in getattr(pkg, "__path__", []):
        pkg.__path__.append(root_str)

    _install_fvcore_stub_if_needed()
    _install_csm_triton_stub_if_needed()


def _deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_yaml(path):
    try:
        import yaml
    except Exception:
        return {}

    path = Path(path)
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_config(root):
    cfg = {
        "MODEL": {
            "NUM_CLASSES": 1000,
            "DROP_PATH_RATE": 0.2,
            "VSSM": {
                "PATCH_SIZE": 4,
                "IN_CHANS": 3,
                "DEPTHS": [2, 2, 4, 2],
                "EMBED_DIM": 96,
                "SSM_D_STATE": 1,
                "SSM_RATIO": 2.0,
                "SSM_RANK_RATIO": 2.0,
                "SSM_DT_RANK": "auto",
                "SSM_ACT_LAYER": "silu",
                "SSM_CONV": 3,
                "SSM_CONV_BIAS": False,
                "SSM_DROP_RATE": 0.0,
                "SSM_INIT": "v0",
                "SSM_FORWARDTYPE": "v3noz",
                "MLP_RATIO": 4.0,
                "MLP_ACT_LAYER": "gelu",
                "MLP_DROP_RATE": 0.0,
                "PATCH_NORM": True,
                "NORM_LAYER": "ln",
                "DOWNSAMPLE": "v3",
                "PATCHEMBED": "v2",
                "GMLP": False,
            },
        },
        "TRAIN": {
            "USE_CHECKPOINT": False,
        },
    }

    cfg_path = os.environ.get("CHANGEMAMBA_CFG")
    if not cfg_path:
        cfg_path = (
            Path(root)
            / "changedetection"
            / "configs"
            / "vssm1"
            / "vssm_tiny_224_0229flex.yaml"
        )
    _deep_update(cfg, _load_yaml(cfg_path))
    return cfg


def _config_to_kwargs(cfg):
    vssm = cfg["MODEL"]["VSSM"]
    dt_rank = vssm["SSM_DT_RANK"]
    return dict(
        patch_size=vssm["PATCH_SIZE"],
        in_chans=vssm["IN_CHANS"],
        num_classes=cfg["MODEL"]["NUM_CLASSES"],
        depths=vssm["DEPTHS"],
        dims=vssm["EMBED_DIM"],
        ssm_d_state=vssm["SSM_D_STATE"],
        ssm_ratio=vssm["SSM_RATIO"],
        ssm_rank_ratio=vssm["SSM_RANK_RATIO"],
        ssm_dt_rank=("auto" if dt_rank == "auto" else int(dt_rank)),
        ssm_act_layer=vssm["SSM_ACT_LAYER"],
        ssm_conv=vssm["SSM_CONV"],
        ssm_conv_bias=vssm["SSM_CONV_BIAS"],
        ssm_drop_rate=vssm["SSM_DROP_RATE"],
        ssm_init=vssm["SSM_INIT"],
        forward_type=vssm["SSM_FORWARDTYPE"],
        mlp_ratio=vssm["MLP_RATIO"],
        mlp_act_layer=vssm["MLP_ACT_LAYER"],
        mlp_drop_rate=vssm["MLP_DROP_RATE"],
        drop_path_rate=cfg["MODEL"]["DROP_PATH_RATE"],
        patch_norm=vssm["PATCH_NORM"],
        norm_layer=vssm["NORM_LAYER"],
        downsample_version=vssm["DOWNSAMPLE"],
        patchembed_version=vssm["PATCHEMBED"],
        gmlp=vssm["GMLP"],
        use_checkpoint=cfg["TRAIN"]["USE_CHECKPOINT"],
    )


class ChangeMamba(nn.Module):
    """MambaBDA-Tiny adapted to the BRIGHT unified trainer."""

    def __init__(
        self,
        in_channels=6,
        num_classes=4,
        output_building=2,
        use_cmca=False,
        cmca_heads=8,
        cmca_sr_ratio=1,
    ):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("ChangeMamba expects paired pre/post channels.")

        root = _repo_root_from_env()
        _ensure_mambacd_package(root)

        from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
        from MambaCD.changedetection.models.ChangeDecoder_BRIGHT import ChangeDecoder
        from MambaCD.changedetection.models.SemanticDecoder import SemanticDecoder
        from MambaCD.classification.models.vmamba import LayerNorm2d

        cfg = _default_config(root)
        kwargs = _config_to_kwargs(cfg)
        pretrained = os.environ.get("CHANGEMAMBA_PRETRAINED") or None
        if pretrained and not Path(pretrained).is_file():
            print(f"[WARN] CHANGEMAMBA_PRETRAINED not found, training from scratch: {pretrained}")
            pretrained = None

        self.encoder_1 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        self.encoder_2 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        norm_layers = {
            "ln": nn.LayerNorm,
            "ln2d": LayerNorm2d,
            "bn": nn.BatchNorm2d,
        }
        act_layers = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }

        norm_layer = norm_layers[kwargs["norm_layer"].lower()]
        ssm_act_layer = act_layers[kwargs["ssm_act_layer"].lower()]
        mlp_act_layer = act_layers[kwargs["mlp_act_layer"].lower()]
        decoder_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("norm_layer", "ssm_act_layer", "mlp_act_layer")
        }

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder_1.dims,
            channel_first=self.encoder_1.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **decoder_kwargs,
        )
        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder_2.dims,
            channel_first=self.encoder_2.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **decoder_kwargs,
        )

        self.use_cmca = bool(use_cmca)
        if self.use_cmca:
            self.cmca = CrossModalChangeAttention(
                dim=self.encoder_2.dims[-1],
                num_heads=cmca_heads,
                sr_ratio=cmca_sr_ratio,
            )

        self.aux_clf = nn.Conv2d(128, output_building, kernel_size=1)
        self.main_clf = nn.Conv2d(128, num_classes, kernel_size=1)
        self.supports_loc_head = True

    @staticmethod
    def _split_inputs(x, post=None):
        if post is not None:
            return x, post, True
        mid = x.shape[1] // 2
        return x[:, :mid], x[:, mid:], False

    def forward(self, x, post=None, return_loc=False):
        pre_data, post_data, explicit_pair = self._split_inputs(x, post)
        return_loc = bool(return_loc or explicit_pair)

        pre_features = self.encoder_1(pre_data)
        post_features = self.encoder_2(post_data)

        if self.use_cmca:
            post_features = list(post_features)
            post_features[-1] = post_features[-1] + self.cmca(pre_features[-1], post_features[-1])

        output_building = self.decoder_building(pre_features)
        output_damage = self.decoder_damage(pre_features, post_features)

        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(
            output_building,
            size=pre_data.size()[-2:],
            mode="bilinear",
            align_corners=False,
        )

        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(
            output_damage,
            size=post_data.size()[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if return_loc:
            return output_building, output_damage
        return output_damage


class ChangeMambaCMCA(ChangeMamba):
    def __init__(self, *args, **kwargs):
        kwargs["use_cmca"] = True
        super().__init__(*args, **kwargs)
