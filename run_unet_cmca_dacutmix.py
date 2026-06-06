# run_unet_cmca_dacutmix.py -- UNet-CMCA + DACutMix (main result)
# =================================================================
# Main launcher for the second contribution:
#   1. CMCA handles optical/SAR cross-modal fusion.
#   2. DACutMix (Damage-Aware CutMix) handles cross-event generalization
#      and Damaged-class data scarcity.
#
# DACutMix is event-aware, damage-aware CutMix applied jointly over
# optical / SAR / label inputs:
#   - donor patches come from a different disaster event,
#   - donor patches must contain >=5% Damaged/Destroyed pixels,
#   - the same crop box is applied to pre/post/label so spatial
#     correspondence across modalities is preserved.
#
# Ablation matrix:
#   UNet                       : run_unet.py
#   UNet + CMCA                : run_unet_cmca.py
#   UNet + DACutMix            : run_unet_dacutmix.py
#   UNet + CMCA + DACutMix     : THIS SCRIPT (main result)
#
import os
import sys
import subprocess


ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "unet_cmca_dacutmix")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")

DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")

os.makedirs(SAVE_DIR, exist_ok=True)


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


run([
    sys.executable, TRAIN_SCRIPT,

    # Data
    "--dataset", "BRIGHT",
    "--train_dataset_path", DATA_PATH,
    "--train_data_list_path", os.path.join(SPLIT_DIR, "train_set.txt"),
    "--val_dataset_path", DATA_PATH,
    "--val_data_list_path", os.path.join(SPLIT_DIR, "val_set.txt"),
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", os.path.join(SPLIT_DIR, "test_set.txt"),

    # Training -- same budget as CMCA baseline
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    # Model
    "--model_type", "UNetCMCA",
    "--model_param_path", SAVE_DIR,

    # DACutMix configuration
    "--damage_class_ids", "2,3",
    "--use_dacutmix",
    "--dacutmix_prob", "0.25",
    "--dacutmix_min_damage_pixels", "200",
    "--dacutmix_min_damage_ratio", "0.05",
    "--dacutmix_patch_min_ratio", "0.12",
    "--dacutmix_patch_max_ratio", "0.35",
    "--dacutmix_box_tries", "10",
    "--dacutmix_donor_tries", "10",

    # Logging
    "--eval_interval", "500",
    "--curve_log_interval", "10",
    "--curve_save_interval", "500",

    # Performance
    "--use_amp",
    "--amp_dtype", "fp16",
    "--pin_memory",
    "--persistent_workers",
    "--prefetch_factor", "2",
    "--device", DEVICE,
])
