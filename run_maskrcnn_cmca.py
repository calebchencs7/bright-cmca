# run_maskrcnn_cmca.py — Train Mask R-CNN + CMCA
# ===============================================
# Trains Mask R-CNN with CMCA stem fusion on BRIGHT.
#
# Input: 4-channel [pre-event RGB (3ch) + post-event SAR (1ch)]
# Loss : torchvision Mask R-CNN losses (cls + box + mask + rpn)
#
# Notes:
#   1) If your machine has no internet, keep --no-pretrained.
#   2) If CUDA OOM, reduce --train_batch_size and/or --crop_size.

import os
import sys
import subprocess

# ===== Paths =====
# Default: current repo root (works directly if this file stays at project root)
ROOT = os.path.abspath(os.path.dirname(__file__))
# If you prefer fixed path, uncomment and edit:
# ROOT = r"D:\Project\haoChen\BRIGHT"

BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "maskrcnn_cmca")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_MaskRCNN_CMCA.py")

DEVICE = "cuda:0"   # "cuda:0" | "cuda:1" | "cpu" | "auto"

os.makedirs(SAVE_DIR, exist_ok=True)


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev

    # Reduce CUDA memory fragmentation.
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
    "--suffix", ".tif",

    # Training — align with UNet-CMCA for fair comparison
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--epochs", "1",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    # Model (CMCA)
    "--cmca_num_heads", "4",
    "--cmca_sr_ratio", "2",
    "--model_param_path", SAVE_DIR,

    # For offline or strict fairness with random init
    "--no-pretrained",

    # Logging
    "--eval_interval", "500",
    "--curve_log_interval", "10",
    "--curve_save_interval", "500",

    # Performance (same style as UNet runner)
    "--use_amp",
    "--amp_dtype", "fp16",
    "--pin_memory",
    "--persistent_workers",
    "--prefetch_factor", "2",
    "--device", DEVICE,
])
