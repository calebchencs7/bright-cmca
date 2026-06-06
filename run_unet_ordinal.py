# run_unet_ordinal.py — UNet + Ordinal Damage Loss (ODL)
# =======================================================
# Trains UNet with ordinal damage loss only.
#
# Loss = CE + 0.75 * Lovász + ordinal_weight * OrdinalBCE
#
# Ordinal loss decomposes 3-class damage into 2 binary thresholds:
#   - threshold1: P(damage >= minor)
#   - threshold2: P(damage >= major)
# This penalizes 2-rank errors (Intact↔Destroyed) more than 1-rank errors.

import os
import sys
import subprocess

ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "unet_ordinal")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")

DEVICE = "cuda"

os.makedirs(SAVE_DIR, exist_ok=True)


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
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

    # Training — same as paper baseline
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    # Model
    "--model_type", "UNet",
    "--model_param_path", SAVE_DIR,

    # Ordinal Damage Loss — the only addition vs baseline
    "--use_ordinal_loss",
    "--ordinal_weight", "0.1",
    # No warmup needed: ODL weight 0.1 is only ~7% of total loss, won't destabilize training
    # Total training is only 50K steps, every step counts
    "--ordinal_warmup_iters", "0",

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
