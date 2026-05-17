# run_unet_cmca.py — UNet-CMCA + Ordinal Damage Loss
# ====================================================
# Trains UNet-CMCA (dual-branch encoder + Cross-Modal Change Attention)
# with optional Ordinal Damage Loss (ODL).
#
# Two innovations combined:
#   1. CMCA — explicitly captures structural change between optical & SAR
#   2. ODL  — exploits ordinal structure of damage levels
#
# Loss = CE + 0.75 * Lovasz + ordinal_weight * OrdinalBCE

import os
import sys
import subprocess

ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "unet_cmca_odl")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")

DEVICE = "cuda:0"

os.makedirs(SAVE_DIR, exist_ok=True)


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    # Reduce CUDA memory fragmentation (prevents OOM after eval rounds)
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

    # Training — same hyperparameters as baseline UNet (gradient checkpointing handles memory)
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    # Model — UNet-CMCA (dual-branch + cross-modal change attention) 
    "--model_type", "UNetCMCA",
    "--model_param_path", SAVE_DIR,

    # Ordinal Damage Loss
    # "--use_ordinal_loss",
    # "--ordinal_weight", "0.15",
    # "--ordinal_warmup_iters", "5000",

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
