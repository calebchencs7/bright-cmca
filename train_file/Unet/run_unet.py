# run_unet.py — Vanilla UNet baseline (reproduces paper results)
# ==============================================================
# Uses the EXACT same hyperparameters as the BRIGHT paper:
#   - AdamW, lr=1e-4, weight_decay=5e-3
#   - Constant LR (no scheduler)
#   - CE + 0.75 * Lovász softmax
#   - max_iters=800000, batch_size=16, crop_size=640

import os
import subprocess
import sys

ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
CKPT_PATH = os.path.join(ROOT, "checkpoints", "unet")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")
DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")
SEED = os.environ.get("SEED", "0")

os.makedirs(CKPT_PATH, exist_ok=True)

cmd = [
    sys.executable,
    TRAIN_SCRIPT,
    "--dataset",
    "BRIGHT",
    "--train_batch_size",
    "16",
    "--eval_batch_size",
    "4",
    "--num_workers",
    "16",
    "--crop_size",
    "640",
    "--max_iters",
    "800000",
    "--learning_rate",
    "1e-4",
    "--weight_decay",
    "5e-3",  # paper default
    "--lr_policy",
    "constant",  # paper default: no scheduler
    "--model_type",
    "UNet",
    "--model_param_path",
    CKPT_PATH,
    "--train_dataset_path",
    DATA_PATH,
    "--train_data_list_path",
    os.path.join(SPLIT_DIR, "train_set.txt"),
    "--val_dataset_path",
    DATA_PATH,
    "--val_data_list_path",
    os.path.join(SPLIT_DIR, "val_set.txt"),
    "--test_dataset_path",
    DATA_PATH,
    "--test_data_list_path",
    os.path.join(SPLIT_DIR, "test_set.txt"),
    "--seed",
    SEED,
    "--device",
    DEVICE,
]

env = os.environ.copy()
previous_pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = (
    BDA_ROOT if not previous_pythonpath else BDA_ROOT + os.pathsep + previous_pythonpath
)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run(cmd, check=True, cwd=ROOT, env=env)
