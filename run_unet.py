# run_unet.py — Vanilla UNet baseline (reproduces paper results)
# ==============================================================
# Uses the EXACT same hyperparameters as the BRIGHT paper:
#   - AdamW, lr=1e-4, weight_decay=5e-3
#   - Constant LR (no scheduler)
#   - CE + 0.75 * Lovász softmax
#   - max_iters=800000, batch_size=16, crop_size=640

import os
import subprocess

DATA_PATH = r"D:\Project\haoChen\BRIGHT\data"
CKPT_PATH = r"D:\Project\haoChen\BRIGHT\checkpoints\unet_baseline"

os.makedirs(CKPT_PATH, exist_ok=True)

cmd = [
    "python",
    "bda_benchmark/script/standard_ML/train_UNet.py",
    "--dataset", "BRIGHT",
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "8",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",        # paper default
    "--lr_policy", "constant",       # paper default: no scheduler
    "--model_type", "UNet",
    "--model_param_path", CKPT_PATH,
    "--train_dataset_path", DATA_PATH,
    "--train_data_list_path", "./bda_benchmark/dataset/splitname/standard_ML/train_set.txt",
    "--val_dataset_path", DATA_PATH,
    "--val_data_list_path", "./bda_benchmark/dataset/splitname/standard_ML/val_set.txt",
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", "./bda_benchmark/dataset/splitname/standard_ML/test_set.txt",
]

subprocess.run(cmd)
