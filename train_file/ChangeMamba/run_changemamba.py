# run_changemamba.py -- ChangeMamba / MambaBDA-Tiny baseline

import os
import subprocess
import sys


ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
CHANGEMAMBA_ROOT = os.environ.get(
    "CHANGEMAMBA_ROOT",
    os.path.join(os.path.dirname(ROOT), "ChangeMamba-master"),
)
CHANGEMAMBA_CFG = os.environ.get(
    "CHANGEMAMBA_CFG",
    os.path.join(
        CHANGEMAMBA_ROOT,
        "changedetection",
        "configs",
        "vssm1",
        "vssm_tiny_224_0229flex.yaml",
    ),
)
CHANGEMAMBA_PRETRAINED = os.environ.get("CHANGEMAMBA_PRETRAINED", "")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "changemamba")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")

DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")

os.makedirs(SAVE_DIR, exist_ok=True)


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CHANGEMAMBA_ROOT"] = CHANGEMAMBA_ROOT
    env["CHANGEMAMBA_CFG"] = CHANGEMAMBA_CFG
    if CHANGEMAMBA_PRETRAINED:
        env["CHANGEMAMBA_PRETRAINED"] = CHANGEMAMBA_PRETRAINED
    print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


run([
    sys.executable, TRAIN_SCRIPT,

    "--dataset", "BRIGHT",
    "--train_dataset_path", DATA_PATH,
    "--train_data_list_path", os.path.join(SPLIT_DIR, "train_set.txt"),
    "--val_dataset_path", DATA_PATH,
    "--val_data_list_path", os.path.join(SPLIT_DIR, "val_set.txt"),
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", os.path.join(SPLIT_DIR, "test_set.txt"),

    "--train_batch_size", "4",
    "--eval_batch_size", "2",
    "--num_workers", "8",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    "--model_type", "ChangeMamba",
    "--model_param_path", SAVE_DIR,
    "--use_loc_loss",
    "--loc_loss_weight", "1.0",

    "--eval_interval", "500",
    "--curve_log_interval", "10",
    "--curve_save_interval", "500",

    "--use_amp",
    "--amp_dtype", "fp16",
    "--pin_memory",
    "--persistent_workers",
    "--prefetch_factor", "2",
    "--device", DEVICE,
])
