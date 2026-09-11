# run_unet_cmca_dacutmix_cross_disaster.py
# Strict zero-shot leave-one-disaster-type-out: UNet-CMCA + DACutMix 0.3

import os
import subprocess
import sys


ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
STANDARD_SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
LEAVE_TYPE_ROOT = os.path.join(
    BDA_ROOT, "dataset", "splitname", "leave_one_disaster_type"
)
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")
SPLIT_SCRIPT = os.path.join(
    BDA_ROOT,
    "script",
    "standard_ML",
    "build_leave_one_disaster_type_splits.py",
)

HOLDOUT_TYPE = os.environ.get("BRIGHT_HOLDOUT_TYPE", "earthquake").lower()
DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")

SPLIT_DIR = os.path.join(LEAVE_TYPE_ROOT, HOLDOUT_TYPE)
SAVE_DIR = os.path.join(
    ROOT,
    "checkpoints",
    "cross_disaster",
    "unet_cmca_dacutmix_0.3",
    HOLDOUT_TYPE,
)

os.makedirs(SAVE_DIR, exist_ok=True)


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if not os.path.isfile(os.path.join(SPLIT_DIR, "train_set.txt")):
    run([
        sys.executable,
        SPLIT_SCRIPT,
        "--source_split_dir", STANDARD_SPLIT_DIR,
        "--output_root", LEAVE_TYPE_ROOT,
        "--holdout_type", HOLDOUT_TYPE,
    ])


run([
    sys.executable, TRAIN_SCRIPT,

    "--dataset", "BRIGHT",
    "--train_dataset_path", DATA_PATH,
    "--train_data_list_path", os.path.join(SPLIT_DIR, "train_set.txt"),
    "--val_dataset_path", DATA_PATH,
    "--val_data_list_path", os.path.join(SPLIT_DIR, "val_set.txt"),
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", os.path.join(SPLIT_DIR, "test_set.txt"),

    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    "--model_type", "UNetCMCA",
    "--model_param_path", SAVE_DIR,

    "--damage_class_ids", "2,3",
    "--use_dacutmix",
    "--dacutmix_prob", "0.3",
    "--dacutmix_min_damage_pixels", "200",
    "--dacutmix_min_damage_ratio", "0.05",
    "--dacutmix_patch_min_ratio", "0.12",
    "--dacutmix_patch_max_ratio", "0.35",
    "--dacutmix_box_tries", "10",
    "--dacutmix_donor_tries", "10",

    "--modality_mode", "both",

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
