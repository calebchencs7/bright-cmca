# run_unet_cmca_dpcl.py — UNet-CMCA + Damage Prototype Contrastive Learning
# ==========================================================================
# Trains UNet-CMCA (your existing first innovation: dual-branch encoder +
# Cross-Modal Change Attention) combined with DPCL (the second innovation:
# class-prototype InfoNCE on dec3 features). This is the FINAL combined
# model that the paper will report.
#
# Story:
#   CMCA — learns *where* cross-modal change evidence comes from
#          (input-side: how to fuse optical and SAR features semantically).
#   DPCL — learns *how* fused features should be organised in embedding
#          space so that ambiguous "Damaged" pixels separate cleanly from
#          "Intact" and "Destroyed".
#
# Loss = CE + 0.75 * Lovász + λ_DPCL(t) * DamagePrototypeContrastive
#
# This script uses SP-DPCL (1 prototype per class) for v1. To upgrade to
# MP-DPCL after v1 ablation: change --dpcl_num_prototypes to "1,3,2" and
# set --dpcl_ortho_weight 0.01.

import os
import sys
import subprocess

ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "unet_cmca_dpcl")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")

DEVICE = "cuda:0"

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

    # Training — same hyper-parameters as CMCA-only run for fair comparison
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

    # DPCL — SP-DPCL (single prototype per class)
    "--use_dpcl",
    "--dpcl_weight", "0.1",
    "--dpcl_num_prototypes", "1,1,1",
    "--dpcl_class_loss_weights", "1.0,2.0,1.0",
    "--dpcl_feat_dim", "256",                     # d3 channel count in UNetCMCA
    "--dpcl_proj_dim", "128",
    "--dpcl_samples_per_class", "512",
    "--dpcl_warmup_iters", "3000",
    "--dpcl_ramp_iters", "2000",
    "--dpcl_momentum", "0.99",
    "--dpcl_temperature", "0.1",
    "--dpcl_ortho_weight", "0.0",

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
