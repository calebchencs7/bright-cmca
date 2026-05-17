# run_unet_dpcl.py — UNet + Damage Prototype Contrastive Learning (DPCL)
# =======================================================================
# Trains vanilla UNet with DPCL only (no CMCA, no ODL, no SGR).
# This is the SP-DPCL ablation: validates that prototype contrastive
# learning by itself improves over the BRIGHT baseline.
#
# Loss = CE + 0.75 * Lovász + λ_DPCL(t) * DamagePrototypeContrastive
#
# DPCL operates on a 128-d projection of dec3 (160x160, 256 ch at crop=640),
# with one prototype per building class (intact / damaged / destroyed).
# Background is excluded as an anchor. Damaged class is up-weighted (2x) in
# the InfoNCE loss to address its persistently low IoU on BRIGHT.
#
# To upgrade to MP-DPCL, change --dpcl_num_prototypes from "1,1,1" to e.g.
# "1,3,2" and set --dpcl_ortho_weight 0.01. Everything else stays the same.

import os
import sys
import subprocess

ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "unet_dpcl")
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

    # Training — paper baseline hyper-parameters
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",
    "--lr_policy", "constant",

    # Model — UNetWithFeatures (subclass of vanilla UNet that exposes dec3 for DPCL).
    # Identical parameters and state dict to UNet; forward() additionally
    # returns the mid-decoder feature when return_features=True.
    "--model_type", "UNetWithFeatures",
    "--model_param_path", SAVE_DIR,

    # DPCL — SP-DPCL (single prototype per class)
    "--use_dpcl",
    "--dpcl_weight", "0.1",
    "--dpcl_num_prototypes", "1,1,1",
    "--dpcl_class_loss_weights", "1.0,2.0,1.0",   # upweight Damaged
    "--dpcl_feat_dim", "256",                     # dec3 channel count
    "--dpcl_proj_dim", "128",
    "--dpcl_samples_per_class", "512",
    "--dpcl_warmup_iters", "3000",                # only update prototypes
    "--dpcl_ramp_iters", "2000",                  # then ramp 0 → 0.1
    "--dpcl_momentum", "0.99",
    "--dpcl_temperature", "0.1",
    "--dpcl_ortho_weight", "0.0",                 # SP: no ortho needed

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
