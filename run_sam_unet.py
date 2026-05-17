# run_sam_unet.py — UNet + SAM-Guided Refinement (SGR)
# =====================================================
# Trains the vanilla UNet backbone with the pluggable SGR post-processing module.
# The backbone itself is UNCHANGED (6-channel input, same as paper baseline).
# SGR takes backbone logits + SAM mask and outputs refined logits.
#
# To ablate SGR: simply remove --use_sgr and --sam_mask_dir flags.
# To switch backbone: change --model_type to DeepLabV3Plus, SiamAttnUNet, etc.

import os
import sys
import subprocess

ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
SAM_CKPT = os.path.join(ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth")
MASK_DIR = os.path.join(ROOT, "outputs", "sam_masks", "standard_ML")
SAVE_DIR = os.path.join(ROOT, "checkpoints", "sam_guided_unet")

GEN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "generate_sam_building_masks.py")
TRAIN_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "train_UNet.py")

# ===== Switches =====
RUN_SAM_MASK_GEN = True    # Set True to regenerate SAM masks (needed after param changes)
OVERWRITE_MASK = True      # Set True to overwrite existing masks
DEVICE = "cuda"
# ====================

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


def _ensure_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


_ensure_exists(TRAIN_SCRIPT, "train script")

# ---------- Step 1: Generate SAM masks (optional, one-time) ----------
if RUN_SAM_MASK_GEN:
    _ensure_exists(GEN_SCRIPT, "mask generation script")
    _ensure_exists(SAM_CKPT, "SAM checkpoint")
    for split_file in ["train_set.txt", "val_set.txt", "test_set.txt"]:
        split_path = os.path.join(SPLIT_DIR, split_file)
        _ensure_exists(split_path, f"split file {split_file}")
        cmd = [
            sys.executable, GEN_SCRIPT,
            "--dataset_path", DATA_PATH,
            "--data_list_path", split_path,
            "--output_dir", MASK_DIR,
            "--sam_checkpoint", SAM_CKPT,
            "--sam_model_type", "vit_b",
            "--device", DEVICE,
            "--source", "both",
            # Sampling: 24 was too sparse; 48 is too slow (~53s/tile);
            # 32 is SAM default and a good balance for VHR building detection
            "--points_per_side", "32",
            "--crop_n_layers", "1",
            "--min_mask_region_area", "100",   # was 200, too high for small buildings
            # Confidence: these are filter_masks thresholds;
            # SAM generator internal thresholds are auto-set 0.05 lower
            "--pred_iou_thresh", "0.86",
            "--stability_score_thresh", "0.90",
            "--min_predicted_iou", "0.86",
            "--min_stability_score", "0.90",
            # Area: 0.2 (20%) was way too loose — kept roads, fields, etc.
            "--max_area_ratio", "0.08",
            "--min_area_ratio", "0.0002",
            # Shape filters: tighter to reject roads/vegetation/parking lots
            "--max_aspect_ratio", "4.0",       # was default 5.0
            "--min_solidity", "0.55",           # was default 0.4
            "--min_extent", "0.38",             # was default 0.25
            "--min_compactness", "0.12",        # was default 0.05
            # Post-processing: always on
            "--morph_open", "3",
            "--morph_close", "5",
            "--cc_min_area", "100",
            "--cc_max_elongation", "4.0",
        ]
        if OVERWRITE_MASK:
            cmd.append("--overwrite")
        run(cmd)

# ---------- Step 2: Train UNet + SGR ----------
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

    # Training — match paper settings exactly
    "--train_batch_size", "16",
    "--eval_batch_size", "4",
    "--num_workers", "16",
    "--crop_size", "640",
    "--max_iters", "800000",
    "--learning_rate", "1e-4",
    "--weight_decay", "5e-3",        # paper default
    "--lr_policy", "constant",       # paper default: no scheduler

    # Model
    "--model_type", "UNet",
    "--model_param_path", SAVE_DIR,

    # SAM-Guided Refinement (the innovation)
    "--sam_mask_dir", MASK_DIR,
    "--use_sgr",
    "--sgr_hidden_dim", "32",
    "--sgr_alpha_init", "0.1",

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
