# run_unet_dacutmix_infer.py -- Inference for UNet + DACutMix
# ============================================================
# DACutMix (Damage-Aware CutMix) is a training-time augmentation only.
# It does not change the model architecture, so inference uses the same
# vanilla UNet backbone.
#
# Usage:
#   1. Set RUN_FOLDER to the folder created under
#      checkpoints/unet_dacutmix/BRIGHT (or set MODEL_PATH directly).
#   2. Run: python run_unet_dacutmix_infer.py

import os
import sys
import subprocess


ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
TEST_LIST = os.path.join(SPLIT_DIR, "test_set.txt")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py")

# ===== Edit one of these =====
# Example folder name:
#   UNet_DACutMix_20260607_153000
RUN_FOLDER = "<run_folder>"
MODEL_PATH = os.path.join(
    ROOT,
    "checkpoints",
    "unet_dacutmix",
    "BRIGHT",
    RUN_FOLDER,
    "best_model.pth",
)

# Or uncomment this and paste the full checkpoint path:
# MODEL_PATH = r"E:\haoChen\BRIGHT\checkpoints\unet_dacutmix\BRIGHT\UNet_DACutMix_YYYYMMDD_HHMMSS\best_model.pth"

OUTPUT_DIR = os.path.join(ROOT, "infer_results", "unet_dacutmix")
DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")
# ============================


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {MODEL_PATH}\n"
            "Update RUN_FOLDER or MODEL_PATH to point to best_model.pth."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run([
        sys.executable,
        INFER_SCRIPT,
        "--model_path", MODEL_PATH,
        "--model_type", "UNet",
        "--test_dataset_path", DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir", OUTPUT_DIR,
        "--device", DEVICE,
    ])

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
