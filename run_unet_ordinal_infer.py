# run_unet_ordinal_infer.py — Inference for UNet + Ordinal Damage Loss (ODL)
# ===========================================================================
# Ordinal loss is training-only; the model architecture is identical to the
# vanilla UNet baseline. This script loads the ODL-trained checkpoint and
# runs standard inference + evaluation.
#
# Produces:
#   <OUTPUT_DIR>/original/   — raw label PNG (0/1/2/3)
#   <OUTPUT_DIR>/colored/    — colour-coded PNG for visual inspection
#   Console: OA, mIoU, per-class IoU, F1, per-event & per-type breakdown

import os
import sys
import subprocess

# ===== Edit these paths =====
ROOT     = r"D:\Project\haoChen\BRIGHT"
# Path to the best checkpoint saved by run_unet_ordinal.py.
# Typically: <SAVE_DIR>/<run_folder>/best_model.pth
MODEL_PATH  = os.path.join(ROOT, "checkpoints", "unet_ordinal",
                           "<run_folder>", "best_model.pth")   # ← update <run_folder>
OUTPUT_DIR  = os.path.join(ROOT, "infer_results", "unet_ordinal")
DEVICE      = "cuda"   # "cuda" | "cpu" | "auto"
# ============================

BDA_ROOT   = os.path.join(ROOT, "bda_benchmark")
DATA_PATH  = os.path.join(ROOT, "data")
SPLIT_DIR  = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
TEST_LIST  = os.path.join(SPLIT_DIR, "test_set.txt")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py")


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Please update MODEL_PATH to point to the best_model.pth saved by run_unet_ordinal.py."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run([
        sys.executable, INFER_SCRIPT,

        "--model_path",         MODEL_PATH,
        "--model_type",         "UNet",
        "--test_dataset_path",  DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir",         OUTPUT_DIR,
        "--device",             DEVICE,

        # ODL is a training-only loss — no extra inference flags needed.
        # To compare with SGR, add: --sam_mask_dir <mask_dir> --use_sgr
    ])

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
