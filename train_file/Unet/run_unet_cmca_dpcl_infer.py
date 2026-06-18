# run_unet_cmca_dpcl_infer.py — Inference for UNet-CMCA + SP-DPCL
# ================================================================
# Loads a checkpoint trained by run_unet_cmca_dpcl.py and runs full
# evaluation on the test set. This is the FINAL combined model that
# pairs your two innovations:
#   - CMCA: input-side cross-modal change attention
#   - DPCL: output-side prototype contrastive embedding geometry
#
# DPCL is training-time only; at inference the network forward is the
# vanilla UNetCMCA forward (no contrastive head, no prototype lookup).
#
# Produces:
#   <OUTPUT_DIR>/original/  — raw label PNG (0/1/2/3)
#   <OUTPUT_DIR>/colored/   — colour-coded PNG for visual inspection
#   Console: OA, mIoU, per-class IoU, F1,
#            per-event mIoU, per-disaster-type mIoU

import os
import sys
import subprocess

# ===== Edit these paths =====
ROOT       = r"D:\Project\haoChen\BRIGHT"
# Path to best checkpoint saved by run_unet_cmca_dpcl.py.
# Typically: checkpoints/unet_cmca_dpcl/BRIGHT/UNetCMCA_SPDPCL_<timestamp>/best_model.pth
MODEL_PATH = os.path.join(ROOT, "checkpoints", "unet_cmca_dpcl",
                          "<run_folder>", "best_model.pth")   # ← update <run_folder>
OUTPUT_DIR = os.path.join(ROOT, "infer_results", "unet_cmca_dpcl")
DEVICE     = "cuda:0"   # "cuda:0" | "cuda:1" | "cpu" | "auto"
# ============================

BDA_ROOT     = os.path.join(ROOT, "bda_benchmark")
DATA_PATH    = os.path.join(ROOT, "data")
SPLIT_DIR    = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
TEST_LIST    = os.path.join(SPLIT_DIR, "test_set.txt")
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
            f"Checkpoint not found:\n  {MODEL_PATH}\n"
            "Please update MODEL_PATH to the best_model.pth saved by run_unet_cmca_dpcl.py.\n"
            "Example: checkpoints/unet_cmca_dpcl/BRIGHT/UNetCMCA_SPDPCL_20260503_120000/best_model.pth"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run([
        sys.executable, INFER_SCRIPT,

        "--model_path",          MODEL_PATH,
        # Must match training: UNetCMCA backbone with cross-modal attention.
        "--model_type",          "UNetCMCA",
        "--test_dataset_path",   DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir",          OUTPUT_DIR,
        "--device",              DEVICE,

        # DPCL is training-only — no extra inference flags needed.
    ])

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
