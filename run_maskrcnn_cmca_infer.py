# run_maskrcnn_cmca_infer.py — Inference for Mask R-CNN + CMCA
# ===============================================================
# Loads a checkpoint trained by run_maskrcnn_cmca.py and runs inference.
#
# Produces:
#   <OUTPUT_DIR>/original/  — raw label PNG (0/1/2/3)
#   <OUTPUT_DIR>/colored/   — colour-coded PNG for visual inspection
#   Optional COCO JSON      — for challenge submission pipeline

import os
import sys
import subprocess

# ===== Edit these paths =====
ROOT = os.path.abspath(os.path.dirname(__file__))
# ROOT = r"D:\Project\haoChen\BRIGHT"

# Path to trained checkpoint from run_maskrcnn_cmca.py
MODEL_PATH = os.path.join(
    ROOT,
    "checkpoints", "maskrcnn_cmca",
    "BRIGHT", "MaskRCNN_CMCA_<run_folder>", "best_model.pth"
)  # <- update <run_folder>

OUTPUT_DIR = os.path.join(ROOT, "infer_results", "maskrcnn_cmca")
DEVICE = "cuda:0"   # "cuda:0" | "cuda:1" | "cpu" | "auto"

# Optional: export COCO prediction JSON
SAVE_COCO_JSON = os.path.join(OUTPUT_DIR, "pred_maskrcnn_cmca.json")
# Optional image-id mapping JSON ({tile_id: coco_image_id}); set None if not needed
IMAGE_ID_MAP_JSON = None
# ============================

BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
TEST_LIST = os.path.join(SPLIT_DIR, "test_set.txt")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_MaskRCNN_CMCA.py")


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
            "Please update MODEL_PATH to the checkpoint saved by run_maskrcnn_cmca.py.\n"
            "Example: checkpoints/maskrcnn_cmca/BRIGHT/MaskRCNN_CMCA_20260422_130000/best_model.pth"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, INFER_SCRIPT,
        "--model_path", MODEL_PATH,
        "--test_dataset_path", DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir", OUTPUT_DIR,
        "--device", DEVICE,

        # Keep consistent with training defaults
        "--cmca_num_heads", "4",
        "--cmca_sr_ratio", "2",
        "--score_thr", "0.3",
        "--mask_thr", "0.5",

        # If test labels exist (e.g., local val/test with labels), print quick mIoU
        "--with_label",

        # Export COCO predictions
        "--save_coco_json", SAVE_COCO_JSON,
    ]

    if IMAGE_ID_MAP_JSON is not None:
        cmd.extend(["--image_id_map_json", IMAGE_ID_MAP_JSON])

    run(cmd)
    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
