import os
import sys
import subprocess

# ========= 修改下面这些路径 =========
PROJECT_ROOT = r"D:\Project\haoChen\BRIGHT"
MODEL_PATH = r"D:\Project\haoChen\BRIGHT\checkpoints\sam_guided_unet\BRIGHT\UNet_SAM_SOFT_20260331_083902\best_model.pth"
DATA_PATH = r"D:\Project\haoChen\BRIGHT\data"
TEST_LIST = r"D:\Project\haoChen\BRIGHT\bda_benchmark\dataset\splitname\standard_ML\test_set.txt"
SAM_MASK_DIR = r"D:\Project\haoChen\BRIGHT\outputs\sam_masks\standard_ML"
OUTPUT_DIR = r"D:\Project\haoChen\BRIGHT\infer_results\unet_sam_soft_best_001"
DEVICE = "cuda"  # auto / cuda / mps / cpu
# ==================================

BDA_ROOT = os.path.join(PROJECT_ROOT, "bda_benchmark")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py")


def _ensure_exists(path, path_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path_name} not found: {path}")


def main():
    _ensure_exists(INFER_SCRIPT, "infer script")
    _ensure_exists(MODEL_PATH, "model")
    _ensure_exists(DATA_PATH, "dataset")
    _ensure_exists(TEST_LIST, "test list")
    _ensure_exists(SAM_MASK_DIR, "SAM mask dir")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = os.environ.copy()
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev_pythonpath else BDA_ROOT + os.pathsep + prev_pythonpath

    cmd = [
    sys.executable,
    INFER_SCRIPT,
    "--model_path", MODEL_PATH,
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", TEST_LIST,
    "--output_dir", OUTPUT_DIR,
    "--sam_mask_dir", SAM_MASK_DIR,
    "--sam_mask_suffix", ".png",
    "--sam_mask_threshold", "127",
    "--sam_mode", "soft",
    "--device", DEVICE,
    ]

    print("\nRunning SAM-guided UNet inference...\n")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print("\nNote: SAM mask files should be named as <tile_id>_building_mask.png\n")

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
