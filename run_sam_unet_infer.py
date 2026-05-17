# run_sam_unet_infer.py — Inference for UNet + SAM-Guided Refinement (SGR)
import os
import sys
import subprocess

PROJECT_ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(PROJECT_ROOT, "bda_benchmark")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py")

# ===== UPDATE THESE PATHS =====
MODEL_PATH = r"D:\Project\haoChen\BRIGHT\checkpoints\sam_guided_unet\BRIGHT\UNet_SGR_YYYYMMDD_HHMMSS\best_model.pth"
DATA_PATH = r"D:\Project\haoChen\BRIGHT\data"
TEST_LIST = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML", "test_set.txt")
SAM_MASK_DIR = r"D:\Project\haoChen\BRIGHT\outputs\sam_masks\standard_ML"
OUTPUT_DIR = r"D:\Project\haoChen\BRIGHT\infer_results\unet_sgr_best"
DEVICE = "cuda"
# ===============================


def main():
    for path, name in [
        (INFER_SCRIPT, "infer script"),
        (MODEL_PATH, "model checkpoint"),
        (DATA_PATH, "dataset"),
        (TEST_LIST, "test list"),
        (SAM_MASK_DIR, "SAM mask dir"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev

    cmd = [
        sys.executable, INFER_SCRIPT,
        "--model_path", MODEL_PATH,
        "--model_type", "UNet",
        "--test_dataset_path", DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir", OUTPUT_DIR,
        "--sam_mask_dir", SAM_MASK_DIR,
        "--use_sgr",
        "--sgr_hidden_dim", "32",
        "--device", DEVICE,
    ]

    print("\nRunning UNet + SGR inference...\n")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, env=env, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
