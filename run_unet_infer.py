import os
import sys
import subprocess

# ========= 修改这三个 =========

MODEL_PATH = r"D:\Project\haoChen\BRIGHT\checkpoints\sam_guided_unet\unet_smoke.pth\BRIGHT\UNet_20260307_182457\best_model.pth"
DATA_PATH = r"D:\Project\haoChen\BRIGHT\data"
TEST_LIST = r"D:\Project\haoChen\BRIGHT\bda_benchmark\dataset\splitname\standard_ML\test_set.txt"
OUTPUT_DIR = r"D:\Project\haoChen\BRIGHT\infer_results\unet_best_001"

# ===============================

PROJECT_ROOT = r"D:\Project\haoChen\BRIGHT"
BDA_ROOT = os.path.join(PROJECT_ROOT, "bda_benchmark")

def main():

    # 强制把 bda_benchmark 加入 PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = BDA_ROOT

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py"),
        "--model_path", MODEL_PATH,
        "--test_dataset_path", DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir", OUTPUT_DIR
    ]

    print("\nRunning inference...\n")
    print(" ".join(cmd))
    print("\n")

    subprocess.run(cmd, env=env)


if __name__ == "__main__": 
    main()