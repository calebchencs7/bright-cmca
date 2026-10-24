import os
import sys
import subprocess

ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")

MODEL_PATH = os.path.join(
    ROOT,
    "checkpoints",
    "unet",
    "BRIGHT",
    "<run_folder>",
    "best_model.pth",
)
DATA_PATH = os.path.join(ROOT, "data")
TEST_LIST = os.path.join(SPLIT_DIR, "test_set.txt")
OUTPUT_DIR = os.path.join(ROOT, "infer_results", "unet_baseline")
DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")

PROJECT_ROOT = ROOT


def main():

    # 强制把 bda_benchmark 加入 PYTHONPATH
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py"),
        "--model_path",
        MODEL_PATH,
        "--test_dataset_path",
        DATA_PATH,
        "--test_data_list_path",
        TEST_LIST,
        "--output_dir",
        OUTPUT_DIR,
        "--device",
        DEVICE,
    ]

    print("\nRunning inference...\n")
    print(" ".join(cmd))
    print("\n")

    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
