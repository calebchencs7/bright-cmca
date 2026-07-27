# run_changemamba_cmca_dacutmix_infer.py -- Inference for ChangeMamba-CMCA + DACutMix
# DACutMix is training-time only; inference uses ChangeMambaCMCA.

import os
import subprocess
import sys


ROOT = os.environ.get("BRIGHT_ROOT", r"E:\haoChen\BRIGHT")
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
CHANGEMAMBA_ROOT = os.environ.get(
    "CHANGEMAMBA_ROOT",
    os.path.join(os.path.dirname(ROOT), "ChangeMamba-master"),
)
CHANGEMAMBA_CFG = os.environ.get(
    "CHANGEMAMBA_CFG",
    os.path.join(
        CHANGEMAMBA_ROOT,
        "changedetection",
        "configs",
        "vssm1",
        "vssm_tiny_224_0229flex.yaml",
    ),
)

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(BDA_ROOT, "dataset", "splitname", "standard_ML")
TEST_LIST = os.path.join(SPLIT_DIR, "test_set.txt")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py")

RUN_FOLDER = "<run_folder>"
MODEL_PATH = os.path.join(
    ROOT,
    "checkpoints",
    "changemamba_cmca_dacutmix",
    "BRIGHT",
    RUN_FOLDER,
    "best_model.pth",
)
# MODEL_PATH = r"E:\haoChen\BRIGHT\checkpoints\changemamba_cmca_dacutmix\BRIGHT\ChangeMambaCMCA_DACutMix_YYYYMMDD_HHMMSS\best_model.pth"

OUTPUT_DIR = os.path.join(ROOT, "infer_results", "changemamba_cmca_dacutmix")
DEVICE = os.environ.get("BRIGHT_DEVICE", "cuda:0")


def run(cmd):
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = BDA_ROOT if not prev else BDA_ROOT + os.pathsep + prev
    env["CHANGEMAMBA_ROOT"] = CHANGEMAMBA_ROOT
    env["CHANGEMAMBA_CFG"] = CHANGEMAMBA_CFG
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
        sys.executable, INFER_SCRIPT,
        "--model_path", MODEL_PATH,
        "--model_type", "ChangeMambaCMCA",
        "--test_dataset_path", DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir", OUTPUT_DIR,
        "--device", DEVICE,
    ])
    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
