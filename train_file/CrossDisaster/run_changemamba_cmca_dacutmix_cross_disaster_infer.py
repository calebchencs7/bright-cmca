# run_changemamba_cmca_dacutmix_cross_disaster_infer.py
# ChangeMamba-CMCA + DACutMix 0.3 cross-disaster inference.
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
HOLDOUT_TYPE = os.environ.get("BRIGHT_HOLDOUT_TYPE", "earthquake").lower()

DATA_PATH = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(
    BDA_ROOT, "dataset", "splitname", "leave_one_disaster_type", HOLDOUT_TYPE
)
TEST_LIST = os.path.join(SPLIT_DIR, "test_set.txt")
INFER_SCRIPT = os.path.join(BDA_ROOT, "script", "standard_ML", "infer_UNet.py")

RUN_FOLDER = os.environ.get("BRIGHT_RUN_FOLDER", "<run_folder>")
MODEL_PATH = os.environ.get(
    "BRIGHT_MODEL_PATH",
    os.path.join(
        ROOT,
        "checkpoints",
        "cross_disaster",
        "changemamba_cmca_dacutmix_0.3",
        HOLDOUT_TYPE,
        "BRIGHT",
        RUN_FOLDER,
        "best_model.pth",
    ),
)
OUTPUT_DIR = os.environ.get(
    "BRIGHT_INFER_OUTPUT_DIR",
    os.path.join(
        ROOT,
        "infer_results",
        "cross_disaster",
        "changemamba_cmca_dacutmix_0.3",
        HOLDOUT_TYPE,
        RUN_FOLDER,
    ),
)
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
            "Set BRIGHT_RUN_FOLDER or BRIGHT_MODEL_PATH to best_model.pth."
        )
    if not os.path.isfile(TEST_LIST):
        raise FileNotFoundError(f"Cross-disaster test split not found:\n  {TEST_LIST}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run([
        sys.executable, INFER_SCRIPT,
        "--model_path", MODEL_PATH,
        "--model_type", "ChangeMambaCMCA",
        "--test_dataset_path", DATA_PATH,
        "--test_data_list_path", TEST_LIST,
        "--output_dir", OUTPUT_DIR,
        "--modality_mode", "both",
        "--device", DEVICE,
    ])
    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
