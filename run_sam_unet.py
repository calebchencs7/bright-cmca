# run_sam_unet.py
import os
import sys
import subprocess

ROOT = r"D:\Project\haoChen\BRIGHT"
DATA_PATH = fr"{ROOT}\data"
SPLIT_DIR = fr"{ROOT}\bda_benchmark\dataset\splitname\standard_ML"

SAM_CKPT = fr"{ROOT}\checkpoints\sam\sam_vit_b_01ec64.pth"
MASK_DIR = fr"{ROOT}\outputs\sam_masks\standard_ML"
SAVE_DIR = fr"{ROOT}\checkpoints\sam_guided_unet"

# ===== 开关区 =====
RUN_SAM_MASK_GEN = False      # True=先生成SAM，False=跳过生成直接训练
OVERWRITE_MASK = False        # 仅在 RUN_SAM_MASK_GEN=True 时有效，OVERWRITE_MASK = True（强制覆盖旧 mask）或 False（只补缺失，断点续跑）
TRAIN_MAX_ITERS = "800000"      # 先1000 smoke test，通了再改成"800000"
# ================

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(SAM_CKPT):
    raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CKPT}")

def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# 1) 可选：生成 train/val/test 的 SAM mask
if RUN_SAM_MASK_GEN:
    for split_file in ["train_set.txt", "val_set.txt", "test_set.txt"]:
        split_path = fr"{SPLIT_DIR}\{split_file}"
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        cmd = [
            sys.executable,
            r"bda_benchmark\script\standard_ML\generate_sam_building_masks.py",
            "--dataset_path", DATA_PATH,
            "--data_list_path", split_path,
            "--output_dir", MASK_DIR,
            "--sam_checkpoint", SAM_CKPT,
            "--sam_model_type", "vit_b",
            "--device", "cuda",     # 如果报CUDA/NMS问题改成 "cpu"
            "--source", "both",
            "--points_per_side", "24",
            "--crop_n_layers", "1",
            "--pred_iou_thresh", "0.86",
            "--stability_score_thresh", "0.92",
            "--max_area_ratio", "0.2",
        ]
        if OVERWRITE_MASK:
            cmd.append("--overwrite")
        run(cmd)

# 2) 训练 SAM-guided UNet（soft 模式）
run([
    sys.executable,
    r"bda_benchmark\script\standard_ML\train_UNet.py",
    "--dataset", "BRIGHT",
    "--train_batch_size", "8",
    "--eval_batch_size", "4",
    "--num_workers", "8",
    "--crop_size", "640",
    "--max_iters", TRAIN_MAX_ITERS,
    "--learning_rate", "1e-4",
    "--model_type", "UNet_SAM_SOFT",
    "--model_param_path", SAVE_DIR,
    "--train_dataset_path", DATA_PATH,
    "--train_data_list_path", fr"{SPLIT_DIR}\train_set.txt",
    "--val_dataset_path", DATA_PATH,
    "--val_data_list_path", fr"{SPLIT_DIR}\val_set.txt",
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", fr"{SPLIT_DIR}\test_set.txt",
    "--sam_mask_dir", MASK_DIR,
    "--sam_mode", "soft"
])
