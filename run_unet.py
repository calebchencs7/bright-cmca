import os
import subprocess

DATA_PATH = r"D:\Project\haoChen\BRIGHT\data"
CKPT_PATH = r"D:\Project\haoChen\BRIGHT\checkpoints\unet_smoke.pth"

os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

cmd = [
    "python",
    "bda_benchmark/script/standard_ML/train_UNet.py",
    "--dataset", "BRIGHT",
    "--train_batch_size", "8",
    "--eval_batch_size", "4",
    "--num_workers", "8", 
    "--crop_size", "640", #each image is cropped to 640x640 for training
    "--max_iters", "800000", # --max_iters 800，000
    "--learning_rate", "1e-4", # learning rate 0.0001
    "--model_type", "UNet",
    "--model_param_path", CKPT_PATH,
    "--train_dataset_path", DATA_PATH,
    "--train_data_list_path", "./bda_benchmark/dataset/splitname/standard_ML/train_set.txt",
    "--val_dataset_path", DATA_PATH,
    "--val_data_list_path", "./bda_benchmark/dataset/splitname/standard_ML/val_set.txt",
    "--test_dataset_path", DATA_PATH,
    "--test_data_list_path", "./bda_benchmark/dataset/splitname/standard_ML/test_set.txt"
]

subprocess.run(cmd)
