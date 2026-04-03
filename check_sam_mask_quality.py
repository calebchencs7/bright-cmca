import argparse
import os
import imageio.v2 as imageio
import numpy as np


def safe_div(a, b, eps=1e-7):
    return a / (b + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default=".png")
    parser.add_argument("--label_suffix", type=str, default=".tif")
    args = parser.parse_args()

    with open(args.data_list_path, "r", encoding="utf-8") as f:
        ids = [x.strip() for x in f if x.strip()]

    TP = FP = FN = TN = 0
    valid_tiles = 0

    for tile_id in ids:
        gt_path = os.path.join(args.dataset_path, "target", tile_id + "_building_damage" + args.label_suffix)
        mask_path = os.path.join(args.mask_dir, tile_id + "_building_mask" + args.mask_suffix)

        if not (os.path.exists(gt_path) and os.path.exists(mask_path)):
            continue

        gt = np.asarray(imageio.imread(gt_path))
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        pred = np.asarray(imageio.imread(mask_path))
        if pred.ndim == 3:
            pred = pred[:, :, 0]

        if gt.shape != pred.shape:
            print(f"[WARN] shape mismatch: {tile_id}, gt={gt.shape}, pred={pred.shape}")
            continue

        # 只在有效标注区域评估；255 是 ignore/unlabeled，排除掉
        valid_mask = (gt != 255)

        # GT 建筑区域定义：标签 > 0 且在有效区域内
        gt_building = (gt > 0) & valid_mask

        # SAM 预测建筑区域定义：mask 像素 > 127 视为建筑，且仅在有效区域内评估
        pred_building = (pred > 127) & valid_mask

        tp = np.logical_and(pred_building, gt_building).sum()
        fp = np.logical_and(pred_building, ~gt_building).sum()
        fn = np.logical_and(~pred_building, gt_building).sum()
        tn = np.logical_and(~pred_building, ~gt_building).sum()

        TP += tp
        FP += fp
        FN += fn
        TN += tn
        valid_tiles += 1

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    iou = safe_div(TP, TP + FP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print(f"Valid tiles: {valid_tiles}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"IoU:       {iou:.4f}")
    print(f"F1:        {f1:.4f}")


if __name__ == "__main__":
    main()
