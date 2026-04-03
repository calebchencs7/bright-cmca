import argparse
import os

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def load_rgb(path):
    image = np.asarray(imageio.imread(path))
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def normalize_device(device_arg, torch_module):
    if device_arg is None:
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    norm = device_arg.strip().lower()
    if norm in ["gpu", "cuda:0", "cuda0"]:
        norm = "cuda"
    if norm in ["auto", "default"]:
        norm = "cuda" if torch_module.cuda.is_available() else "cpu"
    if norm.startswith("cuda") and (not torch_module.cuda.is_available()):
        print("[WARN] CUDA is not available. Falling back to CPU.")
        norm = "cpu"
    return norm


def filter_masks(mask_items, total_area, min_area_ratio, max_area_ratio, top_k):
    selected = []
    for mask_item in mask_items:
        area = float(mask_item["area"])
        area_ratio = area / total_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        selected.append(mask_item)

    selected.sort(key=lambda x: x["area"], reverse=True)
    if top_k > 0:
        selected = selected[:top_k]
    return selected


def merge_masks(mask_items, height, width):
    merged = np.zeros((height, width), dtype=np.uint8)
    for mask_item in mask_items:
        merged[mask_item["segmentation"]] = 1
    return merged


def postprocess_mask(mask, cc_min_area, cc_max_area_ratio, cc_max_elongation, cc_min_extent, morph_open, morph_close):
    """
    Post-process SAM binary mask using connected-components and simple shape filters.
    This stage helps remove false positives like roads/strips and oversized non-building blobs.
    """
    bin_mask = (mask > 0).astype(np.uint8)

    if morph_open > 0:
        k_open = np.ones((morph_open, morph_open), dtype=np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)

    if morph_close > 0:
        k_close = np.ones((morph_close, morph_close), dtype=np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k_close)

    h, w = bin_mask.shape
    max_area = int(max(1, cc_max_area_ratio * h * w))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    filtered = np.zeros_like(bin_mask, dtype=np.uint8)

    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < cc_min_area or area > max_area:
            continue

        elongation = max(bw / float(bh + 1e-6), bh / float(bw + 1e-6))
        if elongation > cc_max_elongation:
            continue

        extent = area / float((bw * bh) + 1e-6)
        if extent < cc_min_extent:
            continue

        filtered[labels == idx] = 1

    return filtered


def generate_for_tile(mask_generator, dataset_path, tile_id, image_suffix, source, min_area_ratio, max_area_ratio, top_k):
    image_paths = []
    if source in ("pre", "both"):
        image_paths.append(os.path.join(dataset_path, "pre-event", tile_id + "_pre_disaster" + image_suffix))
    if source in ("post", "both"):
        image_paths.append(os.path.join(dataset_path, "post-event", tile_id + "_post_disaster" + image_suffix))

    merged_mask = None
    for image_path in image_paths:
        if not os.path.exists(image_path):
            continue
        image = load_rgb(image_path)
        height, width = image.shape[:2]
        proposals = mask_generator.generate(image)
        selected = filter_masks(
            mask_items=proposals,
            total_area=float(height * width),
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            top_k=top_k
        )
        current_mask = merge_masks(selected, height=height, width=width)
        if merged_mask is None:
            merged_mask = current_mask
        else:
            merged_mask = np.maximum(merged_mask, current_mask)

    return merged_mask


def main():
    parser = argparse.ArgumentParser(description="Generate building masks with Segment Anything.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu. 'gpu' is also accepted.")
    parser.add_argument("--source", type=str, default="pre", choices=["pre", "post", "both"])
    parser.add_argument("--image_suffix", type=str, default=".tif")
    parser.add_argument("--mask_suffix", type=str, default=".png")
    parser.add_argument("--points_per_side", type=int, default=24)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    parser.add_argument("--crop_n_layers", type=int, default=1)
    parser.add_argument("--crop_n_points_downscale_factor", type=int, default=2)
    parser.add_argument("--min_mask_region_area", type=int, default=200)
    parser.add_argument("--min_area_ratio", type=float, default=0.0001)
    parser.add_argument("--max_area_ratio", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=0, help="Keep top-k largest masks per image. 0 means keep all.")
    parser.add_argument("--postprocess", action="store_true",
                        help="Enable connected-component and morphology filtering on SAM masks.")
    parser.add_argument("--cc_min_area", type=int, default=120,
                        help="Minimum connected-component area to keep.")
    parser.add_argument("--cc_max_area_ratio", type=float, default=0.03,
                        help="Maximum connected-component area ratio to keep.")
    parser.add_argument("--cc_max_elongation", type=float, default=5.0,
                        help="Maximum elongation (max(w/h, h/w)) to keep components.")
    parser.add_argument("--cc_min_extent", type=float, default=0.18,
                        help="Minimum component extent (area / bbox_area).")
    parser.add_argument("--morph_open", type=int, default=3,
                        help="Morphology open kernel size; 0 disables.")
    parser.add_argument("--morph_close", type=int, default=3,
                        help="Morphology close kernel size; 0 disables.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing mask files instead of skipping them.")
    args = parser.parse_args()

    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except Exception as exc:
        raise ImportError(
            "segment-anything and torch are required. "
            "Install with: pip install segment-anything"
        ) from exc

    os.makedirs(args.output_dir, exist_ok=True)

    device = normalize_device(args.device, torch)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area
    )

    with open(args.data_list_path, "r") as f:
        tile_ids = [line.strip() for line in f if line.strip()]

    processed = 0
    skipped = 0
    failed = 0
    for tile_id in tqdm(tile_ids, desc="Generating SAM masks"):
        output_path = os.path.join(args.output_dir, tile_id + "_building_mask" + args.mask_suffix)
        if os.path.exists(output_path) and (not args.overwrite):
            skipped += 1
            continue

        try:
            building_mask = generate_for_tile(
                mask_generator=mask_generator,
                dataset_path=args.dataset_path,
                tile_id=tile_id,
                image_suffix=args.image_suffix,
                source=args.source,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                top_k=args.top_k
            )
            if building_mask is None:
                skipped += 1
                continue

            if args.postprocess:
                building_mask = postprocess_mask(
                    building_mask,
                    cc_min_area=args.cc_min_area,
                    cc_max_area_ratio=args.cc_max_area_ratio,
                    cc_max_elongation=args.cc_max_elongation,
                    cc_min_extent=args.cc_min_extent,
                    morph_open=args.morph_open,
                    morph_close=args.morph_close
                )

            imageio.imwrite(output_path, (building_mask * 255).astype(np.uint8))
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {tile_id}: {exc}")

    print(f"Finished. processed={processed}, skipped={skipped}, failed={failed}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
