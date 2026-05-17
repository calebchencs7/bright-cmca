"""
SAM Building Mask Generator
============================
Generate high-quality building masks from VHR satellite imagery using SAM.

Key design decisions (v2 rewrite):
    1. Shape-aware filtering: reject elongated strips (roads), low-solidity blobs
    2. SAM confidence filtering: use predicted_iou & stability_score from SAM
    3. Smart bi-temporal merge: pre-disaster primary + post-disaster supplement
       with intersection-based fusion (not naive union)
    4. Mandatory morphological post-processing: open/close + connected-component
       filtering always enabled by default
    5. Tighter area bounds: max_area_ratio default 0.05 (not 0.6)

Usage:
    python generate_sam_building_masks.py \
        --dataset_path /path/to/BRIGHT/data \
        --data_list_path /path/to/train_set.txt \
        --output_dir /path/to/sam_masks \
        --sam_checkpoint /path/to/sam_vit_b.pth \
        --source both
"""

import argparse
import os

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_rgb(path):
    """Load image as uint8 RGB (H, W, 3)."""
    image = np.asarray(imageio.imread(path))
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def normalize_device(device_arg, torch_module):
    """Normalize device string to a valid torch device."""
    if device_arg is None:
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    norm = device_arg.strip().lower()
    if norm in ("gpu", "cuda:0", "cuda0"):
        norm = "cuda"
    if norm in ("auto", "default"):
        norm = "cuda" if torch_module.cuda.is_available() else "cpu"
    if norm.startswith("cuda") and not torch_module.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        norm = "cpu"
    return norm


# ---------------------------------------------------------------------------
# Mask filtering — shape + confidence aware
# ---------------------------------------------------------------------------

def compute_mask_shape_features(segmentation):
    """
    Compute shape descriptors for a single binary mask.

    Returns dict with:
        - aspect_ratio: max(w/h, h/w) of bounding box — buildings are usually < 5
        - solidity: area / convex_hull_area — buildings are usually > 0.5
        - extent: area / bbox_area — buildings are usually > 0.3
        - compactness: 4*pi*area / perimeter^2 — circles=1, elongated→0
    """
    mask_uint8 = segmentation.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"aspect_ratio": 999, "solidity": 0, "extent": 0, "compactness": 0}

    # Use the largest contour
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 1:
        return {"aspect_ratio": 999, "solidity": 0, "extent": 0, "compactness": 0}

    # Bounding box
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
    extent = area / (bw * bh + 1e-6)

    # Convex hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    # Compactness (isoperimetric quotient)
    perimeter = cv2.arcLength(cnt, True)
    compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

    return {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "compactness": compactness,
    }


def filter_masks(
    mask_items,
    total_area,
    min_area_ratio=0.0002,
    max_area_ratio=0.05,
    min_predicted_iou=0.80,
    min_stability_score=0.85,
    max_aspect_ratio=5.0,
    min_solidity=0.4,
    min_extent=0.25,
    min_compactness=0.05,
    top_k=0,
):
    """
    Filter SAM mask proposals using area, confidence, AND shape features.

    This is the critical function that determines mask quality. SAM generates
    masks for ALL objects (roads, fields, water, etc). We need building-specific
    filters.

    Filtering pipeline:
        1. Area range: reject too-small noise and too-large non-buildings
        2. SAM confidence: use SAM's own predicted_iou and stability_score
        3. Shape: reject elongated strips (roads), irregular shapes (vegetation)
    """
    selected = []
    for item in mask_items:
        area = float(item["area"])
        area_ratio = area / total_area

        # --- Stage 1: Area filter ---
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # --- Stage 2: SAM confidence filter ---
        pred_iou = item.get("predicted_iou", 1.0)
        stab_score = item.get("stability_score", 1.0)
        if pred_iou < min_predicted_iou or stab_score < min_stability_score:
            continue

        # --- Stage 3: Shape filter ---
        shape = compute_mask_shape_features(item["segmentation"])
        if shape["aspect_ratio"] > max_aspect_ratio:
            continue
        if shape["solidity"] < min_solidity:
            continue
        if shape["extent"] < min_extent:
            continue
        if shape["compactness"] < min_compactness:
            continue

        selected.append(item)

    # Sort by area (largest first) and optionally keep top-k
    selected.sort(key=lambda x: x["area"], reverse=True)
    if top_k > 0:
        selected = selected[:top_k]

    return selected


# ---------------------------------------------------------------------------
# Mask merging
# ---------------------------------------------------------------------------

def merge_masks(mask_items, height, width):
    """Merge selected mask proposals into a single binary mask."""
    merged = np.zeros((height, width), dtype=np.uint8)
    for item in mask_items:
        merged[item["segmentation"]] = 1
    return merged


# ---------------------------------------------------------------------------
# Morphological post-processing
# ---------------------------------------------------------------------------

def postprocess_mask(
    mask,
    morph_open=3,
    morph_close=3,
    cc_min_area=100,
    cc_max_area_ratio=0.05,
    cc_max_elongation=5.0,
    cc_min_extent=0.2,
):
    """
    Clean binary mask with morphology + connected-component filtering.

    This is always applied (not optional) to remove residual noise after
    SAM filtering. Two stages:
        1. Morphological open: remove thin bridges and small noise
        2. Morphological close: fill small holes in buildings
        3. Connected-component analysis: reject components with wrong shape
    """
    bin_mask = (mask > 0).astype(np.uint8)

    # Morphological opening — remove thin noise/bridges
    if morph_open > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)

    # Morphological closing — fill small holes in buildings
    if morph_close > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k_close)

    # Connected-component filtering
    h, w = bin_mask.shape
    max_area = int(max(1, cc_max_area_ratio * h * w))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    filtered = np.zeros_like(bin_mask, dtype=np.uint8)

    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]

        # Area filter
        if area < cc_min_area or area > max_area:
            continue

        # Elongation filter (reject road-like strips)
        elongation = max(bw / (bh + 1e-6), bh / (bw + 1e-6))
        if elongation > cc_max_elongation:
            continue

        # Extent filter (area / bbox_area — reject irregular blobs)
        extent = area / (bw * bh + 1e-6)
        if extent < cc_min_extent:
            continue

        filtered[labels == idx] = 1

    return filtered


# ---------------------------------------------------------------------------
# Per-tile mask generation
# ---------------------------------------------------------------------------

def generate_for_tile(
    mask_generator,
    dataset_path,
    tile_id,
    image_suffix,
    source,
    filter_kwargs,
    verbose=False,
):
    """
    Generate building mask for one tile.

    Bi-temporal merge strategy:
        - "pre":  use pre-disaster image only (buildings are intact, clearest)
        - "post": use post-disaster image only
        - "both": use pre-disaster as PRIMARY, post-disaster as SUPPLEMENT.
                   Final mask = pre_mask UNION (post_mask AND pre_mask_dilated)
                   Dilation is kept small (7px) to avoid amplifying pre_mask errors.
                   Rationale: destroyed buildings may fragment into many small
                   pieces in post-disaster imagery — we only keep post detections
                   that are near confirmed pre-disaster building locations.
    """
    def _generate_single(image_path):
        if not os.path.exists(image_path):
            return None, 0, 0
        image = load_rgb(image_path)
        h, w = image.shape[:2]
        proposals = mask_generator.generate(image)
        selected = filter_masks(
            mask_items=proposals,
            total_area=float(h * w),
            **filter_kwargs,
        )
        if verbose:
            print(f"    {os.path.basename(image_path)}: "
                  f"{len(proposals)} proposals → {len(selected)} passed filter")
        return merge_masks(selected, height=h, width=w), len(proposals), len(selected)

    pre_path = os.path.join(dataset_path, "pre-event", tile_id + "_pre_disaster" + image_suffix)
    post_path = os.path.join(dataset_path, "post-event", tile_id + "_post_disaster" + image_suffix)

    if source == "pre":
        mask, _, _ = _generate_single(pre_path)
        return mask

    if source == "post":
        mask, _, _ = _generate_single(post_path)
        return mask

    # source == "both": smart merge (pre-disaster is primary)
    pre_mask, pre_total, pre_kept = _generate_single(pre_path)
    post_mask, post_total, post_kept = _generate_single(post_path)

    if pre_mask is None and post_mask is None:
        return None
    if pre_mask is None:
        return post_mask
    if post_mask is None:
        return pre_mask

    # Smart merge: pre-disaster is primary; post-disaster supplements only
    # where it overlaps with SLIGHTLY dilated pre-disaster mask.
    # Dilation kept to 7px (was 15px which was too aggressive and amplified errors).
    # Larger dilation → more post-disaster noise bleeds through.
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pre_dilated = cv2.dilate(pre_mask, dilation_kernel, iterations=1)

    # Post-disaster mask is only kept where it's near pre-disaster buildings
    post_filtered = np.logical_and(post_mask > 0, pre_dilated > 0).astype(np.uint8)

    # Final = union of pre-mask and filtered post-mask
    merged = np.maximum(pre_mask, post_filtered)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality SAM building masks for BRIGHT dataset."
    )

    # Data paths
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # SAM model
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
                        choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", type=str, default="auto")

    # Image source
    parser.add_argument("--source", type=str, default="both",
                        choices=["pre", "post", "both"],
                        help="Which images to use. 'both' uses smart merge.")
    parser.add_argument("--image_suffix", type=str, default=".tif")
    parser.add_argument("--mask_suffix", type=str, default=".png")

    # SAM generator parameters
    # NOTE: pred_iou_thresh and stability_score_thresh here are the VALUES PASSED
    # TO filter_masks (the actual building filter). The SAM generator internal
    # thresholds are set automatically to be 0.05 lower, so that more candidates
    # enter the pipeline and the building-specific filter can do real work.
    parser.add_argument("--points_per_side", type=int, default=48,
                        help="Grid density for auto-prompting. 32→misses small buildings. "
                             "48 is a good balance. 64 for maximum recall (slower).")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86,
                        help="Target predicted_iou threshold for filter_masks. "
                             "SAM generator uses (this - 0.05) internally.")
    parser.add_argument("--stability_score_thresh", type=float, default=0.90,
                        help="Target stability_score threshold for filter_masks. "
                             "SAM generator uses (this - 0.05) internally.")
    parser.add_argument("--crop_n_layers", type=int, default=1,
                        help="Number of crop layers for multi-scale generation. "
                             "1=adds one crop pass, 2=more thorough (much slower).")
    parser.add_argument("--crop_n_points_downscale_factor", type=int, default=2)
    parser.add_argument("--min_mask_region_area", type=int, default=100,
                        help="SAM internal: min connected region to keep. "
                             "200 is too high and kills small buildings. Use 100.")

    # Building-specific filtering
    parser.add_argument("--min_area_ratio", type=float, default=0.0002,
                        help="Min mask area as fraction of image. Default: 0.02%%")
    parser.add_argument("--max_area_ratio", type=float, default=0.08,
                        help="Max mask area as fraction of image. "
                             "5%% is too tight for large buildings (factories, warehouses). "
                             "8%% is safer. Default: 8%%")
    parser.add_argument("--min_predicted_iou", type=float, default=0.86,
                        help="Min SAM predicted IoU to keep a mask. Must be > SAM generator internal.")
    parser.add_argument("--min_stability_score", type=float, default=0.90,
                        help="Min SAM stability score to keep a mask. Must be > SAM generator internal.")
    parser.add_argument("--max_aspect_ratio", type=float, default=4.0,
                        help="Max bbox aspect ratio. Roads/strips > 5. Buildings < 4. Default: 4.")
    parser.add_argument("--min_solidity", type=float, default=0.55,
                        help="Min solidity (area/convex_hull). "
                             "0.4 is too loose (lets in irregular vegetation). "
                             "Buildings are typically > 0.55. Default: 0.55")
    parser.add_argument("--min_extent", type=float, default=0.38,
                        help="Min extent (area/bbox_area). "
                             "0.25 is too loose. Buildings are typically > 0.38. Default: 0.38")
    parser.add_argument("--min_compactness", type=float, default=0.12,
                        help="Min compactness (4*pi*area/perimeter^2). "
                             "0.05 lets in almost everything. Buildings > 0.12. Default: 0.12")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Keep top-k largest masks per image. 0 = keep all.")

    # Post-processing (always on by default now)
    parser.add_argument("--morph_open", type=int, default=3,
                        help="Morphological open kernel size. 0 disables.")
    parser.add_argument("--morph_close", type=int, default=5,
                        help="Morphological close kernel size. 0 disables.")
    parser.add_argument("--cc_min_area", type=int, default=100,
                        help="Min connected-component area in final mask.")
    parser.add_argument("--cc_max_area_ratio", type=float, default=0.05,
                        help="Max connected-component area ratio in final mask.")
    parser.add_argument("--cc_max_elongation", type=float, default=5.0,
                        help="Max elongation in connected-component filtering.")
    parser.add_argument("--cc_min_extent", type=float, default=0.2,
                        help="Min extent in connected-component filtering.")

    # Control
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Disable morphological post-processing (not recommended).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing mask files.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-tile proposal counts for debugging filter effectiveness. "
                             "Use this to diagnose: if 'passed filter' is always 0, filters are too strict. "
                             "If always > 50, filters are too loose.")

    args = parser.parse_args()

    # --- Import torch and SAM ---
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment-anything and torch are required. "
            "Install with: pip install segment-anything"
        ) from exc

    os.makedirs(args.output_dir, exist_ok=True)
    device = normalize_device(args.device, torch)

    # --- Build SAM mask generator ---
    # IMPORTANT: SAM generator internal thresholds are intentionally set LOWER
    # than filter_masks thresholds. This lets more candidates through so that
    # filter_masks can do the real building-specific filtering.
    # If SAM generator thresholds >= filter_masks thresholds, the filter is useless.
    sam_internal_iou_thresh = min(args.pred_iou_thresh, args.min_predicted_iou - 0.05)
    sam_internal_stab_thresh = min(args.stability_score_thresh, args.min_stability_score - 0.05)
    sam_internal_iou_thresh = max(0.5, sam_internal_iou_thresh)
    sam_internal_stab_thresh = max(0.5, sam_internal_stab_thresh)

    print(f"  SAM generator internal thresholds: iou={sam_internal_iou_thresh:.2f}, "
          f"stability={sam_internal_stab_thresh:.2f}")
    print(f"  filter_masks thresholds: iou={args.min_predicted_iou:.2f}, "
          f"stability={args.min_stability_score:.2f}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=sam_internal_iou_thresh,
        stability_score_thresh=sam_internal_stab_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )

    # --- Filter kwargs (passed to filter_masks) ---
    filter_kwargs = {
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "min_predicted_iou": args.min_predicted_iou,
        "min_stability_score": args.min_stability_score,
        "max_aspect_ratio": args.max_aspect_ratio,
        "min_solidity": args.min_solidity,
        "min_extent": args.min_extent,
        "min_compactness": args.min_compactness,
        "top_k": args.top_k,
    }

    # --- Load tile list ---
    with open(args.data_list_path, "r") as f:
        tile_ids = [line.strip() for line in f if line.strip()]

    print(f"Generating SAM building masks for {len(tile_ids)} tiles")
    print(f"  source={args.source}, device={device}")
    print(f"  max_area_ratio={args.max_area_ratio}, min_predicted_iou={args.min_predicted_iou}")
    print(f"  max_aspect_ratio={args.max_aspect_ratio}, min_solidity={args.min_solidity}")
    print(f"  postprocess={'OFF' if args.no_postprocess else 'ON'}")

    # --- Generate ---
    processed = 0
    skipped = 0
    failed = 0

    for tile_id in tqdm(tile_ids, desc="Generating SAM masks"):
        output_path = os.path.join(
            args.output_dir, tile_id + "_building_mask" + args.mask_suffix
        )
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            building_mask = generate_for_tile(
                mask_generator=mask_generator,
                dataset_path=args.dataset_path,
                tile_id=tile_id,
                image_suffix=args.image_suffix,
                source=args.source,
                filter_kwargs=filter_kwargs,
                verbose=args.verbose,
            )

            if building_mask is None:
                skipped += 1
                continue

            # Always post-process unless explicitly disabled
            if not args.no_postprocess:
                building_mask = postprocess_mask(
                    building_mask,
                    morph_open=args.morph_open,
                    morph_close=args.morph_close,
                    cc_min_area=args.cc_min_area,
                    cc_max_area_ratio=args.cc_max_area_ratio,
                    cc_max_elongation=args.cc_max_elongation,
                    cc_min_extent=args.cc_min_extent,
                )

            imageio.imwrite(output_path, (building_mask * 255).astype(np.uint8))
            processed += 1

        except Exception as exc:
            failed += 1
            print(f"[ERROR] {tile_id}: {exc}")

    print(f"\nFinished: processed={processed}, skipped={skipped}, failed={failed}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
