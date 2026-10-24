"""Match BRIGHT inference outputs, browse samples, and export a paper figure.

Run on the machine containing the data; see paper_result_gallery.md.
"""
import argparse
import csv
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = [
    ("UNet", "unet_cmca_dacutmix"),
    ("DeepLabV3+", "deeplabv3plus_cmca_dacutmix"),
    ("SiamCRNN", "siamcrnn_cmca_dacutmix"),
    ("SiamAttnUNet", "siamattnunet_cmca_dacutmix"),
    ("DamageFormer", "damageformer_cmca_dacutmix"),
    ("ChangeMamba", "changemamba_cmca_dacutmix"),
]
PALETTE = np.array([[255, 255, 255], [70, 181, 121],
                    [228, 189, 139], [182, 70, 69]], dtype=np.uint8)
SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def index_folder(folder, suffix):
    result = {}
    if not folder.is_dir():
        return result
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in SUFFIXES or not path.stem.endswith(suffix):
            continue
        relative = path.relative_to(folder).with_suffix("").as_posix()
        key = relative[:-len(suffix)]
        if key in result:
            raise ValueError(f"Duplicate sample {key}: {result[key]} and {path}")
        result[key] = path
    return result


def read_image(path, label=False):
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        arr = tifffile.imread(path)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB") if im.mode == "P" else im)
    if label:
        if arr.ndim == 2:
            if not np.isin(arr, range(4)).all():
                raise ValueError(f"Unknown label IDs in {path}; expected 0,1,2,3")
            arr = PALETTE[arr.astype(np.uint8)]
        else:
            arr = arr[..., :3].astype(np.uint8)
    else:
        if arr.ndim == 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            valid = arr[np.isfinite(arr)]
            if not valid.size:
                raise ValueError(f"No finite image values: {path}")
            low, high = np.percentile(valid, [2, 98])
            arr = (np.clip((arr - low) / max(high-low, 1e-8), 0, 1)*255)
            arr = np.nan_to_num(arr).astype(np.uint8)
    return Image.fromarray(arr)


def panels_for(sample, indexes):
    panels = [read_image(index[sample], label=(i >= 2))
              for i, index in enumerate(indexes)]
    if len({im.size for im in panels}) != 1:
        raise ValueError(f"Different image sizes for {sample}: {[im.size for im in panels]}")
    return panels


def read_classes(path):
    """Decode IDs or exact project RGB colors; never guess another palette."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        arr = tifffile.imread(path)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB") if im.mode == "P" else im)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        if not np.isin(arr, range(4)).all():
            raise ValueError(f"Unknown class IDs: {path}; expected 0–3")
        return arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Unsupported label shape: {path}: {arr.shape}")
    rgb = arr[..., :3]
    decoded = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for cls, color in enumerate(PALETTE):
        decoded[np.all(rgb == color, axis=-1)] = cls
    if (decoded == 255).any():
        raise ValueError(f"Unknown RGB palette: {path}. Use original class maps or configure PALETTE.")
    return decoded


def model_scores(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError(f"Label shape mismatch: {gt.shape} vs {pred.shape}")
    confusion = np.bincount((gt.astype(np.int64)*4+pred).ravel(), minlength=16).reshape(4, 4)
    union = confusion.sum(0)+confusion.sum(1)-confusion.diagonal()
    ious = np.divide(confusion.diagonal(), union, out=np.full(4, np.nan), where=union > 0)
    foreground = float(np.nanmean(ious[1:]))
    damage = float(np.nanmean(ious[2:]))
    return foreground, damage, ious


def rank_samples(output, samples, indexes, titles, args):
    records, failures = [], []
    for n, sample in enumerate(samples):
        try:
            gt = read_classes(indexes[2][sample])
            event = sample.split("/", 1)[0] if "/" in sample else sample.rsplit("_", 1)[0]
            record = {"sample_id": sample, "event": event,
                      "building_pixels": int((gt > 0).sum()),
                      "damaged_pixels": int((gt >= 2).sum()),
                      "damage_classes": ",".join(str(c) for c in (2, 3) if (gt == c).any())}
            reason = []
            if record["building_pixels"] < args.min_building_pixels:
                reason.append("too_few_building_pixels")
            if record["damaged_pixels"] < args.min_damage_pixels:
                reason.append("too_few_damage_pixels")
            record["excluded_reason"] = ";".join(reason)
            if reason:
                records.append(record)
                continue
            scores = []
            for title, index in zip(titles[3:], indexes[3:]):
                foreground, damage, ious = model_scores(gt, read_classes(index[sample]))
                record[title+"_foreground_mIoU"] = foreground
                record[title+"_damage_mIoU"] = damage
                for cls in (1, 2, 3):
                    record[f"{title}_IoU_class_{cls}"] = float(ious[cls])
                scores.append(0.5*foreground+0.5*damage)
            record["mean_model_score"] = float(np.mean(scores))
            record["worst_model_score"] = float(np.min(scores))
            record["selection_score"] = 0.7*np.mean(scores)+0.3*np.min(scores)
            records.append(record)
        except (ValueError, OSError) as exc:
            failures.append({"sample_id": sample, "error": str(exc)})
        finally:
            if (n+1) % 50 == 0:
                print(f"Scored: {n+1}/{len(samples)}")
    eligible = sorted([r for r in records if not r["excluded_reason"]],
                      key=lambda r: (-r["selection_score"], r["sample_id"]))
    ordered = eligible + [r for r in records if r["excluded_reason"]]
    fields = list(dict.fromkeys(key for record in ordered for key in record))
    with (output / "sample_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["sample_id"])
        writer.writeheader()
        writer.writerows(ordered)
    (output / "ranking_errors.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    groups = {}
    for record in eligible:
        groups.setdefault(record["event"], []).append(record)
    recommended, typical = [], []
    for event in sorted(groups):
        group = groups[event]
        recommended.extend(r["sample_id"] for r in group[:args.per_event])
        typical.append(group[len(group)//2]["sample_id"])
    for name, ids in [("recommended.txt", recommended), ("typical.txt", typical)]:
        (output / name).write_text("\n".join(ids)+("\n" if ids else ""), encoding="utf-8")
    print(f"Eligible: {len(eligible)}; excluded: {len(records)-len(eligible)}; errors: {len(failures)}")
    if not eligible:
        raise ValueError("No eligible samples. Inspect sample_scores.csv / ranking_errors.json; lower thresholds if needed.")
    notes = {r["sample_id"]: f"score={r['selection_score']:.3f}; worst model={r['worst_model_score']:.3f}; "
             f"damage pixels={r['damaged_pixels']}" for r in eligible}
    preview_ids = list(dict.fromkeys(recommended+typical))
    gallery(output, preview_ids, indexes, titles, notes)


def gallery(output, samples, indexes, titles, notes=None):
    thumbs = output / "previews"
    thumbs.mkdir(parents=True, exist_ok=True)
    cards = []
    errors = []
    for n, sample in enumerate(samples):
        try:
            panels = panels_for(sample, indexes)
            sheet = Image.new("RGB", (9*180, 180), "white")
            for col, im in enumerate(panels):
                im.thumbnail((176, 176), Image.Resampling.NEAREST if col >= 2
                             else Image.Resampling.LANCZOS)
                sheet.paste(im, (col*180+(180-im.width)//2, (180-im.height)//2))
            name = f"previews/{n:06d}.png"
            sheet.save(output / name)
            escaped = html.escape(sample, quote=True)
            cards.append(f'<article data-id="{escaped}"><label><input type="checkbox" '
                         f'value="{escaped}"> {escaped}</label> '
                         + html.escape((notes or {}).get(sample, "")) + '<div class="titles">'
                         + "".join(f"<span>{html.escape(t)}</span>" for t in titles)
                         + f'</div><a href="{name}" target="_blank"><img loading="lazy" '
                         f'src="{name}"></a></article>')
        except (ValueError, OSError) as exc:
            errors.append({"sample_id": sample, "error": str(exc)})
        if (n+1) % 50 == 0:
            print(f"Previews: {n+1}/{len(samples)}")
    page = '''<!doctype html><meta charset="utf-8"><title>BRIGHT sample selection</title>
<style>body{font:15px Arial;margin:20px}header{position:sticky;top:0;background:white;
padding:12px;z-index:1}article{margin:24px 0;border-bottom:1px solid #ccc;padding-bottom:12px}
img{width:100%}.titles{display:grid;grid-template-columns:repeat(9,1fr);text-align:center;
font-size:12px;margin:8px 0}button,input{padding:8px}label{font-weight:bold}</style>
<header><b>BRIGHT: CMCA + DACutMix</b> · <span id="count">0 selected</span>
<button onclick="download()">Download selected.txt</button>
<input placeholder="Filter sample ID / event" oninput="filter(this.value)">
<p>Select representative samples; click a preview to enlarge. Order follows this gallery.</p>
</header>''' + "\n".join(cards) + '''<script>
document.addEventListener('change',()=>document.getElementById('count').textContent=
document.querySelectorAll('input[type=checkbox]:checked').length+' selected');
function filter(q){document.querySelectorAll('article').forEach(e=>e.hidden=
!e.dataset.id.toLowerCase().includes(q.toLowerCase()))}
function download(){const ids=[...document.querySelectorAll('input[type=checkbox]:checked')]
.map(e=>e.value);if(!ids.length){alert('Select samples first');return}
const url=URL.createObjectURL(new Blob([ids.join('\\n')+'\\n'],{type:'text/plain'}));
const a=document.createElement('a');a.href=url;a.download='selected.txt';a.click();
setTimeout(()=>URL.revokeObjectURL(url),1000)}
</script>'''
    (output / "gallery.html").write_text(page, encoding="utf-8")
    (output / "preview_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"Gallery: {output / 'gallery.html'} ({len(cards)} samples; {len(errors)} errors)")


def disaster_label(sample):
    event = sample.split("/", 1)[0] if "/" in sample else sample.rsplit("_", 1)[0]
    kind = event.rsplit("-", 1)[-1].lower()
    labels = {"earthquake": "Earthquake", "wildfire": "Wildfire", "fire": "Fire",
              "volcano": "Volcano", "hurricane": "Hurricane", "flood": "Flood",
              "flooding": "Flooding", "tsunami": "Tsunami", "explosion": "Explosion",
              "conflict": "Conflict"}
    if kind not in labels:
        raise ValueError(f"Cannot infer disaster type for {sample}; supply --row-labels JSON")
    return labels[kind]


def archive_selection(output, export, samples, indexes, titles, labels, args):
    score_path = output / "sample_scores.csv"
    scores = {}
    if score_path.exists():
        with score_path.open(encoding="utf-8-sig", newline="") as handle:
            scores = {r["sample_id"]: r for r in csv.DictReader(handle)}
    records = []
    for row, (sample, label) in enumerate(zip(samples, labels), start=1):
        files = []
        for col, (title, index) in enumerate(zip(titles, indexes)):
            source = index[sample]
            # Keep original bytes, dimensions and class IDs; do not save thumbnails.
            target = export / "selected_data" / f"row_{row:02d}" / f"column_{col+1:02d}" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({"column": col+1, "title": title,
                          "source": str(source.resolve()),
                          "saved_copy": target.relative_to(export).as_posix()})
        records.append({"row": row, "sample_id": sample, "disaster_type": label,
                        "scores": scores.get(sample), "files": files})
    manifest = {"created_at": datetime.now().astimezone().isoformat(),
                "configuration": "CMCA + DACutMix", "models": MODELS if not args.models else
                list(json.loads(args.models.read_text(encoding="utf-8")).items()),
                "crop_xywh": args.crop, "dpi": args.dpi, "palette": PALETTE.tolist(),
                "score_note": "Full-image selection scores, not scores for cropped panels; null if unavailable.",
                "display_note": "Non-uint8 imagery: per-image 2–98 percentile stretch; uint8 unchanged.",
                "samples": records}
    (export / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (export / "selected.txt").write_text("\n".join(samples)+"\n", encoding="utf-8")
    selected_scores = [scores[s] for s in samples if s in scores]
    if selected_scores:
        with (export / "selected_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(scores.values()))))
            writer.writeheader()
            writer.writerows(selected_scores)
    shutil.copy2(Path(__file__), export / "paper_result_gallery.py")
    for name in ("recommended.txt", "typical.txt"):
        if (output / name).exists():
            shutil.copy2(output / name, export / name)


def figure(output, samples, indexes, titles, crop, dpi, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(samples), 9, figsize=(7.16, len(samples)*0.81+0.55),
                             squeeze=False)
    for row, sample in enumerate(samples):
        axes[row, 0].set_ylabel(labels[row], rotation=90, fontsize=8, labelpad=7)
        for col, im in enumerate(panels_for(sample, indexes)):
            if crop:
                x, y, w, h = crop
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > im.width or y+h > im.height:
                    raise ValueError(f"Crop outside image: {sample}")
                im = im.crop((x, y, x+w, y+h))
            axes[row, col].imshow(im, cmap="gray", vmin=0, vmax=255,
                                   interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
            if row == len(samples)-1:
                axes[row, col].set_xlabel(titles[col], labelpad=4)
    fig.legend(handles=[Patch(facecolor=color/255, edgecolor="gray", label=name)
                        for color, name in zip(PALETTE, ["Background", "Intact", "Damaged", "Destroyed"])],
               loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.052, right=0.995, top=0.995, bottom=0.55/fig.get_figheight(),
                        wspace=0.04, hspace=0.04)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"qualitative_results.{suffix}", dpi=dpi, facecolor="white")
    plt.close(fig)
    (output / "figure_samples.txt").write_text("\n".join(samples)+"\n", encoding="utf-8")
    print(f"Figure saved in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gallery", "rank", "figure"])
    parser.add_argument("--root", type=Path, default=Path(r"E:\haoChen\BRIGHT"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selected", type=Path, help="One sample ID per line")
    parser.add_argument("--row-labels", type=Path, help="Optional JSON object: sample ID -> English disaster label")
    parser.add_argument("--models", type=Path, help="JSON object: display title -> inference folder")
    parser.add_argument("--limit", type=int, default=0, help="Limit gallery size; 0 means all")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--per-event", type=int, default=3, help="Top candidates per event in rank mode")
    parser.add_argument("--min-building-pixels", type=int, default=200)
    parser.add_argument("--min-damage-pixels", type=int, default=100)
    args = parser.parse_args()
    if args.per_event < 1 or min(args.min_building_pixels, args.min_damage_pixels) < 1:
        parser.error("Candidate count and pixel thresholds must be positive")
    models = list(json.loads(args.models.read_text(encoding="utf-8")).items()) if args.models else MODELS
    if len(models) != 6:
        parser.error("Exactly six model directories are required")
    data = args.root / "data"
    output = args.out or args.root / "figure" / "paper_selection"
    output.mkdir(parents=True, exist_ok=True)
    gt = index_folder(data / "target", "_building_damage")
    if not gt:
        gt = index_folder(data / "target_colored", "_building_damage")
    indexes = [index_folder(data / "pre-event", "_pre_disaster"),
               index_folder(data / "post-event", "_post_disaster"), gt]
    titles = ["Pre-event\n(optical)", "Post-event\n(SAR)", "Ground truth"]
    for title, folder in models:
        path = args.root / "infer_results" / folder
        predictions = index_folder(path / "original", "_building_damage")
        if not predictions:
            predictions = index_folder(path / "colored", "_building_damage")
        indexes.append(predictions)
        titles.append(title)
    # Restrict the gallery to test samples, when the standard split is present.
    split = args.root / "bda_benchmark/dataset/splitname/standard_ML/test_set.txt"
    candidates = set(gt)
    if split.exists():
        candidates &= {line.strip().replace("\\", "/") for line in split.read_text().splitlines() if line.strip()}
    complete = []
    with (output / "matching_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "missing_columns"])
        for sample in sorted(candidates):
            missing = [titles[i].replace("\n", " ") for i, index in enumerate(indexes) if sample not in index]
            writer.writerow([sample, "; ".join(missing)])
            if not missing:
                complete.append(sample)
    print("Indexed files:", dict(zip(titles, map(len, indexes))))
    print(f"Fully matched test samples: {len(complete)}; report: {output / 'matching_report.csv'}")
    if not complete:
        parser.error("No fully matched samples. Check directories and matching_report.csv.")
    if args.mode == "gallery":
        gallery(output, complete[:args.limit] if args.limit > 0 else complete, indexes, titles)
    elif args.mode == "rank":
        rank_samples(output, complete, indexes, titles, args)
    else:
        if not args.selected:
            parser.error("figure requires --selected selected.txt")
        samples = [s.strip() for s in args.selected.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if not samples or any(s not in complete for s in samples):
            parser.error("Selection is empty or contains unmatched/non-test IDs")
        overrides = json.loads(args.row_labels.read_text(encoding="utf-8")) if args.row_labels else {}
        labels = [overrides[s] if s in overrides else disaster_label(s) for s in samples]
        export = output / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        export.mkdir(parents=True)
        figure(export, samples, indexes, titles, args.crop, args.dpi, labels)
        archive_selection(output, export, samples, indexes, titles, labels, args)
        print(f"Selected source files, scores and configuration archived in {export}")


if __name__ == "__main__":
    main()
