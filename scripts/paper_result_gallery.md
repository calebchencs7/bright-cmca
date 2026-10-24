# BRIGHT paper result selection

Copy `paper_result_gallery.py` into `E:\haoChen\BRIGHT\scripts` on Windows.
Run in the Python environment used for inference:

```powershell
cd E:\haoChen\BRIGHT
python -m pip install numpy pillow tifffile matplotlib
python scripts\paper_result_gallery.py gallery --limit 100
```

Open `figure\paper_selection\gallery.html` in a browser. Each row contains
pre-event optical imagery, post-event SAR, ground truth, and six predictions.
Filter by event/sample ID, click images to enlarge, tick samples, then click
**Download selected.txt**. Copy the downloaded file into
`figure\paper_selection\selected.txt`. Remove `--limit 100` to generate all
fully matched test samples; many previews can take time and disk space.

```powershell
python scripts\paper_result_gallery.py figure --selected figure\paper_selection\selected.txt
```

Each figure run creates a new timestamped folder under
`figure\paper_selection\exports`, preserving earlier exports. It contains
`qualitative_results.pdf`, `qualitative_results.png` (600 dpi), and
`figure_samples.txt`. Selection file order determines figure row order.
The left side of each row shows its English disaster type, inferred from the
sample's event name. For custom labels, pass `--row-labels labels.json`, a JSON
object such as `{"haiti-earthquake_00000007": "Earthquake"}`.

The same export folder also retains:

- `selected.txt`: selected IDs in figure order.
- `selected_data\row_XX\column_XX`: untouched copies of all nine source files
  used for each row (full images / class maps, not preview thumbnails).
- `selection_manifest.json`: sample IDs, disaster labels, original file paths,
  saved copy paths, available scores, model folders, palette, crop and dpi.
- `selected_scores.csv`: selected samples' scores, if rank mode ran beforehand.
- `recommended.txt` and `typical.txt`: snapshots of available candidate lists.
- `paper_result_gallery.py`: the script version used for the export.

Scores and sample IDs are recorded in these files, not printed in the paper
figure. A previous selection is retained by keeping its `selected.txt` or export
folder. The script cannot recover earlier browser selections that were never
downloaded. Only the actual panel source files are copied; alternate colored
versions and checkpoints are not copied. No source data is changed.

Figure fonts default to installed **Times New Roman**, visually compatible with
the current manuscript's default IEEEtran Times family. Missing fonts produce
an error rather than silently using a different font. Use `--font "Times"` or
another explicitly installed family if needed. For LaTeX-rendered Times glyphs
matching IEEEtran's default `ptm`, add `--tex-font` (requires a working local
LaTeX/Matplotlib usetex toolchain on the Windows machine). This mode is optional;
ordinary export needs no TeX installation. Long model titles wrap over two
lines; the exporter measures label widths to prevent adjacent columns from
overlapping. Font options are included in the archived manifest. Use the PDF
at `\textwidth` in a `figure*` environment to retain intended text sizing.
Optional identical crop for every selected row/panel:

```powershell
python scripts\paper_result_gallery.py figure --selected figure\paper_selection\selected.txt --crop 256 256 512 512
```

Defaults use the six CMCA + DACutMix model folders. To override names/paths,
pass `--models models.json` to both commands. JSON must contain six entries,
with display names as keys and folders relative to `infer_results` as values.

Matching uses relative sample paths and the suffixes `_building_damage`,
`_pre_disaster`, and `_post_disaster`. `matching_report.csv` reports missing
panels; `preview_errors.json` reports unreadable files or mismatched sizes.
The standard test split is applied if its file exists. Ground truth uses
`data\target` if available, otherwise `data\target_colored`. Predictions use
`original` if available, otherwise `colored`. Raw class IDs 0–3 are mapped
to the UNet inference palette (Background, Intact, Damaged, Destroyed).
Existing RGB labels retain their colors: ensure their palette matches if
raw maps are unavailable. Non-uint8 imagery uses a display-only 2–98 percentile
stretch per image; uint8 imagery is displayed unchanged. Class maps are never
stretched. Source files are not modified.

## Automatic shortlist

```powershell
python scripts\paper_result_gallery.py rank --per-event 3
```

This scores all fully matched samples and overwrites `gallery.html` with a
shortlist: three highest-scoring samples per event plus one median-scoring
sample per event. `recommended.txt` contains the top candidates, `typical.txt`
contains median cases, and `sample_scores.csv` contains all per-model scores,
class IoUs, pixel counts, and exclusion reasons. `ranking_errors.json` records
failed decoding / incompatible sizes. Rankings never guess colors: RGB labels
must exactly match PALETTE; use original class maps if colors differ.

Defaults require at least 200 GT building pixels and 100 GT damage pixels.
These are pixel counts, not building counts. Adjust with
`--min-building-pixels` and `--min-damage-pixels`. Intact-only samples are
excluded intentionally from this damage shortlist; use ordinary gallery mode
to inspect these. `--limit` applies only to ordinary gallery mode.

Per model, score = 0.5 foreground mIoU + 0.5 damage mIoU.
Foreground uses class IDs 1,2,3; damage uses 2,3. Classes absent in both GT and
prediction are omitted; absent GT classes with false-positive predictions get
IoU zero. IoU uses the entire image, including background false positives and
missed buildings. Final selection score = 0.7 mean model score + 0.3 worst model
score. This is a transparent selection heuristic, not the benchmark's overall
evaluation metric. It emphasizes damage classes and discourages cases where
one backbone fails badly. Scores are computed on full images, not optional
figure crops. Median cases refer only to candidates passing these filters.

To directly export the suggested candidates, pass
`--selected figure\paper_selection\recommended.txt`; this may produce many
rows, so normally select 4–6 examples in the shortlist gallery first.

Select representative cases across disaster events, damage classes, and
building densities. Include informative difficult cases rather than only
easy cases. Selection rankings do not replace full-test-set evaluation.
State in the paper caption that all six backbones incorporate CMCA and are
trained with DACutMix. Captions are added in the manuscript, not burned into
the exported figure.
