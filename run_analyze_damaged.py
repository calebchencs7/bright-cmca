# run_analyze_damaged.py — launcher for analyze_damaged_distribution.py
# =====================================================================
# Empirically tests the "Damaged pixels concentrate at boundaries"
# hypothesis on your local BRIGHT dataset. Paths are hard-coded — just
# run `python run_analyze_damaged.py` and read the verdict at the end.

import os
import sys
import subprocess

# ----- Paths (edit if your dataset is elsewhere) -----------------------
TARGET_DIR = "/Users/haochen/Documents/Development/Dataset/Bright/target"
SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "analyze_damaged_distribution.py",
)
OUTPUT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "damaged_distribution_report.json",
)

# ----- Knobs ------------------------------------------------------------
# Set MAX_FILES to a small number (e.g. 50) for a quick sanity check, or
# 0 to scan the full training set.
MAX_FILES = 0

# Glob pattern for label files. BRIGHT targets are .tif by default.
PATTERN = "*.tif"


# ----- Run --------------------------------------------------------------

if not os.path.isdir(TARGET_DIR):
    print(f"ERROR: target_dir does not exist: {TARGET_DIR}", file=sys.stderr)
    print("Edit TARGET_DIR in this script to point at your BRIGHT target folder.",
          file=sys.stderr)
    sys.exit(2)

if not os.path.isfile(SCRIPT):
    print(f"ERROR: analysis script not found: {SCRIPT}", file=sys.stderr)
    sys.exit(2)

cmd = [
    sys.executable, SCRIPT,
    "--target_dir", TARGET_DIR,
    "--output_json", OUTPUT_JSON,
    "--pattern", PATTERN,
]
if MAX_FILES and MAX_FILES > 0:
    cmd += ["--max_files", str(MAX_FILES)]

print("RUN:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
subprocess.run(cmd, check=True)
