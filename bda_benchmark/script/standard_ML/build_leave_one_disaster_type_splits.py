"""
Build strict leave-one-disaster-type zero-shot splits for BRIGHT.

All samples of the held-out disaster type are used only for testing. Following
the author's cross-event protocol, non-target samples from the original test
split are merged into training, while the non-target validation split remains
the validation set.
"""

import argparse
import os
import sys
from collections import Counter


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.make_data_loader import parse_disaster_event, parse_disaster_type


STANDARD_SPLIT_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "splitname", "standard_ML"
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "dataset", "splitname", "leave_one_disaster_type"
)


def _read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_list(path, items):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(item + "\n")


def _unique_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _load_standard_splits(split_dir):
    return {
        split: _read_list(os.path.join(split_dir, f"{split}_set.txt"))
        for split in ("train", "val", "test")
    }


def _write_summary(path, holdout_type, train_items, val_items, test_items):
    lines = [
        f"holdout_disaster_type: {holdout_type}",
        "protocol: zero-shot leave-one-disaster-type-out",
        f"train: {len(train_items)}",
        f"val: {len(val_items)}",
        f"test: {len(test_items)}",
        "",
        "[train disaster types]",
    ]
    for name, count in Counter(parse_disaster_type(x) for x in train_items).most_common():
        lines.append(f"{name}: {count}")
    lines.extend(["", "[val disaster types]"])
    for name, count in Counter(parse_disaster_type(x) for x in val_items).most_common():
        lines.append(f"{name}: {count}")
    lines.extend(["", "[test events]"])
    for name, count in Counter(parse_disaster_event(x) for x in test_items).most_common():
        lines.append(f"{name}: {count}")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def build_split(splits, holdout_type, output_root):
    holdout_type = str(holdout_type).lower()
    all_items = _unique_preserve_order(
        splits["train"] + splits["val"] + splits["test"]
    )
    test_items = [
        item for item in all_items
        if parse_disaster_type(item) == holdout_type
    ]
    if not test_items:
        raise ValueError(f"No samples found for disaster type: {holdout_type}")

    train_items = _unique_preserve_order([
        item for item in splits["train"] + splits["test"]
        if parse_disaster_type(item) != holdout_type
    ])
    val_items = [
        item for item in splits["val"]
        if parse_disaster_type(item) != holdout_type
    ]

    train_set = set(train_items)
    val_set = set(val_items)
    test_set = set(test_items)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Generated splits overlap.")

    out_dir = os.path.join(output_root, holdout_type)
    os.makedirs(out_dir, exist_ok=True)
    _write_list(os.path.join(out_dir, "train_set.txt"), train_items)
    _write_list(os.path.join(out_dir, "val_set.txt"), val_items)
    _write_list(os.path.join(out_dir, "test_set.txt"), test_items)
    _write_summary(
        os.path.join(out_dir, "summary.txt"),
        holdout_type,
        train_items,
        val_items,
        test_items,
    )
    return out_dir, len(train_items), len(val_items), len(test_items)


def main():
    parser = argparse.ArgumentParser(
        description="Build BRIGHT leave-one-disaster-type zero-shot splits."
    )
    parser.add_argument("--source_split_dir", default=STANDARD_SPLIT_DIR)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--holdout_type",
        action="append",
        default=[],
        help="Disaster type to hold out. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all_types",
        action="store_true",
        help="Generate one split for every disaster type.",
    )
    args = parser.parse_args()

    splits = _load_standard_splits(args.source_split_dir)
    all_items = splits["train"] + splits["val"] + splits["test"]
    available_types = sorted({parse_disaster_type(x) for x in all_items})
    holdout_types = available_types if args.all_types else args.holdout_type
    if not holdout_types:
        raise ValueError(
            "Pass --holdout_type TYPE or --all_types. "
            f"Available types: {', '.join(available_types)}"
        )

    for holdout_type in holdout_types:
        out_dir, n_train, n_val, n_test = build_split(
            splits, holdout_type, args.output_root
        )
        print(
            f"{holdout_type}: train={n_train}, val={n_val}, "
            f"test={n_test} -> {out_dir}"
        )


if __name__ == "__main__":
    main()
