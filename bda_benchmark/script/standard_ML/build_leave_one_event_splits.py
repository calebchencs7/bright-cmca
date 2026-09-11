"""
Build leave-one-disaster-event splits for BRIGHT.

The script merges the standard train/val/test lists, holds one event out as
the test set, and stratifies all remaining events into train/val lists.
"""

import argparse
import os
import random
import sys
from collections import Counter, defaultdict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.make_data_loader import parse_disaster_event


STANDARD_SPLIT_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "splitname", "standard_ML"
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "dataset", "splitname", "leave_one_event"
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
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_standard_items(split_dir):
    items = []
    for name in ("train_set.txt", "val_set.txt", "test_set.txt"):
        items.extend(_read_list(os.path.join(split_dir, name)))
    return _unique_preserve_order(items)


def _stratified_train_val(items, val_ratio, seed):
    rng = random.Random(seed)
    by_event = defaultdict(list)
    for item in items:
        by_event[parse_disaster_event(item)].append(item)

    train_items = []
    val_items = []
    for event in sorted(by_event):
        event_items = list(by_event[event])
        rng.shuffle(event_items)
        n_val = max(1, int(round(len(event_items) * val_ratio)))
        n_val = min(n_val, max(0, len(event_items) - 1))
        val_items.extend(event_items[:n_val])
        train_items.extend(event_items[n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def _write_summary(path, holdout_event, train_items, val_items, test_items):
    lines = [
        f"holdout_event: {holdout_event}",
        f"train: {len(train_items)}",
        f"val: {len(val_items)}",
        f"test: {len(test_items)}",
        "",
        "[train events]",
    ]
    for event, count in Counter(parse_disaster_event(x) for x in train_items).most_common():
        lines.append(f"{event}: {count}")
    lines.extend(["", "[val events]"])
    for event, count in Counter(parse_disaster_event(x) for x in val_items).most_common():
        lines.append(f"{event}: {count}")
    lines.extend(["", "[test events]"])
    for event, count in Counter(parse_disaster_event(x) for x in test_items).most_common():
        lines.append(f"{event}: {count}")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def build_split(items, holdout_event, output_root, val_ratio, seed):
    holdout_event = str(holdout_event)
    test_items = [x for x in items if parse_disaster_event(x) == holdout_event]
    pool_items = [x for x in items if parse_disaster_event(x) != holdout_event]
    if not test_items:
        raise ValueError(f"No samples found for holdout event: {holdout_event}")

    train_items, val_items = _stratified_train_val(pool_items, val_ratio, seed)
    out_dir = os.path.join(output_root, holdout_event)
    os.makedirs(out_dir, exist_ok=True)

    _write_list(os.path.join(out_dir, "train_set.txt"), train_items)
    _write_list(os.path.join(out_dir, "val_set.txt"), val_items)
    _write_list(os.path.join(out_dir, "test_set.txt"), test_items)
    _write_summary(
        os.path.join(out_dir, "summary.txt"),
        holdout_event,
        train_items,
        val_items,
        test_items,
    )
    return out_dir, len(train_items), len(val_items), len(test_items)


def main():
    parser = argparse.ArgumentParser(description="Build BRIGHT leave-one-event splits.")
    parser.add_argument("--source_split_dir", default=STANDARD_SPLIT_DIR)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--holdout_event", action="append", default=[],
                        help="Event to hold out. Can be passed multiple times.")
    parser.add_argument("--all_events", action="store_true",
                        help="Generate one split for every event.")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = _load_standard_items(args.source_split_dir)
    events = sorted({parse_disaster_event(x) for x in items})
    holdout_events = events if args.all_events else args.holdout_event
    if not holdout_events:
        raise ValueError(
            "Pass --holdout_event EVENT or --all_events. "
            f"Available events: {', '.join(events)}"
        )

    for event in holdout_events:
        out_dir, n_train, n_val, n_test = build_split(
            items, event, args.output_root, args.val_ratio, args.seed
        )
        print(
            f"{event}: train={n_train}, val={n_val}, test={n_test} -> {out_dir}"
        )


if __name__ == "__main__":
    main()
