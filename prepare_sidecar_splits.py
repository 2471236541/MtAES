#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 8:1:1 划分 Prompt7 sidecar 的 train/dev/holdout 数据集。

特性：
- 直接基于现有 cache JSON 行数据（essay_id、essay_text、baseline 预测）。
- 按 essay_id 从 ASAP TSV 自动补全缺失的金标 trait。
- 基于单阶段 direct stratified assignment 做切分，减少两阶段切分带来的隐式偏移。
- 输出 train/dev/holdout JSON 及 split manifest。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input cache JSON path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for split files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument(
        "--stratify-by",
        choices=("combined", "ideas", "organization", "overall"),
        default="overall",
        help="Label used for stratification.",
    )
    parser.add_argument(
        "--min-stratify-count",
        type=int,
        default=5,
        help="Only used for manifest warnings; single-stage split does not collapse labels by default.",
    )
    parser.add_argument(
        "--asap-data",
        default="data/training_set_rel3.tsv",
        help="ASAP TSV path for filling missing gold labels.",
    )
    return parser.parse_args()


def load_gold_map(tsv_path: Path) -> dict[int, dict[str, int]]:
    if not tsv_path.exists():
        raise FileNotFoundError(f"ASAP TSV not found: {tsv_path}")

    gold_by_essay_id: dict[int, dict[str, int]] = {}
    with open(tsv_path, "r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                if int(row["essay_set"]) != 7:
                    continue
                essay_id = int(row["essay_id"])
                gold_by_essay_id[essay_id] = {
                    "gold_ideas": int(row["rater1_trait1"]),
                    "gold_organization": int(row["rater1_trait2"]),
                    "gold_style": int(row["rater1_trait3"]),
                    "gold_conventions": int(row["rater1_trait4"]),
                    "gold_overall": int(row["domain1_score"]),
                }
            except Exception:
                continue
    return gold_by_essay_id


def fill_missing_gold(rows: list[dict[str, Any]], gold_map: dict[int, dict[str, int]]) -> tuple[int, int]:
    touched = 0
    missing_id = 0
    for row in rows:
        try:
            essay_id = int(row["essay_id"])
        except Exception:
            missing_id += 1
            continue
        gold = gold_map.get(essay_id)
        if not gold:
            continue
        changed = False
        for key, value in gold.items():
            if key not in row:
                row[key] = int(value)
                changed = True
        if changed:
            touched += 1
    return touched, missing_id


def required_row_ok(row: dict[str, Any]) -> bool:
    required_keys = (
        "essay_id",
        "essay_text",
        "pred_ideas",
        "pred_organization",
        "pred_style",
        "pred_conventions",
        "gold_ideas",
        "gold_organization",
        "gold_style",
        "gold_conventions",
        "gold_overall",
    )
    return all(k in row for k in required_keys)


def strat_label(row: dict[str, Any], mode: str) -> str:
    if mode == "ideas":
        return str(int(row["gold_ideas"]))
    if mode == "organization":
        return str(int(row["gold_organization"]))
    if mode == "overall":
        return str(int(row["gold_overall"]))
    return f"{int(row['gold_ideas'])}_{int(row['gold_organization'])}"


def label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except Exception:
        return (1, label)


def stable_label_seed(label: str, seed: int) -> int:
    extra = sum((idx + 1) * ord(ch) for idx, ch in enumerate(label))
    return seed + extra


def apportion_counts(total: int, quotas: dict[str, float]) -> dict[str, int]:
    split_order = {"train": 0, "dev": 1, "holdout": 2}
    counts = {name: int(quotas[name]) for name in quotas}
    remain = total - sum(counts.values())
    order = sorted(
        quotas,
        key=lambda name: (quotas[name] - counts[name], -split_order.get(name, 999)),
        reverse=True,
    )
    for name in order[:remain]:
        counts[name] += 1
    return counts


def compute_target_counts(
    total_rows: int,
    train_ratio: float,
    dev_ratio: float,
    holdout_ratio: float,
) -> dict[str, int]:
    quotas = {
        "train": total_rows * train_ratio,
        "dev": total_rows * dev_ratio,
        "holdout": total_rows * holdout_ratio,
    }
    return apportion_counts(total_rows, quotas)


def unique_essay_ids(rows: list[dict[str, Any]]) -> None:
    counts = Counter()
    for row in rows:
        try:
            counts[int(row["essay_id"])] += 1
        except Exception:
            continue
    dup_ids = sorted(essay_id for essay_id, cnt in counts.items() if cnt > 1)
    if dup_ids:
        preview = ",".join(str(x) for x in dup_ids[:10])
        raise ValueError(f"Duplicate essay_id found in filtered rows: {preview}")


def ensure_disjoint_splits(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> None:
    train_ids = set(ids(train_rows))
    dev_ids = set(ids(dev_rows))
    holdout_ids = set(ids(holdout_rows))
    if train_ids & dev_ids:
        raise ValueError("train/dev essay_id overlap detected")
    if train_ids & holdout_ids:
        raise ValueError("train/holdout essay_id overlap detected")
    if dev_ids & holdout_ids:
        raise ValueError("dev/holdout essay_id overlap detected")


def direct_stratified_split(
    rows: list[dict[str, Any]],
    labels: list[str],
    train_ratio: float,
    dev_ratio: float,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(labels):
        raise ValueError("rows and labels length mismatch")
    if not rows:
        raise ValueError("No rows to split")

    split_names = ("train", "dev", "holdout")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row, label in zip(rows, labels):
        grouped.setdefault(label, []).append(row)

    global_targets = compute_target_counts(len(rows), train_ratio, dev_ratio, holdout_ratio)
    remaining_targets = dict(global_targets)
    remaining_rows = len(rows)
    assignments: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    label_allocations: dict[str, dict[str, int]] = {}

    ordered_labels = sorted(grouped, key=lambda label: (len(grouped[label]), label_sort_key(label)))
    for label_idx, label in enumerate(ordered_labels):
        label_rows = sorted(grouped[label], key=lambda row: int(row["essay_id"]))
        rng = random.Random(stable_label_seed(label, seed))
        rng.shuffle(label_rows)

        label_total = len(label_rows)
        if label_idx == len(ordered_labels) - 1:
            counts = dict(remaining_targets)
        else:
            quotas = {
                name: (label_total * remaining_targets[name] / remaining_rows) if remaining_rows else 0.0
                for name in split_names
            }
            counts = apportion_counts(label_total, quotas)

        offset = 0
        for name in split_names:
            take = counts[name]
            assignments[name].extend(label_rows[offset : offset + take])
            offset += take
            remaining_targets[name] -= take
        remaining_rows -= label_total
        label_allocations[label] = counts

    if remaining_rows != 0:
        raise RuntimeError(f"Remaining rows should be 0, got {remaining_rows}")
    if any(v != 0 for v in remaining_targets.values()):
        raise RuntimeError(f"Remaining targets should be 0, got {remaining_targets}")

    ensure_disjoint_splits(assignments["train"], assignments["dev"], assignments["holdout"])
    meta = {
        "mode": "single_stage_direct_stratified",
        "targets": global_targets,
        "label_allocations": {label: label_allocations[label] for label in sorted(label_allocations, key=label_sort_key)},
    }
    return assignments["train"], assignments["dev"], assignments["holdout"], meta


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        try:
            counter[str(int(row[key]))] += 1
        except Exception:
            continue
    return dict(sorted(counter.items(), key=lambda x: int(x[0])))


def ids(rows: list[dict[str, Any]]) -> list[int]:
    out = []
    for row in rows:
        try:
            out.append(int(row["essay_id"]))
        except Exception:
            continue
    return out


def main() -> None:
    args = parse_args()

    total_ratio = args.train_ratio + args.dev_ratio + args.holdout_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"Ratios must sum to 1.0, got {args.train_ratio}+{args.dev_ratio}+{args.holdout_ratio}={total_ratio}"
        )

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    rows = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list.")

    gold_map = load_gold_map(Path(args.asap_data))
    touched, missing_id = fill_missing_gold(rows, gold_map)

    filtered = [r for r in rows if required_row_ok(r)]
    dropped = len(rows) - len(filtered)
    if len(filtered) < 50:
        raise ValueError(f"Too few valid rows after filtering: {len(filtered)}")
    unique_essay_ids(filtered)

    raw_labels = [strat_label(r, args.stratify_by) for r in filtered]
    raw_label_counts = Counter(raw_labels)
    rare_label_counts = {
        label: count
        for label, count in sorted(raw_label_counts.items(), key=lambda item: label_sort_key(item[0]))
        if count < args.min_stratify_count
    }

    train_rows, dev_rows, holdout_rows, split_meta = direct_stratified_split(
        filtered,
        raw_labels,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sidecar_train.json"
    dev_path = out_dir / "sidecar_dev.json"
    holdout_path = out_dir / "sidecar_holdout.json"
    manifest_path = out_dir / "sidecar_split_manifest.json"

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    dev_path.write_text(json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout_path.write_text(json.dumps(holdout_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "input_path": str(in_path),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "dev": float(args.dev_ratio),
            "holdout": float(args.holdout_ratio),
        },
        "stratify_by": args.stratify_by,
        "min_stratify_count": int(args.min_stratify_count),
        "fill_gold": {
            "rows_touched": int(touched),
            "rows_missing_essay_id": int(missing_id),
            "rows_dropped_missing_required": int(dropped),
        },
        "split_mode": split_meta["mode"],
        "split_modes": {
            "stage1_train_vs_temp": "not_used_single_stage_direct",
            "stage2_dev_vs_holdout": "not_used_single_stage_direct",
        },
        "split_targets": split_meta["targets"],
        "label_stats": {
            "raw_counts": {
                label: raw_label_counts[label]
                for label in sorted(raw_label_counts, key=label_sort_key)
            },
            "rare_labels_below_min_count": rare_label_counts,
            "rare_label_strategy": "not_used_single_stage_direct",
            "label_allocations": split_meta["label_allocations"],
        },
        "counts": {
            "input_rows": int(len(rows)),
            "valid_rows": int(len(filtered)),
            "train_rows": int(len(train_rows)),
            "dev_rows": int(len(dev_rows)),
            "holdout_rows": int(len(holdout_rows)),
        },
        "distributions": {
            "train": {
                "gold_ideas": count_values(train_rows, "gold_ideas"),
                "gold_organization": count_values(train_rows, "gold_organization"),
                "gold_overall": count_values(train_rows, "gold_overall"),
            },
            "dev": {
                "gold_ideas": count_values(dev_rows, "gold_ideas"),
                "gold_organization": count_values(dev_rows, "gold_organization"),
                "gold_overall": count_values(dev_rows, "gold_overall"),
            },
            "holdout": {
                "gold_ideas": count_values(holdout_rows, "gold_ideas"),
                "gold_organization": count_values(holdout_rows, "gold_organization"),
                "gold_overall": count_values(holdout_rows, "gold_overall"),
            },
        },
        "essay_ids": {
            "train": ids(train_rows),
            "dev": ids(dev_rows),
            "holdout": ids(holdout_rows),
        },
        "outputs": {
            "train": str(train_path),
            "dev": str(dev_path),
            "holdout": str(holdout_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input_rows={len(rows)} valid_rows={len(filtered)} dropped={dropped}")
    print(f"train_rows={len(train_rows)} saved={train_path}")
    print(f"dev_rows={len(dev_rows)} saved={dev_path}")
    print(f"holdout_rows={len(holdout_rows)} saved={holdout_path}")
    print(f"manifest_saved={manifest_path}")
    print(f"split_mode={split_meta['mode']}")


if __name__ == "__main__":
    main()
