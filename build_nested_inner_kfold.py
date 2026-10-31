#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from make_final_test_protocol import compute_named_targets, direct_named_split
from prepare_sidecar_splits import count_values, direct_stratified_split, ids, label_sort_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build inner K-fold splits for nested Prompt7 sidecar protocol.")
    p.add_argument("--outer-root", required=True, help="Root directory created by build_nested_outer_folds.py.")
    p.add_argument("--outer-folds", default="", help="Optional comma-separated outer folds to process.")
    p.add_argument("--num-inner-folds", type=int, default=3)
    p.add_argument("--inner-seed", type=int, default=20260421)
    p.add_argument("--refit-seed", type=int, default=20260422)
    p.add_argument(
        "--stratify-by",
        choices=("joint", "combined", "ideas", "organization", "overall"),
        default="joint",
        help="Label used for inner-fold and refit stratification.",
    )
    p.add_argument("--min-stratify-count", type=int, default=5)
    p.add_argument(
        "--inner-dev-ratio",
        type=float,
        default=0.10,
        help="Dev ratio inside each inner train-dev pool.",
    )
    p.add_argument("--refit-train-ratio", type=float, default=0.90)
    p.add_argument("--refit-dev-ratio", type=float, default=0.10)
    return p.parse_args()


def nested_strat_label(row: dict[str, Any], mode: str) -> str:
    if mode == "joint":
        return f"{int(row['gold_overall'])}_{int(row['gold_ideas'])}_{int(row['gold_organization'])}"
    if mode == "combined":
        return f"{int(row['gold_ideas'])}_{int(row['gold_organization'])}"
    if mode == "ideas":
        return str(int(row["gold_ideas"]))
    if mode == "organization":
        return str(int(row["gold_organization"]))
    return str(int(row["gold_overall"]))


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row["essay_id"]))


def ensure_disjoint_named(assignments: dict[str, list[dict[str, Any]]], names: list[str], desc: str) -> None:
    id_sets = {name: set(ids(assignments[name])) for name in names}
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                preview = ",".join(str(x) for x in sorted(overlap)[:10])
                raise ValueError(f"{desc} overlap detected for {left}/{right}: {preview}")


def parse_outer_index(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    if not m:
        raise ValueError(f"Cannot parse outer index from {name}")
    return int(m.group(1))


def main() -> None:
    args = parse_args()

    if args.num_inner_folds < 2:
        raise ValueError("--num-inner-folds must be >= 2")
    if not (0.0 < args.inner_dev_ratio < 1.0):
        raise ValueError("--inner-dev-ratio must be in (0,1)")
    if abs(args.refit_train_ratio + args.refit_dev_ratio - 1.0) > 1e-8:
        raise ValueError("--refit-train-ratio and --refit-dev-ratio must sum to 1.0")

    outer_root = Path(args.outer_root)
    if not outer_root.exists():
        raise FileNotFoundError(f"Outer root not found: {outer_root}")

    if args.outer_folds.strip():
        outer_names = [x.strip() for x in args.outer_folds.split(",") if x.strip()]
    else:
        outer_names = sorted(
            path.name for path in outer_root.iterdir() if path.is_dir() and path.name.startswith("outer")
        )
    if not outer_names:
        raise ValueError("No outer folds found")

    top_outputs: dict[str, Any] = {}
    for outer_name in outer_names:
        outer_dir = outer_root / outer_name
        outer_test_path = outer_dir / "sidecar_outer_test.json"
        outer_dev_pool_path = outer_dir / "sidecar_outer_dev_pool.json"
        if not outer_test_path.exists() or not outer_dev_pool_path.exists():
            raise FileNotFoundError(f"Missing outer inputs in {outer_dir}")

        outer_test_rows = sort_rows(json.loads(outer_test_path.read_text(encoding="utf-8")))
        outer_dev_pool_rows = sort_rows(json.loads(outer_dev_pool_path.read_text(encoding="utf-8")))

        outer_idx = parse_outer_index(outer_name)
        inner_seed = args.inner_seed + outer_idx * 100
        refit_seed = args.refit_seed + outer_idx * 100

        pool_labels = [nested_strat_label(row, args.stratify_by) for row in outer_dev_pool_rows]
        pool_label_counts = Counter(pool_labels)
        pool_rare_label_counts = {
            label: count
            for label, count in sorted(pool_label_counts.items(), key=lambda item: label_sort_key(item[0]))
            if count < args.min_stratify_count
        }

        inner_names = [f"s{i}" for i in range(args.num_inner_folds)]
        inner_targets = compute_named_targets(
            len(outer_dev_pool_rows),
            {name: 1.0 / args.num_inner_folds for name in inner_names},
            inner_names,
        )
        inner_buckets, inner_bucket_meta = direct_named_split(
            outer_dev_pool_rows,
            pool_labels,
            target_counts=inner_targets,
            ordered_names=inner_names,
            seed=inner_seed,
        )
        ensure_disjoint_named(inner_buckets, inner_names, f"{outer_name} inner buckets")

        compat_pool_path = outer_dir / "sidecar_training_pool.json"
        compat_final_path = outer_dir / "sidecar_final_test.json"
        compat_pool_path.write_text(json.dumps(outer_dev_pool_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        compat_final_path.write_text(json.dumps(outer_test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        split_outputs: dict[str, Any] = {}
        for inner_name in inner_names:
            split_dir = outer_dir / inner_name
            split_dir.mkdir(parents=True, exist_ok=True)

            holdout_rows = sort_rows(inner_buckets[inner_name])
            train_dev_rows = sort_rows(
                [row for name in inner_names if name != inner_name for row in inner_buckets[name]]
            )
            train_dev_labels = [nested_strat_label(row, args.stratify_by) for row in train_dev_rows]
            train_rows, dev_rows, _, split_meta = direct_stratified_split(
                train_dev_rows,
                train_dev_labels,
                train_ratio=1.0 - args.inner_dev_ratio,
                dev_ratio=args.inner_dev_ratio,
                holdout_ratio=0.0,
                seed=inner_seed + parse_outer_index(inner_name),
            )
            train_rows = sort_rows(train_rows)
            dev_rows = sort_rows(dev_rows)

            train_path = split_dir / "sidecar_train.json"
            dev_path = split_dir / "sidecar_dev.json"
            holdout_path = split_dir / "sidecar_holdout.json"
            manifest_path = split_dir / "sidecar_split_manifest.json"

            train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            dev_path.write_text(json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            holdout_path.write_text(json.dumps(holdout_rows, ensure_ascii=False, indent=2), encoding="utf-8")

            split_manifest = {
                "protocol_mode": "nested_cv_inner_fold",
                "outer_fold": outer_name,
                "inner_fold": inner_name,
                "outer_dev_pool_path": str(outer_dev_pool_path),
                "outer_test_path": str(outer_test_path),
                "seed": int(inner_seed),
                "stratify_by": args.stratify_by,
                "min_stratify_count": int(args.min_stratify_count),
                "inner_dev_ratio": float(args.inner_dev_ratio),
                "bucket_targets": inner_targets,
                "bucket_label_allocations": inner_bucket_meta["label_allocations"],
                "train_dev_split_mode": split_meta["mode"],
                "train_dev_split_targets": split_meta["targets"],
                "counts": {
                    "outer_dev_pool_rows": int(len(outer_dev_pool_rows)),
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
            manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            split_outputs[inner_name] = {
                "train": str(train_path),
                "dev": str(dev_path),
                "holdout": str(holdout_path),
                "manifest": str(manifest_path),
            }

        refit_dir = outer_dir / "refit"
        refit_dir.mkdir(parents=True, exist_ok=True)
        refit_train_rows, refit_dev_rows, _, refit_meta = direct_stratified_split(
            outer_dev_pool_rows,
            pool_labels,
            train_ratio=args.refit_train_ratio,
            dev_ratio=args.refit_dev_ratio,
            holdout_ratio=0.0,
            seed=refit_seed,
        )
        refit_train_rows = sort_rows(refit_train_rows)
        refit_dev_rows = sort_rows(refit_dev_rows)

        refit_train_path = refit_dir / "sidecar_train.json"
        refit_dev_path = refit_dir / "sidecar_dev.json"
        refit_manifest_path = refit_dir / "sidecar_refit_manifest.json"

        refit_train_path.write_text(json.dumps(refit_train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        refit_dev_path.write_text(json.dumps(refit_dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        refit_manifest = {
            "protocol_mode": "nested_cv_refit_split",
            "outer_fold": outer_name,
            "outer_dev_pool_path": str(outer_dev_pool_path),
            "outer_test_path": str(outer_test_path),
            "seed": int(refit_seed),
            "stratify_by": args.stratify_by,
            "refit_train_ratio": float(args.refit_train_ratio),
            "refit_dev_ratio": float(args.refit_dev_ratio),
            "split_mode": refit_meta["mode"],
            "split_targets": refit_meta["targets"],
            "counts": {
                "outer_dev_pool_rows": int(len(outer_dev_pool_rows)),
                "train_rows": int(len(refit_train_rows)),
                "dev_rows": int(len(refit_dev_rows)),
            },
            "distributions": {
                "train": {
                    "gold_ideas": count_values(refit_train_rows, "gold_ideas"),
                    "gold_organization": count_values(refit_train_rows, "gold_organization"),
                    "gold_overall": count_values(refit_train_rows, "gold_overall"),
                },
                "dev": {
                    "gold_ideas": count_values(refit_dev_rows, "gold_ideas"),
                    "gold_organization": count_values(refit_dev_rows, "gold_organization"),
                    "gold_overall": count_values(refit_dev_rows, "gold_overall"),
                },
            },
            "essay_ids": {
                "train": ids(refit_train_rows),
                "dev": ids(refit_dev_rows),
            },
            "outputs": {
                "train": str(refit_train_path),
                "dev": str(refit_dev_path),
            },
        }
        refit_manifest_path.write_text(json.dumps(refit_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        outer_protocol_manifest = {
            "protocol_mode": "nested_cv_outer_fold_ready",
            "outer_fold": outer_name,
            "stratify_by": args.stratify_by,
            "min_stratify_count": int(args.min_stratify_count),
            "inner_seed": int(inner_seed),
            "refit_seed": int(refit_seed),
            "num_inner_folds": int(args.num_inner_folds),
            "inner_dev_ratio": float(args.inner_dev_ratio),
            "refit_train_ratio": float(args.refit_train_ratio),
            "refit_dev_ratio": float(args.refit_dev_ratio),
            "counts": {
                "outer_test_rows": int(len(outer_test_rows)),
                "outer_dev_pool_rows": int(len(outer_dev_pool_rows)),
                "refit_train_rows": int(len(refit_train_rows)),
                "refit_dev_rows": int(len(refit_dev_rows)),
            },
            "label_stats": {
                "outer_dev_pool_raw_counts": {
                    label: pool_label_counts[label]
                    for label in sorted(pool_label_counts, key=label_sort_key)
                },
                "rare_labels_below_min_count": pool_rare_label_counts,
                "inner_targets": inner_targets,
                "inner_bucket_label_allocations": inner_bucket_meta["label_allocations"],
            },
            "essay_ids": {
                "outer_test": ids(outer_test_rows),
                "outer_dev_pool": ids(outer_dev_pool_rows),
            },
            "outputs": {
                "sidecar_training_pool": str(compat_pool_path),
                "sidecar_final_test": str(compat_final_path),
                "inner_splits": split_outputs,
                "refit": {
                    "train": str(refit_train_path),
                    "dev": str(refit_dev_path),
                    "manifest": str(refit_manifest_path),
                },
            },
        }
        protocol_manifest_path = outer_dir / "sidecar_protocol_manifest.json"
        protocol_manifest_path.write_text(
            json.dumps(outer_protocol_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        top_outputs[outer_name] = {
            "protocol_manifest": str(protocol_manifest_path),
            "sidecar_training_pool": str(compat_pool_path),
            "sidecar_final_test": str(compat_final_path),
            "refit": {
                "train": str(refit_train_path),
                "dev": str(refit_dev_path),
            },
            "inner_splits": split_outputs,
        }

        print(
            f"{outer_name}: "
            f"outer_test={len(outer_test_rows)} "
            f"outer_dev_pool={len(outer_dev_pool_rows)} "
            f"refit_train={len(refit_train_rows)} "
            f"refit_dev={len(refit_dev_rows)}"
        )

    top_manifest = {
        "protocol_mode": "nested_cv_full_ready",
        "outer_root": str(outer_root),
        "outer_folds": outer_names,
        "num_inner_folds": int(args.num_inner_folds),
        "stratify_by": args.stratify_by,
        "min_stratify_count": int(args.min_stratify_count),
        "inner_seed": int(args.inner_seed),
        "refit_seed": int(args.refit_seed),
        "inner_dev_ratio": float(args.inner_dev_ratio),
        "refit_train_ratio": float(args.refit_train_ratio),
        "refit_dev_ratio": float(args.refit_dev_ratio),
        "outputs": top_outputs,
    }
    top_manifest_path = outer_root / "nested_inner_manifest.json"
    top_manifest_path.write_text(json.dumps(top_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"nested_inner_manifest_saved={top_manifest_path}")


if __name__ == "__main__":
    main()
