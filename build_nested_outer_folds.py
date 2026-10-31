#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from make_final_test_protocol import (
    IDEAS_VARIANTS,
    ORG_VARIANTS,
    apply_ideas_variant,
    apply_org_variant,
    compute_named_targets,
    direct_named_split,
)
from prepare_sidecar_splits import (
    count_values,
    fill_missing_gold,
    ids,
    label_sort_key,
    load_gold_map,
    required_row_ok,
    unique_essay_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build outer folds for nested Prompt7 sidecar protocol.")
    p.add_argument("--input", required=True, help="Input cache JSON path.")
    p.add_argument("--output-dir", required=True, help="Output directory for nested outer folds.")
    p.add_argument("--num-outer-folds", type=int, default=5)
    p.add_argument("--outer-seed", type=int, default=20260420)
    p.add_argument(
        "--stratify-by",
        choices=("joint", "combined", "ideas", "organization", "overall"),
        default="joint",
        help="Label used for outer-fold stratification.",
    )
    p.add_argument("--min-stratify-count", type=int, default=5)
    p.add_argument(
        "--asap-data",
        default="data/training_set_rel3.tsv",
        help="ASAP TSV path used to fill missing gold labels.",
    )
    p.add_argument(
        "--ideas-variant",
        choices=tuple(IDEAS_VARIANTS.keys()),
        default="iA_devbest",
        help="How to populate pred_ideas before nested splitting.",
    )
    p.add_argument(
        "--org-variant",
        choices=tuple(ORG_VARIANTS.keys()),
        default="orgbest",
        help="How to populate pred_organization before nested splitting.",
    )
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


def ensure_disjoint_partitions(partitions: dict[str, list[dict[str, Any]]], ordered_names: list[str]) -> None:
    id_sets = {name: set(ids(partitions[name])) for name in ordered_names}
    for idx, left in enumerate(ordered_names):
        for right in ordered_names[idx + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                preview = ",".join(str(x) for x in sorted(overlap)[:10])
                raise ValueError(f"{left}/{right} outer-fold overlap detected: {preview}")


def main() -> None:
    args = parse_args()

    if args.num_outer_folds < 2:
        raise ValueError("--num-outer-folds must be >= 2")

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    rows = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list")

    gold_map = load_gold_map(Path(args.asap_data))
    touched, missing_id = fill_missing_gold(rows, gold_map)

    filtered = [row for row in rows if required_row_ok(row)]
    dropped = len(rows) - len(filtered)
    if len(filtered) < args.num_outer_folds * 20:
        raise ValueError(f"Too few valid rows after filtering: {len(filtered)}")
    unique_essay_ids(filtered)

    ideas_variant_summary = apply_ideas_variant(filtered, args.ideas_variant)
    org_variant_summary = apply_org_variant(filtered, args.org_variant)

    filtered = sort_rows(filtered)
    labels = [nested_strat_label(row, args.stratify_by) for row in filtered]
    label_counts = Counter(labels)
    rare_label_counts = {
        label: count
        for label, count in sorted(label_counts.items(), key=lambda item: label_sort_key(item[0]))
        if count < args.min_stratify_count
    }

    outer_names = [f"outer{i}" for i in range(args.num_outer_folds)]
    outer_targets = compute_named_targets(
        len(filtered),
        {name: 1.0 / args.num_outer_folds for name in outer_names},
        outer_names,
    )
    fold_rows, fold_meta = direct_named_split(
        filtered,
        labels,
        target_counts=outer_targets,
        ordered_names=outer_names,
        seed=args.outer_seed,
    )
    ensure_disjoint_partitions(fold_rows, outer_names)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_pool_path = out_dir / "sidecar_full_pool.json"
    full_pool_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")

    outer_outputs: dict[str, Any] = {}
    outer_essay_ids: dict[str, Any] = {}
    for outer_name in outer_names:
        outer_dir = out_dir / outer_name
        outer_dir.mkdir(parents=True, exist_ok=True)

        outer_test_rows = sort_rows(fold_rows[outer_name])
        outer_dev_pool_rows = sort_rows(
            [row for name in outer_names if name != outer_name for row in fold_rows[name]]
        )

        outer_test_ids = set(ids(outer_test_rows))
        outer_dev_pool_ids = set(ids(outer_dev_pool_rows))
        if outer_test_ids & outer_dev_pool_ids:
            raise ValueError(f"{outer_name} outer_test/outer_dev_pool overlap detected")

        outer_test_path = outer_dir / "sidecar_outer_test.json"
        outer_dev_pool_path = outer_dir / "sidecar_outer_dev_pool.json"
        outer_manifest_path = outer_dir / "outer_split_manifest.json"

        outer_test_path.write_text(json.dumps(outer_test_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        outer_dev_pool_path.write_text(
            json.dumps(outer_dev_pool_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        outer_manifest = {
            "protocol_mode": "nested_cv_outer_fold",
            "outer_fold": outer_name,
            "input_path": str(in_path),
            "source_full_pool_path": str(full_pool_path),
            "outer_seed": int(args.outer_seed),
            "num_outer_folds": int(args.num_outer_folds),
            "stratify_by": args.stratify_by,
            "min_stratify_count": int(args.min_stratify_count),
            "ideas_variant": ideas_variant_summary,
            "org_variant": org_variant_summary,
            "fill_gold": {
                "rows_touched": int(touched),
                "rows_missing_essay_id": int(missing_id),
                "rows_dropped_missing_required": int(dropped),
            },
            "counts": {
                "full_pool_rows": int(len(filtered)),
                "outer_test_rows": int(len(outer_test_rows)),
                "outer_dev_pool_rows": int(len(outer_dev_pool_rows)),
            },
            "distributions": {
                "outer_test": {
                    "gold_ideas": count_values(outer_test_rows, "gold_ideas"),
                    "gold_organization": count_values(outer_test_rows, "gold_organization"),
                    "gold_overall": count_values(outer_test_rows, "gold_overall"),
                },
                "outer_dev_pool": {
                    "gold_ideas": count_values(outer_dev_pool_rows, "gold_ideas"),
                    "gold_organization": count_values(outer_dev_pool_rows, "gold_organization"),
                    "gold_overall": count_values(outer_dev_pool_rows, "gold_overall"),
                },
            },
            "essay_ids": {
                "outer_test": ids(outer_test_rows),
                "outer_dev_pool": ids(outer_dev_pool_rows),
            },
            "outputs": {
                "outer_test": str(outer_test_path),
                "outer_dev_pool": str(outer_dev_pool_path),
            },
        }
        outer_manifest_path.write_text(json.dumps(outer_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        outer_outputs[outer_name] = {
            "outer_test": str(outer_test_path),
            "outer_dev_pool": str(outer_dev_pool_path),
            "manifest": str(outer_manifest_path),
        }
        outer_essay_ids[outer_name] = ids(outer_test_rows)

    protocol_manifest = {
        "protocol_mode": "nested_cv_outer_only",
        "input_path": str(in_path),
        "source_full_pool_path": str(full_pool_path),
        "num_outer_folds": int(args.num_outer_folds),
        "outer_seed": int(args.outer_seed),
        "stratify_by": args.stratify_by,
        "min_stratify_count": int(args.min_stratify_count),
        "ideas_variant": ideas_variant_summary,
        "org_variant": org_variant_summary,
        "fill_gold": {
            "rows_touched": int(touched),
            "rows_missing_essay_id": int(missing_id),
            "rows_dropped_missing_required": int(dropped),
        },
        "counts": {
            "input_rows": int(len(rows)),
            "valid_rows": int(len(filtered)),
        },
        "label_stats": {
            "full_raw_counts": {
                label: label_counts[label]
                for label in sorted(label_counts, key=label_sort_key)
            },
            "rare_labels_below_min_count": rare_label_counts,
            "outer_targets": outer_targets,
            "outer_label_allocations": fold_meta["label_allocations"],
        },
        "essay_ids": {
            "outer_test_by_fold": outer_essay_ids,
        },
        "outputs": {
            "full_pool": str(full_pool_path),
            "outer_folds": outer_outputs,
        },
    }
    protocol_manifest_path = out_dir / "nested_outer_manifest.json"
    protocol_manifest_path.write_text(json.dumps(protocol_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input_rows={len(rows)} valid_rows={len(filtered)} dropped={dropped}")
    print(
        "ideas_variant="
        f"{ideas_variant_summary['variant_name']} "
        f"changed_rows={ideas_variant_summary['changed_rows']}/{ideas_variant_summary['total_rows']}"
    )
    print(
        "org_variant="
        f"{org_variant_summary['variant_name']} "
        f"changed_rows={org_variant_summary['changed_rows']}/{org_variant_summary['total_rows']}"
    )
    for outer_name in outer_names:
        print(
            f"{outer_name}_rows="
            f"{len(fold_rows[outer_name])} "
            f"saved={out_dir / outer_name}"
        )
    print(f"nested_outer_manifest_saved={protocol_manifest_path}")


if __name__ == "__main__":
    main()
