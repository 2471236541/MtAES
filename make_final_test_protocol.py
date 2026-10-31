#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an independent final-test protocol for Prompt7 sidecar experiments.

Protocol:
- Freeze one final test split from the full cache.
- Save the remaining rows as a training pool.
- Create 3 seeded train/dev/holdout resplits inside the training pool.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prompt7_mapper import Prompt7Mapper
from prepare_sidecar_splits import (
    count_values,
    direct_stratified_split,
    fill_missing_gold,
    ids,
    label_sort_key,
    load_gold_map,
    required_row_ok,
    stable_label_seed,
    strat_label,
    unique_essay_ids,
)


IDEAS_VARIANTS: dict[str, dict[str, Any]] = {
    "current": {
        "variant_name": "current",
        "type": "cached_pred_ideas",
        "rewrite_pred_ideas": False,
        "reverse_q5": False,
        "weights": [1, 1, 1, 1, 1],
        "thresholds": [0.4, 0.6, 0.8],
    },
    "reverse_q5": {
        "variant_name": "reverse_q5",
        "type": "generic_yesno_weighted",
        "rewrite_pred_ideas": True,
        "reverse_q5": True,
        "weights": [1, 1, 1, 1, 1],
        "thresholds": [0.4, 0.6, 0.8],
    },
    "iA_devbest": {
        "variant_name": "iA_devbest",
        "type": "generic_yesno_weighted",
        "rewrite_pred_ideas": True,
        "reverse_q5": False,
        "weights": [1, 1, 2, 2, 1],
        "thresholds": [0.3, 0.5, 0.7],
    },
    "iB_balanced": {
        "variant_name": "iB_balanced",
        "type": "generic_yesno_weighted",
        "rewrite_pred_ideas": True,
        "reverse_q5": False,
        "weights": [1, 2, 2, 3, 1],
        "thresholds": [0.3, 0.5, 0.7],
    },
    "iC_fieldaware": {
        "variant_name": "iC_fieldaware",
        "type": "generic_yesno_reasoning_caps",
        "rewrite_pred_ideas": True,
        "reverse_q5": False,
        "weights": [1, 1, 2, 2, 1],
        "thresholds": [0.3, 0.5, 0.7],
    },
}

ORG_VARIANTS: dict[str, dict[str, Any]] = {
    "current": {
        "variant_name": "current",
        "type": "cached_pred_organization",
        "rewrite_pred_organization": False,
        "weights": [1, 1, 1, 1, 1, 1],
        "thresholds": [0.4, 0.6, 0.8],
    },
    "orgbest": {
        "variant_name": "orgbest",
        "type": "generic_yesno_weighted",
        "rewrite_pred_organization": True,
        "weights": [1, 2, 2, 1, 1, 2],
        "thresholds": [0.4, 0.6, 0.7],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input cache JSON path.")
    p.add_argument("--output-dir", required=True, help="Output root directory.")
    p.add_argument("--final-test-ratio", type=float, default=0.10)
    p.add_argument("--pool-train-ratio", type=float, default=0.8)
    p.add_argument("--pool-dev-ratio", type=float, default=0.1)
    p.add_argument("--pool-holdout-ratio", type=float, default=0.1)
    p.add_argument("--final-seed", type=int, default=20260415)
    p.add_argument("--split-seeds", default="20260322,20260412,20260413")
    p.add_argument(
        "--stratify-by",
        choices=("combined", "ideas", "organization", "overall"),
        default="overall",
        help="Primary stratification label for final test and pool resplits.",
    )
    p.add_argument(
        "--min-stratify-count",
        type=int,
        default=5,
        help="Only used for manifest warnings; direct split does not collapse labels.",
    )
    p.add_argument(
        "--asap-data",
        default="data/training_set_rel3.tsv",
        help="ASAP TSV path for filling missing gold labels.",
    )
    p.add_argument(
        "--ideas-variant",
        choices=tuple(IDEAS_VARIANTS.keys()),
        default="current",
        help="How to populate pred_ideas in the generated protocol caches.",
    )
    p.add_argument(
        "--org-variant",
        choices=tuple(ORG_VARIANTS.keys()),
        default="current",
        help="How to populate pred_organization in the generated protocol caches.",
    )
    return p.parse_args()


def apportion_named_counts(total: int, quotas: dict[str, float], ordered_names: list[str]) -> dict[str, int]:
    counts = {name: int(quotas[name]) for name in ordered_names}
    remain = total - sum(counts.values())
    rank = {name: idx for idx, name in enumerate(ordered_names)}
    order = sorted(
        ordered_names,
        key=lambda name: (quotas[name] - counts[name], -rank[name]),
        reverse=True,
    )
    for name in order[:remain]:
        counts[name] += 1
    return counts


def compute_named_targets(total_rows: int, ratios: dict[str, float], ordered_names: list[str]) -> dict[str, int]:
    quotas = {name: total_rows * ratios[name] for name in ordered_names}
    return apportion_named_counts(total_rows, quotas, ordered_names)


def ensure_disjoint_named(assignments: dict[str, list[dict[str, Any]]], ordered_names: list[str]) -> None:
    id_sets = {name: set(ids(assignments[name])) for name in ordered_names}
    for idx, left in enumerate(ordered_names):
        for right in ordered_names[idx + 1 :]:
            if id_sets[left] & id_sets[right]:
                raise ValueError(f"{left}/{right} essay_id overlap detected")


def direct_named_split(
    rows: list[dict[str, Any]],
    labels: list[str],
    target_counts: dict[str, int],
    ordered_names: list[str],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(rows) != len(labels):
        raise ValueError("rows and labels length mismatch")
    if not rows:
        raise ValueError("No rows to split")
    if sum(target_counts.values()) != len(rows):
        raise ValueError(f"target_counts must sum to total rows, got {sum(target_counts.values())} vs {len(rows)}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row, label in zip(rows, labels):
        grouped.setdefault(label, []).append(row)

    remaining_targets = dict(target_counts)
    remaining_rows = len(rows)
    assignments = {name: [] for name in ordered_names}
    label_allocations: dict[str, dict[str, int]] = {}

    ordered_labels = sorted(grouped, key=lambda label: (len(grouped[label]), label_sort_key(label)))
    for label_idx, label in enumerate(ordered_labels):
        label_rows = sorted(grouped[label], key=lambda row: int(row["essay_id"]))
        import random

        rng = random.Random(stable_label_seed(label, seed))
        rng.shuffle(label_rows)

        label_total = len(label_rows)
        if label_idx == len(ordered_labels) - 1:
            counts = dict(remaining_targets)
        else:
            quotas = {
                name: (label_total * remaining_targets[name] / remaining_rows) if remaining_rows else 0.0
                for name in ordered_names
            }
            counts = apportion_named_counts(label_total, quotas, ordered_names)

        offset = 0
        for name in ordered_names:
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

    ensure_disjoint_named(assignments, ordered_names)
    meta = {
        "mode": "single_stage_direct_stratified_named",
        "targets": target_counts,
        "label_allocations": {
            label: label_allocations[label]
            for label in sorted(label_allocations, key=label_sort_key)
        },
    }
    return assignments, meta


def extract_yesno_signals(reasoning: str) -> list[float | None]:
    answers = Prompt7Mapper._extract_answer_lines(reasoning)
    out: list[float | None] = []
    for answer in answers:
        sig = Prompt7Mapper._detect_yes_no_partial(answer)
        out.append(None if sig is None else float(sig))
    return out


def score_ideas_generic(
    reasoning: str,
    weights: tuple[int, ...],
    thresholds: tuple[float, float, float],
    reverse_q5: bool,
) -> int:
    signals = extract_yesno_signals(reasoning)
    reverse_idx = {4} if reverse_q5 else set()
    weighted_sum = 0.0
    weight_total = 0.0

    for idx, sig in enumerate(signals):
        if sig is None:
            continue
        val = 1.0 - sig if idx in reverse_idx else sig
        weight = float(weights[idx]) if idx < len(weights) else 1.0
        weighted_sum += val * weight
        weight_total += weight

    if weight_total <= 0:
        return 0

    mean_val = weighted_sum / weight_total
    t1, t2, t3 = thresholds
    if mean_val >= t3:
        return 3
    if mean_val >= t2:
        return 2
    if mean_val >= t1:
        return 1
    return 0


def score_org_generic(
    reasoning: str,
    weights: tuple[int, ...],
    thresholds: tuple[float, float, float],
) -> int:
    signals = extract_yesno_signals(reasoning)
    weighted_sum = 0.0
    weight_total = 0.0

    for idx, sig in enumerate(signals):
        if sig is None:
            continue
        weight = float(weights[idx]) if idx < len(weights) else 1.0
        weighted_sum += float(sig) * weight
        weight_total += weight

    if weight_total <= 0:
        return 0

    mean_val = weighted_sum / weight_total
    t1, t2, t3 = thresholds
    if mean_val >= t3:
        return 3
    if mean_val >= t2:
        return 2
    if mean_val >= t1:
        return 1
    return 0


def apply_ideas_variant(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    spec = dict(IDEAS_VARIANTS[variant_name])
    total_rows = int(len(rows))
    changed_rows = 0
    old_dist = Counter()
    new_dist = Counter()

    if not spec["rewrite_pred_ideas"]:
        for row in rows:
            pred = int(row["pred_ideas"])
            old_dist[pred] += 1
            new_dist[pred] += 1
        spec["total_rows"] = total_rows
        spec["changed_rows"] = 0
        spec["changed_ratio"] = 0.0
        spec["old_pred_distribution"] = {str(k): int(v) for k, v in sorted(old_dist.items())}
        spec["new_pred_distribution"] = {str(k): int(v) for k, v in sorted(new_dist.items())}
        return spec

    for row in rows:
        old_pred = int(row["pred_ideas"])
        if spec["type"] == "generic_yesno_reasoning_caps":
            weights = tuple(int(x) for x in spec["weights"])
            thresholds = tuple(float(x) for x in spec["thresholds"])
            reverse_q5 = bool(spec["reverse_q5"])
            base_pred = int(
                score_ideas_generic(
                    reasoning=str(row.get("ideas_reasoning", "")),
                    weights=weights,
                    thresholds=thresholds,
                    reverse_q5=reverse_q5,
                )
            )
            new_pred, _ = Prompt7Mapper.adjust_ideas_score_with_reasoning_caps(
                score=base_pred,
                reasoning=str(row.get("ideas_reasoning", "")),
            )
            new_pred = int(new_pred)
        else:
            weights = tuple(int(x) for x in spec["weights"])
            thresholds = tuple(float(x) for x in spec["thresholds"])
            reverse_q5 = bool(spec["reverse_q5"])
            new_pred = int(
                score_ideas_generic(
                    reasoning=str(row.get("ideas_reasoning", "")),
                    weights=weights,
                    thresholds=thresholds,
                    reverse_q5=reverse_q5,
                )
            )
        row["pred_ideas"] = new_pred
        old_dist[old_pred] += 1
        new_dist[new_pred] += 1
        if old_pred != new_pred:
            changed_rows += 1

    spec["total_rows"] = total_rows
    spec["changed_rows"] = int(changed_rows)
    spec["changed_ratio"] = float(changed_rows / max(1, total_rows))
    spec["old_pred_distribution"] = {str(k): int(v) for k, v in sorted(old_dist.items())}
    spec["new_pred_distribution"] = {str(k): int(v) for k, v in sorted(new_dist.items())}
    return spec


def apply_org_variant(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    spec = dict(ORG_VARIANTS[variant_name])
    total_rows = int(len(rows))
    changed_rows = 0
    old_dist = Counter()
    new_dist = Counter()

    if not spec["rewrite_pred_organization"]:
        for row in rows:
            pred = int(row["pred_organization"])
            old_dist[pred] += 1
            new_dist[pred] += 1
        spec["total_rows"] = total_rows
        spec["changed_rows"] = 0
        spec["changed_ratio"] = 0.0
        spec["old_pred_distribution"] = {str(k): int(v) for k, v in sorted(old_dist.items())}
        spec["new_pred_distribution"] = {str(k): int(v) for k, v in sorted(new_dist.items())}
        return spec

    weights = tuple(int(x) for x in spec["weights"])
    thresholds = tuple(float(x) for x in spec["thresholds"])

    for row in rows:
        old_pred = int(row["pred_organization"])
        new_pred = int(
            score_org_generic(
                reasoning=str(row.get("organization_reasoning", "")),
                weights=weights,
                thresholds=thresholds,
            )
        )
        row["pred_organization"] = new_pred
        old_dist[old_pred] += 1
        new_dist[new_pred] += 1
        if old_pred != new_pred:
            changed_rows += 1

    spec["total_rows"] = total_rows
    spec["changed_rows"] = int(changed_rows)
    spec["changed_ratio"] = float(changed_rows / max(1, total_rows))
    spec["old_pred_distribution"] = {str(k): int(v) for k, v in sorted(old_dist.items())}
    spec["new_pred_distribution"] = {str(k): int(v) for k, v in sorted(new_dist.items())}
    return spec


def main() -> None:
    args = parse_args()

    pool_total = args.pool_train_ratio + args.pool_dev_ratio + args.pool_holdout_ratio
    if abs(pool_total - 1.0) > 1e-8:
        raise ValueError(
            "Pool ratios must sum to 1.0, got "
            f"{args.pool_train_ratio}+{args.pool_dev_ratio}+{args.pool_holdout_ratio}={pool_total}"
        )
    if not (0.0 < args.final_test_ratio < 1.0):
        raise ValueError(f"--final-test-ratio must be in (0,1), got {args.final_test_ratio}")

    split_seeds = [int(x.strip()) for x in args.split_seeds.split(",") if x.strip()]
    if len(split_seeds) != 3:
        raise ValueError("--split-seeds must contain exactly 3 integer seeds.")
    split_tags = ["s0", "s1", "s2"]

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
    if len(filtered) < 100:
        raise ValueError(f"Too few valid rows after filtering: {len(filtered)}")
    unique_essay_ids(filtered)

    ideas_variant_summary = apply_ideas_variant(filtered, args.ideas_variant)
    org_variant_summary = apply_org_variant(filtered, args.org_variant)

    full_labels = [strat_label(r, args.stratify_by) for r in filtered]
    full_label_counts = Counter(full_labels)
    rare_label_counts = {
        label: count
        for label, count in sorted(full_label_counts.items(), key=lambda item: label_sort_key(item[0]))
        if count < args.min_stratify_count
    }

    partition_names = ["training_pool", "final_test"]
    partition_targets = compute_named_targets(
        len(filtered),
        {
            "training_pool": 1.0 - args.final_test_ratio,
            "final_test": args.final_test_ratio,
        },
        partition_names,
    )
    partition_rows, partition_meta = direct_named_split(
        filtered,
        full_labels,
        target_counts=partition_targets,
        ordered_names=partition_names,
        seed=args.final_seed,
    )

    training_pool_rows = partition_rows["training_pool"]
    final_test_rows = partition_rows["final_test"]
    pool_labels = [strat_label(r, args.stratify_by) for r in training_pool_rows]
    pool_label_counts = Counter(pool_labels)
    pool_rare_label_counts = {
        label: count
        for label, count in sorted(pool_label_counts.items(), key=lambda item: label_sort_key(item[0]))
        if count < args.min_stratify_count
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "sidecar_training_pool.json"
    final_test_path = out_dir / "sidecar_final_test.json"
    protocol_manifest_path = out_dir / "sidecar_protocol_manifest.json"

    pool_path.write_text(json.dumps(training_pool_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    final_test_path.write_text(json.dumps(final_test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    split_outputs: dict[str, str] = {}
    for split_tag, split_seed in zip(split_tags, split_seeds):
        split_dir = out_dir / split_tag
        split_dir.mkdir(parents=True, exist_ok=True)

        train_rows, dev_rows, holdout_rows, split_meta = direct_stratified_split(
            training_pool_rows,
            pool_labels,
            train_ratio=args.pool_train_ratio,
            dev_ratio=args.pool_dev_ratio,
            holdout_ratio=args.pool_holdout_ratio,
            seed=split_seed,
        )

        train_path = split_dir / "sidecar_train.json"
        dev_path = split_dir / "sidecar_dev.json"
        holdout_path = split_dir / "sidecar_holdout.json"
        manifest_path = split_dir / "sidecar_split_manifest.json"

        train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        dev_path.write_text(json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        holdout_path.write_text(json.dumps(holdout_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        split_manifest = {
            "protocol_manifest_path": str(protocol_manifest_path),
            "input_path": str(in_path),
            "source_training_pool_path": str(pool_path),
            "final_test_path": str(final_test_path),
            "seed": int(split_seed),
            "ratios": {
                "train": float(args.pool_train_ratio),
                "dev": float(args.pool_dev_ratio),
                "holdout": float(args.pool_holdout_ratio),
            },
            "stratify_by": args.stratify_by,
            "min_stratify_count": int(args.min_stratify_count),
            "fill_gold": {
                "rows_touched": int(touched),
                "rows_missing_essay_id": int(missing_id),
                "rows_dropped_missing_required": int(dropped),
            },
            "split_mode": split_meta["mode"],
            "ideas_variant": ideas_variant_summary,
            "org_variant": org_variant_summary,
            "split_modes": {
                "stage1_train_vs_temp": "not_used_single_stage_direct",
                "stage2_dev_vs_holdout": "not_used_single_stage_direct",
            },
            "split_targets": split_meta["targets"],
            "label_stats": {
                "training_pool_raw_counts": {
                    label: pool_label_counts[label]
                    for label in sorted(pool_label_counts, key=label_sort_key)
                },
                "rare_labels_below_min_count": pool_rare_label_counts,
                "rare_label_strategy": "not_used_single_stage_direct",
                "label_allocations": split_meta["label_allocations"],
            },
            "counts": {
                "input_rows": int(len(rows)),
                "valid_rows": int(len(filtered)),
                "training_pool_rows": int(len(training_pool_rows)),
                "final_test_rows": int(len(final_test_rows)),
                "train_rows": int(len(train_rows)),
                "dev_rows": int(len(dev_rows)),
                "holdout_rows": int(len(holdout_rows)),
            },
            "distributions": {
                "training_pool": {
                    "gold_ideas": count_values(training_pool_rows, "gold_ideas"),
                    "gold_organization": count_values(training_pool_rows, "gold_organization"),
                    "gold_overall": count_values(training_pool_rows, "gold_overall"),
                },
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
                "final_test": {
                    "gold_ideas": count_values(final_test_rows, "gold_ideas"),
                    "gold_organization": count_values(final_test_rows, "gold_organization"),
                    "gold_overall": count_values(final_test_rows, "gold_overall"),
                },
            },
            "essay_ids": {
                "training_pool": ids(training_pool_rows),
                "train": ids(train_rows),
                "dev": ids(dev_rows),
                "holdout": ids(holdout_rows),
                "final_test": ids(final_test_rows),
            },
            "outputs": {
                "train": str(train_path),
                "dev": str(dev_path),
                "holdout": str(holdout_path),
            },
        }
        manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        split_outputs[split_tag] = str(split_dir)

    protocol_manifest = {
        "protocol_mode": "independent_final_test_then_pool_resplits",
        "input_path": str(in_path),
        "stratify_by": args.stratify_by,
        "min_stratify_count": int(args.min_stratify_count),
        "final_test_ratio": float(args.final_test_ratio),
        "pool_ratios": {
            "train": float(args.pool_train_ratio),
            "dev": float(args.pool_dev_ratio),
            "holdout": float(args.pool_holdout_ratio),
        },
        "seeds": {
            "final_test_seed": int(args.final_seed),
            "pool_split_seeds": {
                split_tag: int(split_seed)
                for split_tag, split_seed in zip(split_tags, split_seeds)
            },
        },
        "fill_gold": {
            "rows_touched": int(touched),
            "rows_missing_essay_id": int(missing_id),
            "rows_dropped_missing_required": int(dropped),
        },
        "counts": {
            "input_rows": int(len(rows)),
            "valid_rows": int(len(filtered)),
            "training_pool_rows": int(len(training_pool_rows)),
            "final_test_rows": int(len(final_test_rows)),
        },
        "ideas_variant": ideas_variant_summary,
        "org_variant": org_variant_summary,
        "label_stats": {
            "full_raw_counts": {
                label: full_label_counts[label]
                for label in sorted(full_label_counts, key=label_sort_key)
            },
            "rare_labels_below_min_count": rare_label_counts,
            "partition_targets": partition_meta["targets"],
            "partition_label_allocations": partition_meta["label_allocations"],
            "training_pool_raw_counts": {
                label: pool_label_counts[label]
                for label in sorted(pool_label_counts, key=label_sort_key)
            },
            "training_pool_rare_labels_below_min_count": pool_rare_label_counts,
        },
        "distributions": {
            "training_pool": {
                "gold_ideas": count_values(training_pool_rows, "gold_ideas"),
                "gold_organization": count_values(training_pool_rows, "gold_organization"),
                "gold_overall": count_values(training_pool_rows, "gold_overall"),
            },
            "final_test": {
                "gold_ideas": count_values(final_test_rows, "gold_ideas"),
                "gold_organization": count_values(final_test_rows, "gold_organization"),
                "gold_overall": count_values(final_test_rows, "gold_overall"),
            },
        },
        "essay_ids": {
            "training_pool": ids(training_pool_rows),
            "final_test": ids(final_test_rows),
        },
        "outputs": {
            "training_pool": str(pool_path),
            "final_test": str(final_test_path),
            "pool_splits": split_outputs,
        },
    }
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
    print(f"training_pool_rows={len(training_pool_rows)} saved={pool_path}")
    print(f"final_test_rows={len(final_test_rows)} saved={final_test_path}")
    for split_tag in split_tags:
        print(f"{split_tag}_saved={out_dir / split_tag}")
    print(f"protocol_manifest_saved={protocol_manifest_path}")


if __name__ == "__main__":
    main()
