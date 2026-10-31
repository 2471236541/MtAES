#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


METRICS = ("overall", "ideas", "organization", "style", "conventions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect nested-CV inner and outer metrics.")
    p.add_argument("--pred-root", required=True, help="Prediction root from run_v10_nested_pipeline.sh")
    p.add_argument("--outer-folds", default="outer0,outer1,outer2,outer3,outer4")
    p.add_argument("--inner-splits", default="s0,s1,s2")
    p.add_argument("--seeds", required=True, help="Comma-separated integer seeds.")
    p.add_argument(
        "--allow-outer",
        action="store_true",
        help="Enable reading outer test cells. Keep off during inner-only tuning.",
    )
    p.add_argument("--prefix", default="v10ncv")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--rng-seed", type=int, default=20260420)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": std(values),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(values: list[float], rng: random.Random, n_boot: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return [means[lo_idx], means[hi_idx]]


def load_qwk_metrics(path: Path) -> dict[str, float]:
    d = load_json(path)
    trait = d.get("trait_qwk", {})
    return {
        "overall": float(d["overall_qwk"]),
        "ideas": float(trait.get("ideas", 0.0)),
        "organization": float(trait.get("organization", 0.0)),
        "style": float(trait.get("style", 0.0)),
        "conventions": float(trait.get("conventions", 0.0)),
    }


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {metric: summarize([row[metric] for row in rows]) for metric in METRICS}


def metric_mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {metric: mean([row[metric] for row in rows]) for metric in METRICS}


def diff_metrics(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {metric: left[metric] - right[metric] for metric in METRICS}


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)
    outer_folds = [x.strip() for x in args.outer_folds.split(",") if x.strip()]
    inner_splits = [x.strip() for x in args.inner_splits.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rng = random.Random(args.rng_seed)

    inner_cells: list[dict[str, Any]] = []
    outer_cells: list[dict[str, Any]] = []
    missing_inner: list[str] = []
    missing_outer: list[str] = []

    for outer in outer_folds:
        for split in inner_splits:
            for seed in seeds:
                out_dir = pred_root / outer / split / f"seed{seed}"
                dev_prefix = f"{args.prefix}_{outer}_{split}_seed{seed}_dev"
                holdout_prefix = f"{args.prefix}_{outer}_{split}_seed{seed}_holdout"
                dev_selected = out_dir / f"{dev_prefix}_selected_qwk.json"
                dev_baseline = out_dir / f"{dev_prefix}_baseline_qwk.json"
                holdout_selected = out_dir / f"{holdout_prefix}_selected_qwk.json"
                holdout_baseline = out_dir / f"{holdout_prefix}_baseline_qwk.json"
                if any(not p.exists() for p in (dev_selected, dev_baseline, holdout_selected, holdout_baseline)):
                    missing_inner.append(f"{outer}/{split}/seed{seed}")
                    continue

                dev_m = load_qwk_metrics(dev_selected)
                dev_b = load_qwk_metrics(dev_baseline)
                holdout_m = load_qwk_metrics(holdout_selected)
                holdout_b = load_qwk_metrics(holdout_baseline)
                inner_cells.append(
                    {
                        "outer": outer,
                        "split": split,
                        "seed": seed,
                        "dev": dev_m,
                        "dev_baseline": dev_b,
                        "holdout": holdout_m,
                        "holdout_baseline": holdout_b,
                        "dev_gain_vs_baseline": diff_metrics(dev_m, dev_b),
                        "holdout_gain_vs_baseline": diff_metrics(holdout_m, holdout_b),
                    }
                )

    if args.allow_outer:
        for outer in outer_folds:
            for seed in seeds:
                out_dir = pred_root / outer / "refit" / f"seed{seed}"
                outer_prefix = f"{args.prefix}_{outer}_seed{seed}_outer"
                outer_selected = out_dir / f"{outer_prefix}_selected_qwk.json"
                outer_baseline = out_dir / f"{outer_prefix}_baseline_qwk.json"
                if any(not p.exists() for p in (outer_selected, outer_baseline)):
                    missing_outer.append(f"{outer}/seed{seed}")
                    continue

                outer_m = load_qwk_metrics(outer_selected)
                outer_b = load_qwk_metrics(outer_baseline)
                outer_cells.append(
                    {
                        "outer": outer,
                        "seed": seed,
                        "outer_test": outer_m,
                        "outer_test_baseline": outer_b,
                        "outer_test_gain_vs_baseline": diff_metrics(outer_m, outer_b),
                    }
                )

    by_outer: dict[str, Any] = {}
    outer_fold_means: list[dict[str, Any]] = []
    for outer in outer_folds:
        inner_rows = [row for row in inner_cells if row["outer"] == outer]
        outer_rows = [row for row in outer_cells if row["outer"] == outer]

        outer_test_mean = metric_mean([row["outer_test"] for row in outer_rows]) if outer_rows else {m: 0.0 for m in METRICS}
        outer_test_baseline_mean = (
            metric_mean([row["outer_test_baseline"] for row in outer_rows]) if outer_rows else {m: 0.0 for m in METRICS}
        )
        outer_gain_mean = diff_metrics(outer_test_mean, outer_test_baseline_mean)
        if outer_rows:
            outer_fold_means.append(
                {
                    "outer": outer,
                    "outer_test": outer_test_mean,
                    "outer_test_baseline": outer_test_baseline_mean,
                    "outer_test_gain_vs_baseline": outer_gain_mean,
                }
            )

        by_outer[outer] = {
            "inner_completed_cells": len(inner_rows),
            "outer_completed_seeds": len(outer_rows),
            "inner_dev": summarize_metric_rows([row["dev"] for row in inner_rows]) if inner_rows else {},
            "inner_holdout": summarize_metric_rows([row["holdout"] for row in inner_rows]) if inner_rows else {},
            "outer_test_seed_level": summarize_metric_rows([row["outer_test"] for row in outer_rows]) if outer_rows else {},
            "outer_test_baseline_seed_level": (
                summarize_metric_rows([row["outer_test_baseline"] for row in outer_rows]) if outer_rows else {}
            ),
            "outer_test_gain_vs_baseline_seed_level": {
                metric: summarize([row["outer_test_gain_vs_baseline"][metric] for row in outer_rows])
                for metric in METRICS
            }
            if outer_rows
            else {},
            "outer_test_fold_mean": outer_test_mean if outer_rows else {},
            "outer_test_baseline_fold_mean": outer_test_baseline_mean if outer_rows else {},
            "outer_test_gain_vs_baseline_fold_mean": outer_gain_mean if outer_rows else {},
        }

    outer_fold_metric_summary = summarize_metric_rows([row["outer_test"] for row in outer_fold_means]) if outer_fold_means else {}
    outer_fold_metric_ci = {
        metric: bootstrap_ci([row["outer_test"][metric] for row in outer_fold_means], rng, args.bootstrap)
        for metric in METRICS
    }
    outer_fold_baseline_summary = (
        summarize_metric_rows([row["outer_test_baseline"] for row in outer_fold_means]) if outer_fold_means else {}
    )
    outer_fold_gain_summary = {
        metric: summarize([row["outer_test_gain_vs_baseline"][metric] for row in outer_fold_means])
        for metric in METRICS
    }
    outer_fold_gain_ci = {
        metric: bootstrap_ci([row["outer_test_gain_vs_baseline"][metric] for row in outer_fold_means], rng, args.bootstrap)
        for metric in METRICS
    }

    expected_outer_cells = len(outer_folds) * len(seeds) if args.allow_outer else 0
    payload: dict[str, Any] = {
        "meta": {
            "pred_root": str(pred_root),
            "outer_folds": outer_folds,
            "inner_splits": inner_splits,
            "seeds": seeds,
            "allow_outer_eval": args.allow_outer,
            "expected_inner_cells": len(outer_folds) * len(inner_splits) * len(seeds),
            "completed_inner_cells": len(inner_cells),
            "missing_inner_cells": missing_inner,
            "expected_outer_cells": expected_outer_cells,
            "completed_outer_cells": len(outer_cells),
            "missing_outer_cells": missing_outer,
            "bootstrap": args.bootstrap,
            "rng_seed": args.rng_seed,
        },
        "inner_selection": {
            "dev": summarize_metric_rows([row["dev"] for row in inner_cells]) if inner_cells else {},
            "holdout": summarize_metric_rows([row["holdout"] for row in inner_cells]) if inner_cells else {},
            "dev_baseline": summarize_metric_rows([row["dev_baseline"] for row in inner_cells]) if inner_cells else {},
            "holdout_baseline": (
                summarize_metric_rows([row["holdout_baseline"] for row in inner_cells]) if inner_cells else {}
            ),
            "dev_gain_vs_baseline": {
                metric: summarize([row["dev_gain_vs_baseline"][metric] for row in inner_cells])
                for metric in METRICS
            }
            if inner_cells
            else {},
            "holdout_gain_vs_baseline": {
                metric: summarize([row["holdout_gain_vs_baseline"][metric] for row in inner_cells])
                for metric in METRICS
            }
            if inner_cells
            else {},
        },
        "outer_test": {
            "fold_mean": outer_fold_metric_summary,
            "fold_mean_ci95_bootstrap": outer_fold_metric_ci,
            "fold_mean_baseline": outer_fold_baseline_summary,
            "fold_mean_gain_vs_baseline": outer_fold_gain_summary,
            "fold_mean_gain_vs_baseline_ci95_bootstrap": outer_fold_gain_ci,
            "seed_level": summarize_metric_rows([row["outer_test"] for row in outer_cells]) if outer_cells else {},
            "seed_level_baseline": (
                summarize_metric_rows([row["outer_test_baseline"] for row in outer_cells]) if outer_cells else {}
            ),
        },
        "by_outer": by_outer,
        "outer_fold_means": outer_fold_means,
        "inner_cells": inner_cells,
        "outer_cells": outer_cells,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    outer_overall = payload["outer_test"]["fold_mean"].get("overall", {})
    inner_holdout_overall = payload["inner_selection"]["holdout"].get("overall", {})
    print(f"saved: {out_path}")
    print(
        "summary "
        f"outer_completed={len(outer_fold_means)}/{len(outer_folds)} "
        f"outer_mean={outer_overall.get('mean', 0.0):.4f} "
        f"outer_std={outer_overall.get('std', 0.0):.4f} "
        f"inner_holdout_mean={inner_holdout_overall.get('mean', 0.0):.4f}"
    )
    if not args.allow_outer:
        print("note: outer metrics skipped (inner-only mode)")


if __name__ == "__main__":
    main()
