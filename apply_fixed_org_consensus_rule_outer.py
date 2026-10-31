#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from select_and_apply_org_overlay_rules_oof import compare_rows, load_json, qwk_payload, save_json, seed_csv, split_csv
from select_org_consensus_rules_dev import ConsensusRule, apply_consensus


METRICS = ("overall", "ideas", "organization", "style", "conventions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply one fixed organization consensus rule to sealed outer/refit predictions."
    )
    p.add_argument("--base-root", required=True)
    p.add_argument("--donor-a-root", required=True)
    p.add_argument("--donor-b-root", required=True)
    p.add_argument("--reference-root", default="")
    p.add_argument("--output-root", required=True)
    p.add_argument("--donor-a-name", required=True)
    p.add_argument("--donor-b-name", required=True)
    p.add_argument("--donor-a-template", required=True)
    p.add_argument("--donor-b-template", required=True)
    p.add_argument("--outer-folds", default="outer0,outer1,outer2,outer3,outer4")
    p.add_argument("--seeds", required=True)
    p.add_argument("--prefix", default="v10ncv")
    p.add_argument("--direction", required=True, choices=("changed", "no_down", "up", "down"))
    p.add_argument("--prob-ge", type=float, required=True)
    p.add_argument("--margin-ge", type=float, required=True)
    p.add_argument("--prob-mode", required=True, choices=("both", "avg"))
    p.add_argument("--margin-mode", required=True, choices=("both", "avg"))
    p.add_argument("--word-filter", default="all")
    p.add_argument("--bucket-word-threshold", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--rng-seed", type=int, default=20260427)
    p.add_argument("--summary-output", default="")
    p.add_argument(
        "--scope-note",
        default="fixed dev-selected e59 organization consensus rule applied once on sealed outer_test",
    )
    return p.parse_args()


def render(template: str, stem_value: str) -> str:
    return str(template).format(stem=stem_value)


def asdict_rule(rule: ConsensusRule) -> dict[str, Any]:
    return {
        "donor_a": rule.donor_a,
        "donor_b": rule.donor_b,
        "direction": rule.direction,
        "prob_ge": rule.prob_ge,
        "margin_ge": rule.margin_ge,
        "prob_mode": rule.prob_mode,
        "margin_mode": rule.margin_mode,
        "word_filter": rule.word_filter,
        "tag": rule.tag,
    }


def metric_from_qwk(path: Path) -> dict[str, float]:
    d = load_json(path)
    trait = d.get("trait_qwk", {})
    return {
        "overall": float(d["overall_qwk"]),
        "ideas": float(trait.get("ideas", 0.0)),
        "organization": float(trait.get("organization", 0.0)),
        "style": float(trait.get("style", 0.0)),
        "conventions": float(trait.get("conventions", 0.0)),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    m = mean(values)
    return {
        "n": len(values),
        "mean": m,
        "std": math.sqrt(sum((x - m) ** 2 for x in values) / len(values)),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(values: list[float], *, n_boot: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(max(1, int(n_boot))):
        means.append(mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = int(0.025 * (len(means) - 1))
    hi = int(0.975 * (len(means) - 1))
    return [means[lo], means[hi]]


def diff(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {metric: float(left[metric] - right[metric]) for metric in METRICS}


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    donor_a_root = Path(args.donor_a_root)
    donor_b_root = Path(args.donor_b_root)
    reference_root = Path(args.reference_root) if str(args.reference_root).strip() else None
    output_root = Path(args.output_root)
    rule = ConsensusRule(
        donor_a=str(args.donor_a_name),
        donor_b=str(args.donor_b_name),
        direction=str(args.direction),
        prob_ge=float(args.prob_ge),
        margin_ge=float(args.margin_ge),
        prob_mode=str(args.prob_mode),
        margin_mode=str(args.margin_mode),
        word_filter=str(args.word_filter),
    )

    seed_cells: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for outer in split_csv(args.outer_folds):
        for seed in seed_csv(args.seeds):
            s = f"{args.prefix}_{outer}_seed{seed}_outer"
            base_dir = base_root / outer / "refit" / f"seed{seed}"
            donor_a_dir = donor_a_root / outer / "refit" / f"seed{seed}"
            donor_b_dir = donor_b_root / outer / "refit" / f"seed{seed}"
            out_dir = output_root / outer / "refit" / f"seed{seed}"

            base_path = base_dir / f"{s}_selected_preds.json"
            base_qwk_path = base_dir / f"{s}_selected_qwk.json"
            donor_a_path = donor_a_dir / render(args.donor_a_template, s)
            donor_b_path = donor_b_dir / render(args.donor_b_template, s)
            reference_path = reference_root / outer / "refit" / f"seed{seed}" / f"{s}_selected_preds.json" if reference_root else base_path
            reference_qwk_path = reference_root / outer / "refit" / f"seed{seed}" / f"{s}_selected_qwk.json" if reference_root else base_qwk_path
            for label, path in (
                ("base", base_path),
                ("base_qwk", base_qwk_path),
                ("donor_a", donor_a_path),
                ("donor_b", donor_b_path),
                ("reference", reference_path),
                ("reference_qwk", reference_qwk_path),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Missing {label}: {path}")

            out_rows, counts = apply_consensus(load_json(base_path), load_json(donor_a_path), load_json(donor_b_path), rule)
            pred_path = out_dir / f"{s}_selected_preds.json"
            qwk_path = out_dir / f"{s}_selected_qwk.json"
            report_path = out_dir / f"{s}_selected_report.json"
            save_json(pred_path, out_rows)
            save_json(qwk_path, qwk_payload(out_rows, pred_path))
            save_json(
                report_path,
                {
                    "scope": args.scope_note,
                    "outer_test_touched": True,
                    "selection_on_this_split": False,
                    "selection_uses_gold": False,
                    "rule": asdict_rule(rule),
                    "paths": {
                        "base_preds": str(base_path),
                        "donor_a_preds": str(donor_a_path),
                        "donor_b_preds": str(donor_b_path),
                        "reference_preds": str(reference_path),
                        "output_preds": str(pred_path),
                    },
                    "counts": counts,
                },
            )

            cmp = compare_rows(out_rows, load_json(reference_path), int(args.bucket_word_threshold))
            base_m = metric_from_qwk(base_qwk_path)
            ref_m = metric_from_qwk(reference_qwk_path)
            cand_m = metric_from_qwk(qwk_path)
            seed_cells.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "stem": s,
                    "counts": counts,
                    "base": base_m,
                    "reference": ref_m,
                    "candidate": cand_m,
                    "candidate_gain_vs_base": diff(cand_m, base_m),
                    "candidate_gain_vs_reference": diff(cand_m, ref_m),
                    **cmp,
                }
            )
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + int(value)

    fold_rows: list[dict[str, Any]] = []
    for outer in split_csv(args.outer_folds):
        rows = [row for row in seed_cells if row["outer"] == outer]
        if not rows:
            continue
        fold = {"outer": outer}
        for side in ("base", "reference", "candidate", "candidate_gain_vs_base", "candidate_gain_vs_reference"):
            fold[side] = {metric: mean([float(row[side][metric]) for row in rows]) for metric in METRICS}
        fold["seeds"] = [int(row["seed"]) for row in rows]
        fold_rows.append(fold)

    def metric_summary(side: str) -> dict[str, Any]:
        return {metric: summarize([float(row[side][metric]) for row in fold_rows]) for metric in METRICS}

    def metric_ci(side: str) -> dict[str, list[float]]:
        return {
            metric: bootstrap_ci(
                [float(row[side][metric]) for row in fold_rows],
                n_boot=int(args.bootstrap),
                seed=int(args.rng_seed),
            )
            for metric in METRICS
        }

    summary = {
        "scope": args.scope_note,
        "outer_test_touched": True,
        "selection_on_outer": False,
        "selection_uses_gold": False,
        "rule": asdict_rule(rule),
        "paths": {
            "base_root": str(base_root),
            "donor_a_root": str(donor_a_root),
            "donor_b_root": str(donor_b_root),
            "reference_root": str(reference_root) if reference_root else "",
            "output_root": str(output_root),
        },
        "seed_cells": seed_cells,
        "fold_rows": fold_rows,
        "counts": totals
        | {
            "changed_ratio": totals.get("changed_rows", 0) / totals.get("rows", 1),
            "agreement_ratio": totals.get("agreement_rows", 0) / totals.get("rows", 1),
        },
        "fold_level": {
            "base": metric_summary("base"),
            "reference": metric_summary("reference"),
            "candidate": metric_summary("candidate"),
            "candidate_gain_vs_base": metric_summary("candidate_gain_vs_base"),
            "candidate_gain_vs_reference": metric_summary("candidate_gain_vs_reference"),
            "candidate_ci95": metric_ci("candidate"),
            "candidate_gain_vs_base_ci95": metric_ci("candidate_gain_vs_base"),
            "candidate_gain_vs_reference_ci95": metric_ci("candidate_gain_vs_reference"),
        },
    }
    summary_path = Path(args.summary_output) if str(args.summary_output).strip() else output_root / "outer_e59_summary.json"
    save_json(summary_path, summary)
    overall = summary["fold_level"]["candidate"]["overall"]
    gain_base = summary["fold_level"]["candidate_gain_vs_base"]["overall"]
    print(f"rule={rule.tag}")
    print(f"outer_folds={len(fold_rows)} seed_cells={len(seed_cells)} rows={totals.get('rows', 0)}")
    print(f"changed={totals.get('changed_rows', 0)} ({summary['counts']['changed_ratio']:.2%})")
    print(
        "outer_candidate "
        f"overall={overall['mean']:.6f} std={overall['std']:.6f} "
        f"ci95=[{summary['fold_level']['candidate_ci95']['overall'][0]:.6f},"
        f"{summary['fold_level']['candidate_ci95']['overall'][1]:.6f}]"
    )
    print(
        "gain_vs_base "
        f"overall={gain_base['mean']:+.6f} "
        f"ci95=[{summary['fold_level']['candidate_gain_vs_base_ci95']['overall'][0]:+.6f},"
        f"{summary['fold_level']['candidate_gain_vs_base_ci95']['overall'][1]:+.6f}]"
    )
    print(f"summary_saved: {summary_path}")


if __name__ == "__main__":
    main()
