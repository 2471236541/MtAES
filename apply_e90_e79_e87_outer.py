#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from prompt7_mapper import Prompt7Mapper


METRICS = ("overall", "ideas", "organization", "style", "conventions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen e90 e79+e87 both-direction ideas rule on sealed outer predictions.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--donor-roots", required=True, help="Comma-separated donor roots, fixed order e79,e87.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--outer-folds", default="outer0,outer1,outer2,outer3,outer4")
    parser.add_argument("--seeds", default="20260325,20260412")
    parser.add_argument("--prefix", default="v10ncv")
    parser.add_argument("--label", default="e90_e79_e87_all_both_both_dir_up0m0_dn0m0_step3")
    parser.add_argument("--bucket-word-threshold", type=int, default=180)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--rng-seed", type=int, default=20260504)
    parser.add_argument("--scope-note", default="sealed outer primary: frozen e90 e79+e87 both-direction ideas donor rule; no outer-based tuning")
    return parser.parse_args()


def split_items(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).replace(" ", ",").split(",") if x.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def qwk(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    if mn == mx:
        return 1.0
    n = mx - mn + 1
    total = len(y_true)
    obs = [[0.0] * n for _ in range(n)]
    true_hist = [0.0] * n
    pred_hist = [0.0] * n
    for true, pred in zip(y_true, y_pred):
        obs[int(true) - mn][int(pred) - mn] += 1.0
        true_hist[int(true) - mn] += 1.0
        pred_hist[int(pred) - mn] += 1.0
    scale = float((n - 1) ** 2) if n > 1 else 1.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / scale
            numerator += weight * obs[i][j]
            denominator += weight * true_hist[i] * pred_hist[j] / total
    return 1.0 - numerator / denominator if denominator else 0.0


def metric(rows: list[dict[str, Any]], metric_name: str) -> float:
    return qwk([int(row[f"gold_{metric_name}"]) for row in rows], [int(row[metric_name]) for row in rows])


def metric_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt": 7,
        "samples": len(rows),
        "skipped": 0,
        "overall_qwk": metric(rows, "overall"),
        "ideas_qwk": metric(rows, "ideas"),
        "organization_qwk": metric(rows, "organization"),
        "style_qwk": metric(rows, "style"),
        "conventions_qwk": metric(rows, "conventions"),
        "trait_qwk": {trait: metric(rows, trait) for trait in ("ideas", "organization", "style", "conventions")},
    }


def metric_block(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric_name: metric(rows, metric_name) for metric_name in METRICS}


def mae_block(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric_name: sum(abs(int(row[metric_name]) - int(row[f"gold_{metric_name}"])) for row in rows) / len(rows)
        if rows
        else 0.0
        for metric_name in METRICS
    }


def wc(row: dict[str, Any]) -> int:
    return len(str(row.get("essay_text", "")).split())


def donor_pred(row: dict[str, Any]) -> int:
    return int(row.get("bert_ideas_pred", row.get("ideas")))


def donor_prob(row: dict[str, Any]) -> float:
    return float(row.get("bert_ideas_max_prob", 0.0))


def donor_margin(row: dict[str, Any]) -> float:
    return float(row.get("bert_ideas_margin", 0.0))


def recompute(row: dict[str, Any]) -> None:
    row["overall"] = int(
        Prompt7Mapper.compute_overall_score(
            int(row["ideas"]),
            int(row["organization"]),
            int(row["style"]),
            int(row["conventions"]),
        )
    )
    row["pred_overall"] = int(row["overall"])


def choose_directional_replacement(
    base_ideas: int,
    donor_items: list[tuple[str, dict[str, Any]]],
    *,
    max_step: int,
) -> dict[str, Any] | None:
    replacements: list[dict[str, Any]] = []
    for direction in ("up", "down"):
        if direction == "up":
            valid = [(name, row) for name, row in donor_items if donor_pred(row) > base_ideas]
        else:
            valid = [(name, row) for name, row in donor_items if donor_pred(row) < base_ideas]
        if len(valid) < 2:
            continue
        chosen_name, chosen_row = max(
            valid,
            key=lambda item: (donor_prob(item[1]), donor_margin(item[1]), abs(donor_pred(item[1]) - base_ideas)),
        )
        raw_pred = donor_pred(chosen_row)
        if max_step > 0:
            if direction == "up":
                raw_pred = min(raw_pred, base_ideas + max_step)
            else:
                raw_pred = max(raw_pred, base_ideas - max_step)
        if raw_pred == base_ideas:
            continue
        replacements.append(
            {
                "pred": int(raw_pred),
                "prob": donor_prob(chosen_row),
                "margin": donor_margin(chosen_row),
                "direction": direction,
                "chosen_donor": chosen_name,
                "donors": sorted(name for name, _ in valid),
            }
        )
    if not replacements:
        return None
    return max(replacements, key=lambda item: (float(item["prob"]), float(item["margin"]), 1 if item["direction"] == "up" else 0))


def apply_rule(
    base_rows: list[dict[str, Any]],
    donor_rows_by_name: dict[str, list[dict[str, Any]]],
    *,
    label: str,
    max_step: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    donor_maps = {
        name: {int(row["essay_id"]): row for row in rows}
        for name, rows in donor_rows_by_name.items()
    }
    counts = {
        "rows": 0,
        "changed": 0,
        "up": 0,
        "down": 0,
        "short_changed": 0,
        "long_changed": 0,
        "short_up": 0,
        "short_down": 0,
        "long_up": 0,
        "long_down": 0,
        "base_kept": 0,
    }
    out: list[dict[str, Any]] = []
    for base in base_rows:
        essay_id = int(base["essay_id"])
        donor_items = []
        for donor_name, donor_map in donor_maps.items():
            if essay_id not in donor_map:
                raise KeyError(f"missing donor={donor_name} essay_id={essay_id}")
            donor_items.append((donor_name, donor_map[essay_id]))

        row = dict(base)
        counts["rows"] += 1
        bucket = "short" if wc(base) <= 180 else "long"
        replacement = choose_directional_replacement(int(base["ideas"]), donor_items, max_step=max_step)
        row["e90_rule"] = label
        row["e90_applied"] = False
        if replacement is None:
            counts["base_kept"] += 1
            out.append(row)
            continue

        direction = str(replacement["direction"])
        row["e90_applied"] = True
        row["e90_direction"] = direction
        row["e90_ideas_from"] = int(row["ideas"])
        row["e90_ideas_to"] = int(replacement["pred"])
        row["e90_donors"] = list(replacement["donors"])
        row["e90_chosen_donor"] = str(replacement["chosen_donor"])
        row["e90_prob"] = float(replacement["prob"])
        row["e90_margin"] = float(replacement["margin"])
        row["ideas"] = int(replacement["pred"])
        row["pred_ideas"] = int(replacement["pred"])
        recompute(row)

        counts["changed"] += 1
        counts[direction] += 1
        counts[f"{bucket}_changed"] += 1
        counts[f"{bucket}_{direction}"] += 1
        out.append(row)
    return out, counts


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    avg = mean(values)
    return {
        "n": len(values),
        "mean": avg,
        "std": math.sqrt(sum((x - avg) ** 2 for x in values) / len(values)),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(values: list[float], *, n_boot: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    means = [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(max(1, int(n_boot)))]
    means.sort()
    lo = int(0.025 * (len(means) - 1))
    hi = int(0.975 * (len(means) - 1))
    return [means[lo], means[hi]]


def diff(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {metric_name: float(left[metric_name] - right[metric_name]) for metric_name in METRICS}


def bucket_metrics(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, predicate in (
        ("short", lambda row: wc(row) <= threshold),
        ("long", lambda row: wc(row) > threshold),
    ):
        base_bucket = [row for row in base_rows if predicate(row)]
        cand_bucket = [row for row in cand_rows if predicate(row)]
        result[name] = {
            "rows": len(cand_bucket),
            "base": metric_block(base_bucket) if base_bucket else {metric_name: 0.0 for metric_name in METRICS},
            "candidate": metric_block(cand_bucket) if cand_bucket else {metric_name: 0.0 for metric_name in METRICS},
        }
        result[name]["candidate_gain_vs_base"] = diff(result[name]["candidate"], result[name]["base"])
    return result


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    donor_roots = [Path(item) for item in split_items(args.donor_roots)]
    donor_names = ["e79", "e87"]
    if len(donor_roots) != len(donor_names):
        raise ValueError("e90 sealed rule requires exactly two donor roots: e79,e87")
    output_root = Path(args.output_root)
    outer_folds = split_items(args.outer_folds)
    seeds = [int(item) for item in split_items(args.seeds)]
    rule = {
        "label": str(args.label),
        "donors": donor_names,
        "short_up_mode": "both_dir",
        "short_down_mode": "both_dir",
        "long_up_mode": "both_dir",
        "long_down_mode": "both_dir",
        "short_up_prob": 0.0,
        "short_up_margin": 0.0,
        "long_up_prob": 0.0,
        "long_up_margin": 0.0,
        "short_down_prob": 0.0,
        "short_down_margin": 0.0,
        "long_down_prob": 0.0,
        "long_down_margin": 0.0,
        "max_step": int(args.max_step),
    }

    seed_cells: list[dict[str, Any]] = []
    totals = {
        "rows": 0,
        "changed": 0,
        "up": 0,
        "down": 0,
        "short_changed": 0,
        "long_changed": 0,
        "short_up": 0,
        "short_down": 0,
        "long_up": 0,
        "long_down": 0,
        "base_kept": 0,
    }
    for outer in outer_folds:
        for seed in seeds:
            stem = f"{args.prefix}_{outer}_seed{seed}_outer"
            base_dir = base_root / outer / "refit" / f"seed{seed}"
            donor_dirs = [root / outer / "refit" / f"seed{seed}" for root in donor_roots]
            out_dir = output_root / outer / "refit" / f"seed{seed}"
            base_path = base_dir / f"{stem}_selected_preds.json"
            donor_paths = [donor_dir / f"{stem}_ideas_with_bert.json" for donor_dir in donor_dirs]
            for path_label, path in [("base", base_path), *[(name, path) for name, path in zip(donor_names, donor_paths)]]:
                if not path.exists():
                    raise FileNotFoundError(f"Missing {path_label}: {path}")

            base_rows = read_json(base_path)
            donor_rows = {name: read_json(path) for name, path in zip(donor_names, donor_paths)}
            cand_rows, counts = apply_rule(base_rows, donor_rows, label=str(args.label), max_step=int(args.max_step))
            for key in totals:
                totals[key] += int(counts[key])

            pred_path = out_dir / f"{stem}_selected_preds.json"
            qwk_path = out_dir / f"{stem}_selected_qwk.json"
            report_path = out_dir / f"{stem}_selected_report.json"
            write_json(pred_path, cand_rows)
            write_json(qwk_path, metric_payload(cand_rows))

            base_metrics = metric_block(base_rows)
            cand_metrics = metric_block(cand_rows)
            base_mae = mae_block(base_rows)
            cand_mae = mae_block(cand_rows)
            report = {
                "scope": args.scope_note,
                "outer_test_touched": True,
                "selection_on_this_split": False,
                "selection_uses_gold": False,
                "rule": rule,
                "paths": {
                    "base_preds": str(base_path),
                    "donor_preds": [str(path) for path in donor_paths],
                    "output_preds": str(pred_path),
                },
                "counts": counts,
                "base": base_metrics,
                "candidate": cand_metrics,
                "candidate_gain_vs_base": diff(cand_metrics, base_metrics),
                "base_mae": base_mae,
                "candidate_mae": cand_mae,
                "candidate_mae_gain_vs_base": diff(cand_mae, base_mae),
                "buckets": bucket_metrics(base_rows, cand_rows, int(args.bucket_word_threshold)),
            }
            write_json(report_path, report)
            seed_cells.append({"outer": outer, "seed": seed, "stem": stem, **report})

    fold_rows: list[dict[str, Any]] = []
    for outer in outer_folds:
        rows = [row for row in seed_cells if row["outer"] == outer]
        if not rows:
            continue
        fold: dict[str, Any] = {"outer": outer, "seeds": [int(row["seed"]) for row in rows]}
        for side in ("base", "candidate", "candidate_gain_vs_base", "base_mae", "candidate_mae", "candidate_mae_gain_vs_base"):
            fold[side] = {metric_name: mean([float(row[side][metric_name]) for row in rows]) for metric_name in METRICS}
        fold["counts"] = {key: sum(int(row["counts"][key]) for row in rows) for key in totals}
        fold_rows.append(fold)

    def metric_summary(side: str) -> dict[str, Any]:
        return {metric_name: summarize([float(row[side][metric_name]) for row in fold_rows]) for metric_name in METRICS}

    def metric_ci(side: str) -> dict[str, list[float]]:
        return {
            metric_name: bootstrap_ci(
                [float(row[side][metric_name]) for row in fold_rows],
                n_boot=int(args.bootstrap),
                seed=int(args.rng_seed),
            )
            for metric_name in METRICS
        }

    summary = {
        "scope": args.scope_note,
        "outer_test_touched": True,
        "selection_on_outer": False,
        "selection_uses_gold": False,
        "rule": rule,
        "paths": {
            "base_root": str(base_root),
            "donor_roots": [str(path) for path in donor_roots],
            "output_root": str(output_root),
        },
        "seed_cells": seed_cells,
        "fold_rows": fold_rows,
        "counts": totals
        | {
            "changed_ratio": totals["changed"] / totals["rows"] if totals["rows"] else 0.0,
        },
        "fold_level": {
            "base": metric_summary("base"),
            "candidate": metric_summary("candidate"),
            "candidate_gain_vs_base": metric_summary("candidate_gain_vs_base"),
            "base_mae": metric_summary("base_mae"),
            "candidate_mae": metric_summary("candidate_mae"),
            "candidate_mae_gain_vs_base": metric_summary("candidate_mae_gain_vs_base"),
            "candidate_ci95": metric_ci("candidate"),
            "candidate_gain_vs_base_ci95": metric_ci("candidate_gain_vs_base"),
        },
    }
    write_json(Path(args.summary_output), summary)
    overall = summary["fold_level"]["candidate"]["overall"]
    gain = summary["fold_level"]["candidate_gain_vs_base"]["overall"]
    print(f"rule={args.label}")
    print(f"outer_folds={len(fold_rows)} seed_cells={len(seed_cells)} rows={totals['rows']}")
    print(f"changed={totals['changed']} ({summary['counts']['changed_ratio']:.2%}) up={totals['up']} down={totals['down']}")
    print(
        "outer_candidate "
        f"overall={overall['mean']:.6f} std={overall['std']:.6f} "
        f"ci95=[{summary['fold_level']['candidate_ci95']['overall'][0]:.6f},"
        f"{summary['fold_level']['candidate_ci95']['overall'][1]:.6f}]"
    )
    print(
        "gain_vs_base "
        f"overall={gain['mean']:+.6f} "
        f"ci95=[{summary['fold_level']['candidate_gain_vs_base_ci95']['overall'][0]:+.6f},"
        f"{summary['fold_level']['candidate_gain_vs_base_ci95']['overall'][1]:+.6f}]"
    )
    print(f"summary_saved: {args.summary_output}")


if __name__ == "__main__":
    main()
