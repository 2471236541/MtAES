#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Prompt7 QWK from prediction JSON.

Supports:
1) Rows already containing gold_* fields.
2) Rows with essay_id only (gold loaded from ASAP TSV).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Predictions JSON path.")
    parser.add_argument("--asap-data", default="data/training_set_rel3.tsv", help="ASAP TSV path.")
    parser.add_argument("--prompt-id", type=int, default=7)
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    return parser.parse_args()


def load_gold_map(tsv_path: Path, prompt_id: int) -> dict[int, dict[str, int]]:
    if not tsv_path.exists():
        raise FileNotFoundError(f"ASAP TSV not found: {tsv_path}")
    gold = {}
    with open(tsv_path, "r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                if int(row["essay_set"]) != prompt_id:
                    continue
                essay_id = int(row["essay_id"])
                gold[essay_id] = {
                    "ideas": int(row["rater1_trait1"]),
                    "organization": int(row["rater1_trait2"]),
                    "style": int(row["rater1_trait3"]),
                    "conventions": int(row["rater1_trait4"]),
                    "overall": int(row["domain1_score"]),
                }
            except Exception:
                continue
    return gold


def get_int(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in row:
            try:
                return int(row[key])
            except Exception:
                continue
    return None


def qwk(y_true: list[int], y_pred: list[int], min_rating: int, max_rating: int) -> float:
    n = max_rating - min_rating + 1
    N = len(y_true)
    if N == 0:
        return 0.0

    O = [[0.0 for _ in range(n)] for _ in range(n)]
    for a, b in zip(y_true, y_pred):
        ai = max(min_rating, min(max_rating, int(a))) - min_rating
        bi = max(min_rating, min(max_rating, int(b))) - min_rating
        O[ai][bi] += 1.0

    hist_true = [0.0 for _ in range(n)]
    hist_pred = [0.0 for _ in range(n)]
    for a in y_true:
        hist_true[max(min_rating, min(max_rating, int(a))) - min_rating] += 1.0
    for b in y_pred:
        hist_pred[max(min_rating, min(max_rating, int(b))) - min_rating] += 1.0

    E = [[(hist_true[i] * hist_pred[j]) / float(N) for j in range(n)] for i in range(n)]
    W = [[0.0 for _ in range(n)] for _ in range(n)]
    denom = float((n - 1) ** 2) if n > 1 else 1.0
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i - j) ** 2) / denom

    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            num += W[i][j] * O[i][j]
            den += W[i][j] * E[i][j]
    if den == 0.0:
        return 0.0
    return 1.0 - (num / den)


def evaluate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    trait_names = ("ideas", "organization", "style", "conventions")
    trait_qwk = {}
    for trait in trait_names:
        y_true = [int(s["gold_trait_scores"][trait]) for s in samples]
        y_pred = [int(s["pred_trait_scores"][trait]) for s in samples]
        trait_qwk[trait] = float(qwk(y_true, y_pred, 0, 3))

    y_true_overall = [int(s["gold_overall_score"]) for s in samples]
    y_pred_overall = [int(s["pred_overall_score"]) for s in samples]
    overall_qwk = float(qwk(y_true_overall, y_pred_overall, 0, 30))
    return {
        "trait_qwk": trait_qwk,
        "overall_qwk": overall_qwk,
    }


def main() -> None:
    args = parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    rows = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Predictions JSON must be a non-empty list.")

    gold_map = load_gold_map(Path(args.asap_data), args.prompt_id)
    samples = []
    skipped = 0

    for row in rows:
        essay_id = get_int(row, ("essay_id",))
        if essay_id is None:
            skipped += 1
            continue

        pred_ideas = get_int(row, ("ideas", "pred_ideas"))
        pred_org = get_int(row, ("organization", "pred_organization"))
        pred_style = get_int(row, ("style", "pred_style"))
        pred_conv = get_int(row, ("conventions", "pred_conventions"))
        pred_overall = get_int(row, ("overall", "pred_overall"))
        if None in (pred_ideas, pred_org, pred_style, pred_conv, pred_overall):
            skipped += 1
            continue

        if all(k in row for k in ("gold_ideas", "gold_organization", "gold_style", "gold_conventions", "gold_overall")):
            gold = {
                "ideas": int(row["gold_ideas"]),
                "organization": int(row["gold_organization"]),
                "style": int(row["gold_style"]),
                "conventions": int(row["gold_conventions"]),
                "overall": int(row["gold_overall"]),
            }
        else:
            gold = gold_map.get(essay_id)
            if gold is None:
                skipped += 1
                continue

        samples.append(
            {
                "prompt_id": int(args.prompt_id),
                "gold_trait_scores": {
                    "ideas": gold["ideas"],
                    "organization": gold["organization"],
                    "style": gold["style"],
                    "conventions": gold["conventions"],
                },
                "pred_trait_scores": {
                    "ideas": pred_ideas,
                    "organization": pred_org,
                    "style": pred_style,
                    "conventions": pred_conv,
                },
                "gold_overall_score": gold["overall"],
                "pred_overall_score": pred_overall,
            }
        )

    if not samples:
        raise ValueError("No valid rows to evaluate.")

    result = evaluate_samples(samples)
    result["prompt_id"] = int(args.prompt_id)
    result["num_samples"] = int(len(samples))
    result["skipped_rows"] = int(skipped)
    result["predictions_path"] = str(pred_path)

    print(f"prompt={result['prompt_id']} samples={result['num_samples']} skipped={result['skipped_rows']}")
    for trait, score in result["trait_qwk"].items():
        print(f"{trait}_qwk={score:.4f}")
    print(f"overall_qwk={result['overall_qwk']:.4f}")

    if args.output.strip():
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"prompt7_qwk_results_sidecar_{ts}.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
