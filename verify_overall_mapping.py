#!/usr/bin/env python3
"""Verify the Prompt 7 overall mapping without changing sealed predictions.

For each outer fold, the script fits ordinary least squares on that fold's
training pool only, using gold trait scores to predict the gold overall label.
It then checks whether the fitted fold-specific mapping changes any final e90
outer prediction relative to the canonical mapping in Prompt7Mapper.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any


CANONICAL_A = 1.32
CANONICAL_B = 3.04


def composite(row: dict[str, Any], keys: tuple[str, str, str, str]) -> int:
    ideas, organization, style, conventions = (int(row[key]) for key in keys)
    return (2 * ideas) + organization + style + conventions


def map_score(a: float, b: float, score: int) -> int:
    return max(2, min(24, int(round((a * score) + b))))


def fit_ols(rows: list[dict[str, Any]]) -> dict[str, float]:
    xs = [composite(row, ("gold_ideas", "gold_organization", "gold_style", "gold_conventions")) for row in rows]
    ys = [int(row["gold_overall"]) for row in rows]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Cannot fit overall mapping: training composite has zero variance.")
    a = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    b = y_mean - (a * x_mean)
    continuous = [(a * x) + b for x in xs]
    total_variance = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - (sum((prediction - gold) ** 2 for prediction, gold in zip(continuous, ys)) / total_variance)
    mae_continuous = mean(abs(prediction - gold) for prediction, gold in zip(continuous, ys))
    mae_rounded = mean(abs(map_score(a, b, x) - gold) for x, gold in zip(xs, ys))
    return {
        "a": a,
        "b": b,
        "r2": r2,
        "mae_continuous": mae_continuous,
        "mae_rounded": mae_rounded,
        "n_train": len(rows),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("sidecar_preds/v10ncv_clean_outer_e90_e79_e87_sealed_20260504/e90_primary_outer"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/overall_mapping_validation_20260904.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_reports: list[dict[str, Any]] = []
    total_predictions = 0
    total_changed = 0

    for outer_fold in range(5):
        train_path = args.data_root / f"outer{outer_fold}" / "sidecar_training_pool.json"
        train_rows = load_rows(train_path)
        fit = fit_ols(train_rows)
        all_scores_equivalent = all(
            map_score(fit["a"], fit["b"], score) == map_score(CANONICAL_A, CANONICAL_B, score)
            for score in range(16)
        )

        prediction_pattern = str(
            args.prediction_root
            / f"outer{outer_fold}"
            / "refit"
            / "seed*"
            / f"v10ncv_outer{outer_fold}_seed*_outer_selected_preds.json"
        )
        prediction_files = [Path(path) for path in sorted(glob.glob(prediction_pattern))]
        if not prediction_files:
            raise FileNotFoundError(f"No final e90 prediction files found for outer fold {outer_fold}.")

        changed = 0
        rows_seen = 0
        for prediction_file in prediction_files:
            for row in load_rows(prediction_file):
                score = composite(row, ("ideas", "organization", "style", "conventions"))
                recomputed = map_score(fit["a"], fit["b"], score)
                if recomputed != int(row["overall"]):
                    changed += 1
                rows_seen += 1

        total_predictions += rows_seen
        total_changed += changed
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "training_pool": str(train_path),
                "prediction_files": [str(path) for path in prediction_files],
                "discrete_scores_equivalent_to_canonical": all_scores_equivalent,
                "changed_final_prediction_rows": changed,
                "final_prediction_rows": rows_seen,
                **fit,
            }
        )

    report = {
        "purpose": "Post-hoc reproducibility validation; no source predictions are modified.",
        "canonical_mapping": {
            "formula": "clip_[2,24](round(1.32 * (2*ideas + organization + style + conventions) + 3.04))",
            "a": CANONICAL_A,
            "b": CANONICAL_B,
        },
        "fit_target": "gold_overall",
        "fit_features": "gold_ideas, gold_organization, gold_style, gold_conventions",
        "training_rule": "Each OLS fit uses only the corresponding outer training pool.",
        "folds": fold_reports,
        "summary": {
            "mean_a": mean(fold["a"] for fold in fold_reports),
            "mean_b": mean(fold["b"] for fold in fold_reports),
            "mean_r2": mean(fold["r2"] for fold in fold_reports),
            "mean_mae_continuous": mean(fold["mae_continuous"] for fold in fold_reports),
            "all_discrete_mappings_equivalent": all(fold["discrete_scores_equivalent_to_canonical"] for fold in fold_reports),
            "changed_final_prediction_rows": total_changed,
            "final_prediction_rows": total_predictions,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
