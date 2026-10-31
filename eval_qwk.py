#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight QWK evaluator for already-generated prediction JSON.

Examples:
  python eval_qwk.py \
      --pred sidecar_preds/.../selected_preds.json \
      --gold sidecar_data/.../sidecar_holdout.json

  python eval_qwk.py \
      --pred sidecar_preds/.../selected_preds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_qwk import qwk


TRAITS = ("ideas", "organization", "style", "conventions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate overall/trait QWK from existing JSON files.")
    parser.add_argument("--pred", required=True, help="Prediction JSON path, e.g. *_selected_preds.json")
    parser.add_argument(
        "--gold",
        default="",
        help="Optional gold JSON path. When omitted, gold_* fields must already exist in --pred rows.",
    )
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"JSON must be a non-empty list: {p}")
    return rows


def get_int(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in row:
            try:
                return int(row[key])
            except Exception:
                continue
    return None


def build_gold_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    gold_map: dict[int, dict[str, int]] = {}
    for row in rows:
        essay_id = get_int(row, ("essay_id",))
        if essay_id is None:
            continue
        gold_item: dict[str, int] = {}
        ok = True
        for trait in TRAITS:
            value = get_int(row, (f"gold_{trait}", trait))
            if value is None:
                ok = False
                break
            gold_item[trait] = value
        overall = get_int(row, ("gold_overall", "gold_overall_score", "overall"))
        if overall is None:
            ok = False
        if ok:
            gold_item["overall"] = int(overall)
            gold_map[int(essay_id)] = gold_item
    return gold_map


def evaluate(pred_rows: list[dict[str, Any]], gold_map: dict[int, dict[str, int]]) -> dict[str, Any]:
    trait_true: dict[str, list[int]] = {trait: [] for trait in TRAITS}
    trait_pred: dict[str, list[int]] = {trait: [] for trait in TRAITS}
    overall_true: list[int] = []
    overall_pred: list[int] = []
    skipped = 0

    for row in pred_rows:
        essay_id = get_int(row, ("essay_id",))
        if essay_id is None:
            skipped += 1
            continue
        gold = gold_map.get(int(essay_id))
        if gold is None:
            skipped += 1
            continue

        pred_traits: dict[str, int] = {}
        ok = True
        for trait in TRAITS:
            value = get_int(row, (trait, f"pred_{trait}"))
            if value is None:
                ok = False
                break
            pred_traits[trait] = value
        pred_overall = get_int(row, ("overall", "pred_overall"))
        if not ok or pred_overall is None:
            skipped += 1
            continue

        for trait in TRAITS:
            trait_true[trait].append(int(gold[trait]))
            trait_pred[trait].append(int(pred_traits[trait]))
        overall_true.append(int(gold["overall"]))
        overall_pred.append(int(pred_overall))

    if not overall_true:
        raise ValueError("No valid matched rows found between --pred and --gold.")

    trait_qwk = {
        trait: float(qwk(trait_true[trait], trait_pred[trait], 0, 3))
        for trait in TRAITS
    }
    result = {
        "num_samples": int(len(overall_true)),
        "skipped_rows": int(skipped),
        "trait_qwk": trait_qwk,
        "overall_qwk": float(qwk(overall_true, overall_pred, 0, 30)),
    }
    return result


def main() -> None:
    args = parse_args()
    pred_rows = load_rows(args.pred)
    if args.gold.strip():
        gold_rows = load_rows(args.gold)
        gold_map = build_gold_map(gold_rows)
    else:
        gold_map = build_gold_map(pred_rows)
    if not gold_map:
        raise ValueError("Could not build gold labels. Provide --gold or ensure --pred contains gold_* fields.")

    result = evaluate(pred_rows, gold_map)
    result["pred_path"] = str(Path(args.pred))
    result["gold_path"] = str(Path(args.gold)) if args.gold.strip() else ""

    print(f"num_samples: {result['num_samples']}")
    print(f"skipped_rows: {result['skipped_rows']}")
    print(f"overall_qwk: {result['overall_qwk']:.4f}")
    for trait in TRAITS:
        print(f"{trait}_qwk: {result['trait_qwk'][trait]:.4f}")

    if args.output.strip():
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
