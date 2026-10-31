#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对单个 sidecar 模型的 style 预测做门控融合。

用途：
- 固定 ideas/organization/conventions。
- 仅当 sidecar 置信度通过门控时替换 style。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prompt7_mapper import Prompt7Mapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input", required=True, help="Base predictions JSON (contains ideas/org/style/conventions).")
    parser.add_argument("--model-input", required=True, help="Inference JSON from style sidecar model.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    parser.add_argument("--report-output", default="", help="Optional report JSON path.")

    parser.add_argument("--max-prob", type=float, default=0.72, help="Min max_prob gate.")
    parser.add_argument("--margin", type=float, default=0.16, help="Min margin gate.")
    parser.add_argument("--min-delta", type=int, default=1, help="Min abs(style_sidecar - baseline_style).")
    parser.add_argument("--max-replace-ratio", type=float, default=0.12, help="Maximum replacement ratio.")
    parser.add_argument(
        "--allow-upward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow increasing style score (default True).",
    )
    parser.add_argument(
        "--distribution-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback replacements if style distribution drift vs gold worsens too much.",
    )
    parser.add_argument(
        "--max-l1-increase",
        type=float,
        default=0.05,
        help="Allowed increase for L1(pred_dist, gold_dist) under distribution guard.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"JSON must be a list: {p}")
    return rows


def get_int(row: dict[str, Any], keys: tuple[str, ...], default: int | None = None) -> int | None:
    for key in keys:
        if key in row:
            try:
                return int(row[key])
            except Exception:
                continue
    return default


def distribution_l1(pred: list[int], gold: list[int], labels: tuple[int, ...] = (0, 1, 2, 3)) -> float:
    if not pred or not gold or len(pred) != len(gold):
        return 0.0
    n = len(pred)
    l1 = 0.0
    for lb in labels:
        p = sum(1 for x in pred if x == lb) / n
        g = sum(1 for x in gold if x == lb) / n
        l1 += abs(p - g)
    return float(l1)


def main() -> None:
    args = parse_args()
    base_rows = load_rows(args.base_input)
    model_rows = load_rows(args.model_input)

    if not base_rows:
        raise ValueError("base-input rows are empty.")
    if len(base_rows) != len(model_rows):
        raise ValueError("Input row counts mismatch between base-input and model-input.")

    model_map = {get_int(r, ("essay_id",), None): r for r in model_rows}
    candidates: list[tuple[int, float]] = []
    baseline_style: list[int] = []
    sidecar_style: list[int] = []

    for idx, row in enumerate(base_rows):
        essay_id = get_int(row, ("essay_id",), None)
        base = get_int(row, ("style", "pred_style"), default=None)
        if base is None:
            baseline_style.append(0)
            sidecar_style.append(0)
            continue
        base = int(base)
        baseline_style.append(base)

        if essay_id is None or essay_id not in model_map:
            sidecar_style.append(base)
            continue
        m = model_map[essay_id]
        pred = get_int(m, ("bert_style_pred",), default=None)
        prob = m.get("bert_style_max_prob")
        margin = m.get("bert_style_margin")
        if pred is None:
            sidecar_style.append(base)
            continue
        pred = int(pred)
        sidecar_style.append(pred)

        try:
            prob_f = float(prob)
            margin_f = float(margin)
        except Exception:
            continue

        if prob_f < float(args.max_prob):
            continue
        if margin_f < float(args.margin):
            continue
        if abs(pred - base) < int(args.min_delta):
            continue
        if (not args.allow_upward) and (pred > base):
            continue

        score = prob_f * margin_f
        candidates.append((idx, score))

    cap = max(0, int(len(base_rows) * float(args.max_replace_ratio)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = set(idx for idx, _ in candidates[:cap])

    final_style: list[int] = []
    used_mask: list[bool] = []
    for i, base in enumerate(baseline_style):
        if i in selected:
            final_style.append(int(sidecar_style[i]))
            used_mask.append(True)
        else:
            final_style.append(int(base))
            used_mask.append(False)

    report: dict[str, Any] = {
        "rows": len(base_rows),
        "gate": {
            "max_prob": float(args.max_prob),
            "margin": float(args.margin),
            "min_delta": int(args.min_delta),
            "max_replace_ratio": float(args.max_replace_ratio),
            "allow_upward": bool(args.allow_upward),
        },
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
    }

    if args.distribution_guard and all("gold_style" in r for r in base_rows):
        gold = [int(r["gold_style"]) for r in base_rows]
        l1_base = distribution_l1(baseline_style, gold)
        l1_final = distribution_l1(final_style, gold)
        report["l1_baseline_vs_gold"] = l1_base
        report["l1_final_vs_gold"] = l1_final
        if l1_final > (l1_base + float(args.max_l1_increase)):
            final_style = baseline_style[:]
            used_mask = [False] * len(base_rows)
            report["rolled_back_by_distribution_guard"] = True

    merged_rows: list[dict[str, Any]] = []
    used_count = 0
    for idx, row in enumerate(base_rows):
        out = dict(row)
        ideas = get_int(out, ("ideas", "pred_ideas"), default=0) or 0
        org = get_int(out, ("organization", "pred_organization"), default=0) or 0
        conv = get_int(out, ("conventions", "pred_conventions"), default=0) or 0
        base_style_score = baseline_style[idx]
        style = int(final_style[idx])
        overall = int(Prompt7Mapper.compute_overall_score(int(ideas), int(org), style, int(conv)))

        used = bool(used_mask[idx])
        used_count += int(used)
        out["baseline_style"] = int(base_style_score)
        out["sidecar_used_style"] = used
        out["style"] = style
        out["overall"] = overall
        merged_rows.append(out)

    report["final_used_count"] = int(used_count)
    report["final_used_ratio"] = float(used_count / max(1, len(base_rows)))
    report["final_prediction_distribution"] = {
        "style": {str(k): int(v) for k, v in sorted(Counter(r["style"] for r in merged_rows).items())}
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(merged_rows)}")
    print(f"style_replaced={used_count} ({used_count/len(base_rows):.2%})")
    print(f"saved: {out_path}")

    if args.report_output.strip():
        rp = Path(args.report_output)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_saved: {rp}")


if __name__ == "__main__":
    main()
