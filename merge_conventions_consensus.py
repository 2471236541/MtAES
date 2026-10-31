#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conventions 共识门控融合（双模型一致性）。

用途：
- 固定 ideas/organization/style。
- 仅当两个 conventions sidecar 模型预测一致且通过严格置信门控时才替换。
- 可选只做单向下调修正（稳定性更高）。
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
    parser.add_argument("--model-a-input", required=True, help="Inference JSON from conventions model A.")
    parser.add_argument("--model-b-input", required=True, help="Inference JSON from conventions model B.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    parser.add_argument("--report-output", default="", help="Optional report JSON path.")

    parser.add_argument("--max-prob", type=float, default=0.82, help="Min averaged max_prob gate.")
    parser.add_argument("--margin", type=float, default=0.22, help="Min averaged margin gate.")
    parser.add_argument("--min-delta", type=int, default=1, help="Min abs(consensus - baseline) to consider replacement.")
    parser.add_argument("--max-replace-ratio", type=float, default=0.08, help="Maximum replacement ratio.")
    parser.add_argument(
        "--allow-upward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow increasing conventions score (default False: only downward correction).",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("none", "23_down", "23_bi"),
        default="none",
        help=(
            "Boundary constraints for replacements: "
            "none=any delta>=min-delta; "
            "23_down=only 3->2; "
            "23_bi=only between 2 and 3 (2<->3)."
        ),
    )
    parser.add_argument(
        "--distribution-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback replacements if conventions distribution drift vs gold worsens too much.",
    )
    parser.add_argument(
        "--max-l1-increase",
        type=float,
        default=0.03,
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
    a_rows = load_rows(args.model_a_input)
    b_rows = load_rows(args.model_b_input)

    if not base_rows:
        raise ValueError("base-input rows are empty.")
    if len(base_rows) != len(a_rows) or len(base_rows) != len(b_rows):
        raise ValueError("Input row counts mismatch among base/model-a/model-b.")

    a_map = {get_int(r, ("essay_id",), None): r for r in a_rows}
    b_map = {get_int(r, ("essay_id",), None): r for r in b_rows}

    candidates: list[tuple[int, float]] = []
    baseline_conv: list[int] = []
    consensus_conv: list[int] = []

    for idx, row in enumerate(base_rows):
        essay_id = get_int(row, ("essay_id",), None)
        if essay_id is None or essay_id not in a_map or essay_id not in b_map:
            baseline_conv.append(get_int(row, ("conventions", "pred_conventions"), default=0) or 0)
            consensus_conv.append(get_int(row, ("conventions", "pred_conventions"), default=0) or 0)
            continue

        base = get_int(row, ("conventions", "pred_conventions"), default=None)
        if base is None:
            baseline_conv.append(0)
            consensus_conv.append(0)
            continue
        base = int(base)

        a = a_map[essay_id]
        b = b_map[essay_id]
        a_pred = get_int(a, ("bert_conventions_pred",), default=None)
        b_pred = get_int(b, ("bert_conventions_pred",), default=None)
        a_prob = a.get("bert_conventions_max_prob")
        b_prob = b.get("bert_conventions_max_prob")
        a_margin = a.get("bert_conventions_margin")
        b_margin = b.get("bert_conventions_margin")

        baseline_conv.append(base)
        if a_pred is None or b_pred is None:
            consensus_conv.append(base)
            continue

        a_pred = int(a_pred)
        b_pred = int(b_pred)
        if a_pred != b_pred:
            consensus_conv.append(base)
            continue
        consensus = a_pred
        consensus_conv.append(consensus)

        try:
            avg_prob = (float(a_prob) + float(b_prob)) / 2.0
            avg_margin = (float(a_margin) + float(b_margin)) / 2.0
        except Exception:
            continue

        if avg_prob < float(args.max_prob):
            continue
        if avg_margin < float(args.margin):
            continue
        if abs(consensus - base) < int(args.min_delta):
            continue
        if (not args.allow_upward) and (consensus > base):
            continue
        if args.boundary_mode == "23_down":
            if not (base == 3 and consensus == 2):
                continue
        elif args.boundary_mode == "23_bi":
            if not ({base, consensus} == {2, 3}):
                continue

        score = avg_prob * avg_margin
        candidates.append((idx, score))

    cap = max(0, int(len(base_rows) * float(args.max_replace_ratio)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = set(idx for idx, _ in candidates[:cap])

    final_conv: list[int] = []
    used_mask: list[bool] = []
    for i, base in enumerate(baseline_conv):
        if i in selected:
            final_conv.append(int(consensus_conv[i]))
            used_mask.append(True)
        else:
            final_conv.append(int(base))
            used_mask.append(False)

    report: dict[str, Any] = {
        "rows": len(base_rows),
        "gate": {
            "max_prob": float(args.max_prob),
            "margin": float(args.margin),
            "min_delta": int(args.min_delta),
            "max_replace_ratio": float(args.max_replace_ratio),
            "allow_upward": bool(args.allow_upward),
            "boundary_mode": str(args.boundary_mode),
        },
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
    }

    # 分布保护（有金标时生效）
    if args.distribution_guard and all("gold_conventions" in r for r in base_rows):
        gold = [int(r["gold_conventions"]) for r in base_rows]
        l1_base = distribution_l1(baseline_conv, gold)
        l1_final = distribution_l1(final_conv, gold)
        report["l1_baseline_vs_gold"] = l1_base
        report["l1_final_vs_gold"] = l1_final
        if l1_final > (l1_base + float(args.max_l1_increase)):
            final_conv = baseline_conv[:]
            used_mask = [False] * len(base_rows)
            report["rolled_back_by_distribution_guard"] = True

    merged_rows: list[dict[str, Any]] = []
    used_count = 0
    for idx, row in enumerate(base_rows):
        out = dict(row)
        ideas = get_int(out, ("ideas", "pred_ideas"), default=0) or 0
        org = get_int(out, ("organization", "pred_organization"), default=0) or 0
        style = get_int(out, ("style", "pred_style"), default=0) or 0
        base_conv_score = baseline_conv[idx]
        conv = int(final_conv[idx])
        overall = int(Prompt7Mapper.compute_overall_score(int(ideas), int(org), int(style), conv))

        used = bool(used_mask[idx])
        used_count += int(used)
        out["baseline_conventions"] = int(base_conv_score)
        out["sidecar_used_conventions"] = used
        out["conventions"] = conv
        out["overall"] = overall
        merged_rows.append(out)

    report["final_used_count"] = int(used_count)
    report["final_used_ratio"] = float(used_count / max(1, len(base_rows)))
    report["final_prediction_distribution"] = {
        "conventions": {str(k): int(v) for k, v in sorted(Counter(r["conventions"] for r in merged_rows).items())}
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(merged_rows)}")
    print(f"conventions_replaced={used_count} ({used_count/len(base_rows):.2%})")
    print(f"saved: {out_path}")

    if args.report_output.strip():
        rp = Path(args.report_output)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_saved: {rp}")


if __name__ == "__main__":
    main()
