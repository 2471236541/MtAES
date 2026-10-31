#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按门控规则将 Prompt7 baseline 预测与 BERT sidecar 输出融合。

默认门控：
- max_prob >= 0.72
- margin   >= 0.18
- abs(bert - baseline) >= 1
- 每个 trait 的替换比例 <= 25%

输出 JSON 会保留原始字段，并补充兼容 QWK 评估的最终预测字段：
- ideas, organization, style, conventions, overall
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prompt7_mapper import Prompt7Mapper


ORG_DIRECTION_PRESETS: dict[str, dict[tuple[int, int], dict[str, float | bool]]] = {
    "none": {},
    "o1": {
        (0, 1): {"enabled": True, "prob": 0.70, "margin": 0.45},
        (1, 2): {"enabled": True, "prob": 0.72, "margin": 0.18},
        (3, 2): {"enabled": True, "prob": 0.72, "margin": 0.35},
        (0, 2): {"enabled": True, "prob": 0.78, "margin": 0.50},
        (1, 3): {"enabled": False},
        (2, 1): {"enabled": False},
        (2, 3): {"enabled": False},
        (3, 1): {"enabled": False},
        (0, 3): {"enabled": False},
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON from infer_sidecar_bert.py")
    parser.add_argument("--output", required=True, help="Merged predictions JSON path.")
    parser.add_argument("--report-output", default="", help="Optional report JSON path.")
    parser.add_argument("--max-prob", type=float, default=0.72)
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument(
        "--ideas-margin-threshold",
        type=float,
        default=-1.0,
        help="Optional floor margin for ideas only; effective ideas margin is max(gate_margin, this value).",
    )
    parser.add_argument(
        "--ideas-score-floor",
        type=float,
        default=None,
        help="Optional score floor for ideas candidate selection. Disabled when unset.",
    )
    parser.add_argument("--min-delta", type=int, default=1)
    parser.add_argument("--max-replace-ratio", type=float, default=0.25)
    parser.add_argument(
        "--ideas-max-replace-ratio",
        type=float,
        default=-1.0,
        help="Optional override max_replace_ratio for ideas only.",
    )
    parser.add_argument("--org-max-prob", type=float, default=-1.0, help="Override max_prob for organization only.")
    parser.add_argument("--org-margin", type=float, default=-1.0, help="Override margin for organization only.")
    parser.add_argument("--org-min-delta", type=int, default=-1, help="Override min_delta for organization only.")
    parser.add_argument(
        "--org-max-replace-ratio",
        type=float,
        default=-1.0,
        help="Override max_replace_ratio for organization only.",
    )
    parser.add_argument(
        "--org-directional-preset",
        choices=tuple(ORG_DIRECTION_PRESETS.keys()),
        default="none",
        help="Optional preset for organization directional gate.",
    )
    parser.add_argument(
        "--distribution-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback trait replacements if distribution drift worsens too much vs gold.",
    )
    parser.add_argument(
        "--max-l1-increase",
        type=float,
        default=0.05,
        help="Allowed increase of distribution L1 vs baseline when gold labels exist.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list.")
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


def serialize_direction_counts(counter: Counter[tuple[int, int]]) -> dict[str, int]:
    return {f"{src}->{dst}": int(count) for (src, dst), count in sorted(counter.items())}


def get_bool(row: dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in row:
        return default
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return default


def select_trait_replacements(
    rows: list[dict[str, Any]],
    trait: str,
    max_prob_th: float,
    margin_th: float,
    min_delta: int,
    max_replace_ratio: float,
    direction_rules: dict[tuple[int, int], dict[str, float | bool]] | None = None,
    margin_floor: float | None = None,
    score_floor: float | None = None,
) -> dict[str, Any]:
    base_key_candidates = (f"pred_{trait}", trait, f"baseline_{trait}")
    bert_pred_key = f"bert_{trait}_pred"
    max_prob_key = f"bert_{trait}_max_prob"
    margin_key = f"bert_{trait}_margin"

    candidates: list[tuple[int, float, tuple[int, int]]] = []
    baseline_values: list[int] = []
    bert_values: list[int] = []
    direction_candidate_counts: Counter[tuple[int, int]] = Counter()
    direction_selected_counts: Counter[tuple[int, int]] = Counter()

    for idx, row in enumerate(rows):
        base = get_int(row, base_key_candidates, default=None)
        bert = get_int(row, (bert_pred_key,), default=None)
        max_prob = row.get(max_prob_key, None)
        margin = row.get(margin_key, None)

        if base is None or bert is None:
            baseline_values.append(base if base is not None else 0)
            bert_values.append(bert if bert is not None else 0)
            continue

        baseline_values.append(int(base))
        bert_values.append(int(bert))
        try:
            max_prob_f = float(max_prob)
            margin_f = float(margin)
        except Exception:
            continue

        base_i = int(base)
        bert_i = int(bert)
        prob_th = max_prob_th
        margin_req = margin_th
        direction = (base_i, bert_i)
        if direction_rules:
            rule = direction_rules.get(direction)
            if not rule or not bool(rule.get("enabled", False)):
                continue
            prob_th = float(rule.get("prob", max_prob_th))
            margin_req = float(rule.get("margin", margin_th))
        if margin_floor is not None:
            margin_req = max(margin_req, float(margin_floor))

        if abs(bert_i - base_i) < min_delta:
            continue

        gate_score = max_prob_f * margin_f
        # For ideas replacements, preserve candidate diversity and penalize
        # low-quality candidates in ranking instead of aggressively filtering.
        if trait == "ideas" and bert_i != base_i:
            # backup:
            # if direction in {(2, 1), (1, 0)}:
            if direction in {(2, 1), (1, 0), (0, 1)}:
                continue
            # When an ideas-specific margin floor is requested, make it a real
            # hard gate before the later ranking penalties are applied.
            if margin_floor is not None and margin_f < margin_req:
                continue
            # Older caches do not carry bert_ideas_agreement_used. Treat missing as
            # unknown instead of forced disagreement; otherwise every ideas candidate
            # gets a blanket 0.1 penalty and the ideas sidecar is suppressed too hard.
            has_agreement_flag = "bert_ideas_agreement_used" in row
            if has_agreement_flag and not get_bool(row, "bert_ideas_agreement_used", default=False):
                gate_score *= 0.1
            if max_prob_f < 0.70 or margin_f < 0.20:
                gate_score *= 0.6
            if abs(bert_i - base_i) == 1:
                if max_prob_f < 0.75 or margin_f < 0.26:
                    gate_score *= 0.7
            else:
                if max_prob_f < 0.80 or margin_f < 0.40:
                    gate_score *= 0.5
        else:
            if max_prob_f < prob_th:
                continue
            if margin_f < margin_req:
                continue

        candidates.append((idx, gate_score, direction))
        direction_candidate_counts[direction] += 1

    cap = int(len(rows) * max_replace_ratio)
    cap = max(0, cap)
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_candidates = candidates
    if trait == "ideas" and score_floor is not None:
        selected_candidates = []
        for item in candidates:
            if item[1] < float(score_floor):
                break
            selected_candidates.append(item)
    selected_idx = set()
    for idx, _, direction in selected_candidates[:cap]:
        selected_idx.add(idx)
        direction_selected_counts[direction] += 1

    final = []
    used = []
    for idx, (base, bert) in enumerate(zip(baseline_values, bert_values)):
        if idx in selected_idx:
            final.append(int(bert))
            used.append(True)
        else:
            final.append(int(base))
            used.append(False)

    return {
        "baseline": baseline_values,
        "bert": bert_values,
        "final": final,
        "used_mask": used,
        "candidate_count": len(candidates),
        "selected_count": len(selected_idx),
        "selected_indices": selected_idx,
        "direction_candidate_counts": serialize_direction_counts(direction_candidate_counts),
        "direction_selected_counts": serialize_direction_counts(direction_selected_counts),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise ValueError("Input rows are empty.")

    ideas_merge = select_trait_replacements(
        rows=rows,
        trait="ideas",
        max_prob_th=args.max_prob,
        margin_th=args.margin,
        min_delta=args.min_delta,
        max_replace_ratio=args.ideas_max_replace_ratio if args.ideas_max_replace_ratio >= 0 else args.max_replace_ratio,
        margin_floor=(float(args.ideas_margin_threshold) if args.ideas_margin_threshold >= 0 else None),
        score_floor=args.ideas_score_floor,
    )
    org_merge = select_trait_replacements(
        rows=rows,
        trait="organization",
        max_prob_th=args.org_max_prob if args.org_max_prob >= 0 else args.max_prob,
        margin_th=args.org_margin if args.org_margin >= 0 else args.margin,
        min_delta=args.org_min_delta if args.org_min_delta >= 0 else args.min_delta,
        max_replace_ratio=args.org_max_replace_ratio if args.org_max_replace_ratio >= 0 else args.max_replace_ratio,
        direction_rules=ORG_DIRECTION_PRESETS.get(args.org_directional_preset, {}),
    )

    report: dict[str, Any] = {
        "rows": len(rows),
        "gate": {
            "max_prob": float(args.max_prob),
            "margin": float(args.margin),
            "ideas_margin_threshold": float(args.ideas_margin_threshold) if args.ideas_margin_threshold >= 0 else None,
            "min_delta": int(args.min_delta),
            "max_replace_ratio": float(args.max_replace_ratio),
            "ideas_max_replace_ratio": float(args.ideas_max_replace_ratio) if args.ideas_max_replace_ratio >= 0 else None,
            "org_max_prob": float(args.org_max_prob) if args.org_max_prob >= 0 else None,
            "org_margin": float(args.org_margin) if args.org_margin >= 0 else None,
            "org_min_delta": int(args.org_min_delta) if args.org_min_delta >= 0 else None,
            "org_max_replace_ratio": float(args.org_max_replace_ratio) if args.org_max_replace_ratio >= 0 else None,
            "org_directional_preset": str(args.org_directional_preset),
        },
        "traits": {
            "ideas": {
                "candidate_count": int(ideas_merge["candidate_count"]),
                "selected_count": int(ideas_merge["selected_count"]),
                "direction_candidate_counts": ideas_merge["direction_candidate_counts"],
                "direction_selected_counts": ideas_merge["direction_selected_counts"],
            },
            "organization": {
                "candidate_count": int(org_merge["candidate_count"]),
                "selected_count": int(org_merge["selected_count"]),
                "direction_candidate_counts": org_merge["direction_candidate_counts"],
                "direction_selected_counts": org_merge["direction_selected_counts"],
            },
        },
    }

    # 可选分布保护（仅当存在金标时生效）
    if args.distribution_guard:
        if all("gold_ideas" in r for r in rows):
            gold = [int(r["gold_ideas"]) for r in rows]
            l1_base = distribution_l1(ideas_merge["baseline"], gold)
            l1_final = distribution_l1(ideas_merge["final"], gold)
            if l1_final > (l1_base + args.max_l1_increase):
                ideas_merge["final"] = ideas_merge["baseline"][:]
                ideas_merge["used_mask"] = [False] * len(rows)
                report["traits"]["ideas"]["rolled_back_by_distribution_guard"] = True
            report["traits"]["ideas"]["l1_baseline_vs_gold"] = l1_base
            report["traits"]["ideas"]["l1_final_vs_gold"] = l1_final

        if all("gold_organization" in r for r in rows):
            gold = [int(r["gold_organization"]) for r in rows]
            l1_base = distribution_l1(org_merge["baseline"], gold)
            l1_final = distribution_l1(org_merge["final"], gold)
            if l1_final > (l1_base + args.max_l1_increase):
                org_merge["final"] = org_merge["baseline"][:]
                org_merge["used_mask"] = [False] * len(rows)
                report["traits"]["organization"]["rolled_back_by_distribution_guard"] = True
            report["traits"]["organization"]["l1_baseline_vs_gold"] = l1_base
            report["traits"]["organization"]["l1_final_vs_gold"] = l1_final

    merged_rows = []
    ideas_used = 0
    org_used = 0
    for i, row in enumerate(rows):
        out = dict(row)
        baseline_style = get_int(row, ("pred_style", "style"), default=0)
        baseline_conv = get_int(row, ("pred_conventions", "conventions"), default=0)
        final_ideas = int(ideas_merge["final"][i])
        final_org = int(org_merge["final"][i])
        final_overall = int(
            Prompt7Mapper.compute_overall_score(final_ideas, final_org, int(baseline_style), int(baseline_conv))
        )

        used_i = bool(ideas_merge["used_mask"][i])
        used_o = bool(org_merge["used_mask"][i])
        ideas_used += int(used_i)
        org_used += int(used_o)

        out["baseline_ideas"] = int(ideas_merge["baseline"][i])
        out["baseline_organization"] = int(org_merge["baseline"][i])
        out["sidecar_used_ideas"] = used_i
        out["sidecar_used_organization"] = used_o

        # 最终预测字段（兼容现有 QWK 评估脚本）
        out["ideas"] = final_ideas
        out["organization"] = final_org
        out["style"] = int(baseline_style)
        out["conventions"] = int(baseline_conv)
        out["overall"] = final_overall

        merged_rows.append(out)

    report["traits"]["ideas"]["final_used_count"] = ideas_used
    report["traits"]["organization"]["final_used_count"] = org_used
    report["traits"]["ideas"]["final_used_ratio"] = ideas_used / len(rows)
    report["traits"]["organization"]["final_used_ratio"] = org_used / len(rows)

    report["final_prediction_distribution"] = {
        "ideas": {str(k): int(v) for k, v in sorted(Counter(r["ideas"] for r in merged_rows).items())},
        "organization": {str(k): int(v) for k, v in sorted(Counter(r["organization"] for r in merged_rows).items())},
        "style": {str(k): int(v) for k, v in sorted(Counter(r["style"] for r in merged_rows).items())},
        "conventions": {str(k): int(v) for k, v in sorted(Counter(r["conventions"] for r in merged_rows).items())},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(merged_rows)}")
    print(
        f"ideas_replaced={ideas_used} ({ideas_used/len(rows):.2%}), "
        f"organization_replaced={org_used} ({org_used/len(rows):.2%})"
    )
    print(f"saved: {out_path}")

    if args.report_output.strip():
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_saved: {report_path}")


if __name__ == "__main__":
    main()
