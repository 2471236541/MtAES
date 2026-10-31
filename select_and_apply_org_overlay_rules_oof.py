#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from prompt7_mapper import Prompt7Mapper


TRAITS = ("ideas", "organization", "style", "conventions")


@dataclass(frozen=True)
class Rule:
    direction: str
    prob_ge: float
    margin_ge: float
    word_filter: str
    base_org_filter: str
    donor_org_filter: str

    @property
    def tag(self) -> str:
        p = int(round(self.prob_ge * 100))
        m = int(round(self.margin_ge * 100))
        raw = f"{self.direction}_{self.word_filter}_p{p:02d}_m{m:02d}_{self.base_org_filter}_{self.donor_org_filter}"
        return re.sub(r"[^A-Za-z0-9_]+", "_", raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select organization overlay rules on OOF train predictions, then apply top "
            "rules to dev. No holdout or outer_test is used."
        )
    )
    p.add_argument("--oof-pair-root", required=True)
    p.add_argument("--dev-base-root", required=True)
    p.add_argument("--dev-donor-root", required=True)
    p.add_argument("--dev-reference-root", required=True, help="Reference dev root, usually current e48 best.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--outer-folds", default="outer0,outer1,outer2,outer3,outer4")
    p.add_argument("--inner-splits", default="s0,s1,s2")
    p.add_argument("--seeds", required=True)
    p.add_argument("--prefix", default="v10ncv")
    p.add_argument("--split-kind", default="dev")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-oof-changed-ratio", type=float, default=0.02)
    p.add_argument("--bucket-word-threshold", type=int, default=200)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return int(default)


def get_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def split_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def seed_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def stem(prefix: str, outer: str, split: str, seed: int, split_kind: str) -> str:
    return f"{prefix}_{outer}_{split}_seed{seed}_{split_kind}"


def word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("essay_text", "")).split())


def recompute_overall(row: dict[str, Any]) -> None:
    row["overall"] = int(
        Prompt7Mapper.compute_overall_score(
            get_int(row, "ideas"),
            get_int(row, "organization"),
            get_int(row, "style"),
            get_int(row, "conventions"),
        )
    )
    row["pred_overall"] = int(row["overall"])


def qwk(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    if min_rating == max_rating:
        return 1.0
    n = max_rating - min_rating + 1
    total = len(y_true)
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    true_hist = [0.0 for _ in range(n)]
    pred_hist = [0.0 for _ in range(n)]
    for true, pred in zip(y_true, y_pred):
        observed[true - min_rating][pred - min_rating] += 1.0
        true_hist[true - min_rating] += 1.0
        pred_hist[pred - min_rating] += 1.0
    denom_scale = float((n - 1) ** 2) if n > 1 else 1.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / denom_scale
            expected = true_hist[i] * pred_hist[j] / total
            numerator += weight * observed[i][j]
            denominator += weight * expected
    return float(1.0 - numerator / denominator) if denominator else 0.0


def metric_qwk(rows: list[dict[str, Any]], metric: str) -> float:
    gold_key = "gold_overall" if metric == "overall" else f"gold_{metric}"
    pred_key = "overall" if metric == "overall" else metric
    return qwk([get_int(row, gold_key) for row in rows], [get_int(row, pred_key) for row in rows])


def qwk_payload(rows: list[dict[str, Any]], pred_path: Path) -> dict[str, Any]:
    return {
        "num_samples": len(rows),
        "skipped_rows": 0,
        "trait_qwk": {trait: metric_qwk(rows, trait) for trait in TRAITS},
        "overall_qwk": metric_qwk(rows, "overall"),
        "pred_path": str(pred_path),
        "gold_path": "",
    }


def direction_ok(direction: str, base_value: int, donor_value: int) -> bool:
    if direction == "changed":
        return donor_value != base_value
    if direction == "up":
        return donor_value > base_value
    if direction == "down":
        return donor_value < base_value
    if direction == "no_down":
        return donor_value >= base_value and donor_value != base_value
    if direction == "no_up":
        return donor_value <= base_value and donor_value != base_value
    raise ValueError(f"Unsupported direction: {direction}")


def word_ok(word_filter: str, row: dict[str, Any]) -> bool:
    wc = word_count(row)
    if word_filter == "all":
        return True
    if word_filter.startswith("le"):
        return wc <= int(word_filter[2:])
    if word_filter.startswith("gt"):
        return wc > int(word_filter[2:])
    raise ValueError(f"Unsupported word filter: {word_filter}")


def org_filter_ok(filter_name: str, value: int) -> bool:
    if filter_name == "any":
        return True
    if filter_name == "le1":
        return value <= 1
    if filter_name == "le2":
        return value <= 2
    if filter_name == "ge2":
        return value >= 2
    if filter_name == "ge3":
        return value >= 3
    if filter_name.startswith("eq"):
        return value == int(filter_name[2:])
    raise ValueError(f"Unsupported org filter: {filter_name}")


def donor_org(row: dict[str, Any]) -> int:
    if "bert_organization_pred" in row:
        return get_int(row, "bert_organization_pred")
    return get_int(row, "organization")


def rule_ok(rule: Rule, base: dict[str, Any], donor: dict[str, Any]) -> bool:
    base_org = get_int(base, "organization")
    d_org = donor_org(donor)
    return (
        direction_ok(rule.direction, base_org, d_org)
        and word_ok(rule.word_filter, base)
        and org_filter_ok(rule.base_org_filter, base_org)
        and org_filter_ok(rule.donor_org_filter, d_org)
        and get_float(donor, "bert_organization_max_prob") >= rule.prob_ge
        and get_float(donor, "bert_organization_margin") >= rule.margin_ge
    )


def apply_rule(base_rows: list[dict[str, Any]], donor_rows: list[dict[str, Any]], rule: Rule) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donor_by_id = {get_int(row, "essay_id"): row for row in donor_rows}
    out_rows: list[dict[str, Any]] = []
    counts = {
        "rows": 0,
        "eligible_rows": 0,
        "changed_rows": 0,
        "up_rows": 0,
        "down_rows": 0,
        "base_kept_rows": 0,
    }
    for base in base_rows:
        essay_id = get_int(base, "essay_id")
        if essay_id not in donor_by_id:
            raise ValueError(f"Missing donor essay_id={essay_id}")
        donor = donor_by_id[essay_id]
        out = dict(base)
        counts["rows"] += 1
        base_org = get_int(base, "organization")
        d_org = donor_org(donor)
        if rule_ok(rule, base, donor):
            counts["eligible_rows"] += 1
            for key in (
                "bert_organization_pred",
                "bert_organization_probs",
                "bert_organization_max_prob",
                "bert_organization_margin",
            ):
                if key in donor:
                    out[key] = donor[key]
            out["organization"] = int(d_org)
            out["pred_organization"] = int(d_org)
            out["sidecar_used_organization"] = bool(d_org != get_int(out, "baseline_organization", get_int(out, "pred_organization")))
            out["overlay_applied"] = True
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = [
                f"direction={rule.direction}",
                f"word={rule.word_filter}",
                f"donor_organization_max_prob>={rule.prob_ge:.2f}",
                f"donor_organization_margin>={rule.margin_ge:.2f}",
                f"base_org={rule.base_org_filter}",
                f"donor_org={rule.donor_org_filter}",
            ]
            recompute_overall(out)
            if d_org != base_org:
                counts["changed_rows"] += 1
                if d_org > base_org:
                    counts["up_rows"] += 1
                elif d_org < base_org:
                    counts["down_rows"] += 1
        else:
            counts["base_kept_rows"] += 1
            out["overlay_applied"] = False
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = []
        out_rows.append(out)
    return out_rows, counts


def generate_rules() -> list[Rule]:
    # Keep this grid intentionally small. OOF is for selecting a robust rule,
    # not for exhaustive threshold mining.
    directions = ["changed", "up", "down"]
    probs = [0.50, 0.55, 0.60, 0.65]
    margins = [0.25, 0.30, 0.35, 0.40]
    word_filters = ["all", "le240", "gt200"]
    base_filters = ["any", "le2", "ge2"]
    donor_filters = ["any", "ge3"]
    rules: list[Rule] = []
    for direction in directions:
        for prob in probs:
            for margin in margins:
                for word_filter in word_filters:
                    for base_filter in base_filters:
                        for donor_filter in donor_filters:
                            rules.append(
                                Rule(
                                    direction=direction,
                                    prob_ge=prob,
                                    margin_ge=margin,
                                    word_filter=word_filter,
                                    base_org_filter=base_filter,
                                    donor_org_filter=donor_filter,
                                )
                            )
    return rules


def load_oof_pairs(root: Path, outers: list[str], splits: list[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for outer in outers:
        for split in splits:
            pair_dir = root / outer / split / "oof_train"
            base_path = pair_dir / "base_selected_preds.json"
            donor_path = pair_dir / "candidate_selected_preds.json"
            if not base_path.exists():
                raise FileNotFoundError(f"Missing OOF base: {base_path}")
            if not donor_path.exists():
                raise FileNotFoundError(f"Missing OOF donor: {donor_path}")
            pairs.append(
                {
                    "outer": outer,
                    "split": split,
                    "base_rows": load_json(base_path),
                    "donor_rows": load_json(donor_path),
                }
            )
    return pairs


def load_dev_cells(args: argparse.Namespace, rule: Rule) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    base_root = Path(args.dev_base_root)
    donor_root = Path(args.dev_donor_root)
    ref_root = Path(args.dev_reference_root)
    for outer in split_csv(args.outer_folds):
        for split in split_csv(args.inner_splits):
            for seed in seed_csv(args.seeds):
                s = stem(str(args.prefix), outer, split, seed, str(args.split_kind))
                base_path = base_root / outer / split / f"seed{seed}" / f"{s}_selected_preds.json"
                donor_path = donor_root / outer / split / f"seed{seed}" / f"{s}_org_highscore_with_bert.json"
                ref_path = ref_root / outer / split / f"seed{seed}" / f"{s}_selected_preds.json"
                if not base_path.exists():
                    raise FileNotFoundError(f"Missing dev base: {base_path}")
                if not donor_path.exists():
                    raise FileNotFoundError(f"Missing dev donor: {donor_path}")
                if not ref_path.exists():
                    raise FileNotFoundError(f"Missing dev reference: {ref_path}")
                base_rows = load_json(base_path)
                donor_rows = load_json(donor_path)
                out_rows, counts = apply_rule(base_rows, donor_rows, rule)
                cells.append(
                    {
                        "outer": outer,
                        "split": split,
                        "seed": seed,
                        "stem": s,
                        "rows": out_rows,
                        "reference_rows": load_json(ref_path),
                        "counts": counts,
                    }
                )
    return cells


def summarize_oof_rule(rule: Rule, pairs: list[dict[str, Any]], min_changed_ratio: float) -> dict[str, Any] | None:
    overall_proxy_deltas: list[float] = []
    org_accuracy_deltas: list[float] = []
    changed = 0
    rows = 0
    for pair in pairs:
        base_rows = pair["base_rows"]
        donor_rows = pair["donor_rows"]
        donor_by_id = {get_int(row, "essay_id"): row for row in donor_rows}
        benefit_sum = 0.0
        org_delta_sum = 0.0
        pair_rows = 0
        pair_changed = 0
        for base in base_rows:
            pair_rows += 1
            essay_id = get_int(base, "essay_id")
            donor = donor_by_id[essay_id]
            old_overall = get_int(base, "overall")
            old_org = get_int(base, "organization")
            new_org = old_org
            if rule_ok(rule, base, donor):
                d_org = donor_org(donor)
                if d_org != old_org:
                    new_org = d_org
                    pair_changed += 1
            if new_org != old_org:
                new_overall = int(
                    Prompt7Mapper.compute_overall_score(
                        get_int(base, "ideas"),
                        new_org,
                        get_int(base, "style"),
                        get_int(base, "conventions"),
                    )
                )
            else:
                new_overall = old_overall
            gold_overall = get_int(base, "gold_overall")
            old_err = old_overall - gold_overall
            new_err = new_overall - gold_overall
            benefit_sum += float((old_err * old_err) - (new_err * new_err))
            gold_org = get_int(base, "gold_organization")
            org_delta_sum += float(int(new_org == gold_org) - int(old_org == gold_org))
        overall_proxy_deltas.append(benefit_sum / pair_rows if pair_rows else 0.0)
        org_accuracy_deltas.append(org_delta_sum / pair_rows if pair_rows else 0.0)
        changed += int(pair_changed)
        rows += int(pair_rows)
    changed_ratio = changed / rows if rows else 0.0
    if changed_ratio < float(min_changed_ratio):
        return None
    return {
        "rule": asdict(rule),
        "tag": rule.tag,
        "oof_overall_delta_mean": mean(overall_proxy_deltas),
        "oof_overall_positive_sets": sum(1 for x in overall_proxy_deltas if x > 0.0),
        "oof_organization_delta_mean": mean(org_accuracy_deltas),
        "oof_organization_positive_sets": sum(1 for x in org_accuracy_deltas if x > 0.0),
        "oof_metric": "mean_squared_overall_error_reduction_proxy",
        "oof_org_metric": "organization_accuracy_delta_proxy",
        "oof_sets": len(overall_proxy_deltas),
        "oof_changed_rows": changed,
        "oof_rows": rows,
        "oof_changed_ratio": changed_ratio,
    }


def compare_rows(candidate: list[dict[str, Any]], reference: list[dict[str, Any]], bucket_threshold: int) -> dict[str, Any]:
    ref_by_id = {get_int(row, "essay_id"): row for row in reference}
    cand_by_id = {get_int(row, "essay_id"): row for row in candidate}
    ids = sorted(set(ref_by_id) & set(cand_by_id))
    ref = [ref_by_id[i] for i in ids]
    cand = [cand_by_id[i] for i in ids]
    payload = {
        "reference_qwk": {"overall": metric_qwk(ref, "overall")} | {t: metric_qwk(ref, t) for t in TRAITS},
        "candidate_qwk": {"overall": metric_qwk(cand, "overall")} | {t: metric_qwk(cand, t) for t in TRAITS},
    }
    payload["delta_qwk"] = {
        key: payload["candidate_qwk"][key] - payload["reference_qwk"][key]
        for key in ("overall", *TRAITS)
    }
    buckets: dict[str, dict[str, Any]] = {}
    for name, pred in (
        ("short", lambda r: word_count(r) <= int(bucket_threshold)),
        ("long", lambda r: word_count(r) > int(bucket_threshold)),
    ):
        ref_b = [r for r in ref if pred(r)]
        cand_b = [cand_by_id[get_int(r, "essay_id")] for r in ref_b]
        if ref_b:
            ref_q = {"overall": metric_qwk(ref_b, "overall")} | {t: metric_qwk(ref_b, t) for t in TRAITS}
            cand_q = {"overall": metric_qwk(cand_b, "overall")} | {t: metric_qwk(cand_b, t) for t in TRAITS}
            buckets[name] = {
                "rows": len(ref_b),
                "reference": ref_q,
                "candidate": cand_q,
                "delta": {key: cand_q[key] - ref_q[key] for key in ("overall", *TRAITS)},
            }
    payload["buckets"] = buckets
    return payload


def write_dev_variant(args: argparse.Namespace, rule_summary: dict[str, Any]) -> dict[str, Any]:
    rule = Rule(**rule_summary["rule"])
    out_root = Path(args.output_root) / "dev_variants" / rule.tag
    cells = load_dev_cells(args, rule)
    cell_reports: list[dict[str, Any]] = []
    for cell in cells:
        out_dir = out_root / cell["outer"] / cell["split"] / f"seed{cell['seed']}"
        pred_path = out_dir / f"{cell['stem']}_selected_preds.json"
        qwk_path = out_dir / f"{cell['stem']}_selected_qwk.json"
        report_path = out_dir / f"{cell['stem']}_selected_report.json"
        save_json(pred_path, cell["rows"])
        save_json(qwk_path, qwk_payload(cell["rows"], pred_path))
        save_json(report_path, {"rule": rule_summary["rule"], "counts": cell["counts"]})
        cmp = compare_rows(cell["rows"], cell["reference_rows"], int(args.bucket_word_threshold))
        cell_reports.append(
            {
                "outer": cell["outer"],
                "split": cell["split"],
                "seed": cell["seed"],
                "counts": cell["counts"],
                **cmp,
            }
        )

    def collect_delta(metric: str) -> list[float]:
        return [float(cell["delta_qwk"][metric]) for cell in cell_reports]

    changed = sum(int(cell["counts"]["changed_rows"]) for cell in cell_reports)
    total_rows = sum(int(cell["counts"]["rows"]) for cell in cell_reports)
    short_delta = [float(cell["buckets"]["short"]["delta"]["overall"]) for cell in cell_reports if "short" in cell["buckets"]]
    long_delta = [float(cell["buckets"]["long"]["delta"]["overall"]) for cell in cell_reports if "long" in cell["buckets"]]
    result = {
        "rule": rule_summary["rule"],
        "tag": rule.tag,
        "output_root": str(out_root),
        "oof": rule_summary,
        "dev_overall_delta_vs_reference": mean(collect_delta("overall")),
        "dev_overall_positive_vs_reference": sum(1 for x in collect_delta("overall") if x > 0.0),
        "dev_organization_delta_vs_reference": mean(collect_delta("organization")),
        "dev_organization_positive_vs_reference": sum(1 for x in collect_delta("organization") if x > 0.0),
        "dev_short_overall_delta_vs_reference": mean(short_delta) if short_delta else 0.0,
        "dev_long_overall_delta_vs_reference": mean(long_delta) if long_delta else 0.0,
        "dev_changed_rows": changed,
        "dev_rows": total_rows,
        "dev_changed_ratio": changed / total_rows if total_rows else 0.0,
        "cells": cell_reports,
    }
    save_json(out_root / "e58_dev_variant_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    oof_pairs = load_oof_pairs(Path(args.oof_pair_root), split_csv(args.outer_folds), split_csv(args.inner_splits))
    rules = generate_rules()
    ranking: list[dict[str, Any]] = []
    for rule in rules:
        summary = summarize_oof_rule(rule, oof_pairs, float(args.min_oof_changed_ratio))
        if summary is not None:
            ranking.append(summary)
    ranking.sort(
        key=lambda x: (
            float(x["oof_overall_delta_mean"]),
            int(x["oof_overall_positive_sets"]),
            float(x["oof_organization_delta_mean"]),
        ),
        reverse=True,
    )
    selected = ranking[: int(args.top_k)]
    dev_results = [write_dev_variant(args, item) for item in selected]
    dev_results.sort(
        key=lambda x: (
            float(x["dev_overall_delta_vs_reference"]),
            int(x["dev_overall_positive_vs_reference"]),
            float(x["dev_organization_delta_vs_reference"]),
        ),
        reverse=True,
    )
    report = {
        "scope": "OOF train selects organization overlay rules; dev validates; no holdout; no outer_test",
        "outer_test_touched": False,
        "holdout_touched": False,
        "oof_pair_root": str(Path(args.oof_pair_root)),
        "dev_base_root": str(Path(args.dev_base_root)),
        "dev_donor_root": str(Path(args.dev_donor_root)),
        "dev_reference_root": str(Path(args.dev_reference_root)),
        "rules_evaluated": len(rules),
        "rules_after_min_changed_filter": len(ranking),
        "top_k_applied_to_dev": len(selected),
        "min_oof_changed_ratio": float(args.min_oof_changed_ratio),
        "selection_rule": "OOF ranks rules; dev is confirmation. Only consider holdout if dev overall delta >= +0.002 and positive >= 20/30.",
        "oof_ranking_top": ranking[:100],
        "dev_ranking": dev_results,
    }
    report_path = Path(args.summary_output) if str(args.summary_output).strip() else out_root / "e58_oof_rule_select_dev_summary.json"
    save_json(report_path, report)
    print(f"rules_evaluated={len(rules)}")
    print(f"rules_after_filter={len(ranking)}")
    print(f"dev_variants={len(dev_results)}")
    if ranking:
        print(
            "best_oof "
            f"{ranking[0]['tag']} overall={ranking[0]['oof_overall_delta_mean']:+.6f} "
            f"pos={ranking[0]['oof_overall_positive_sets']}/{ranking[0]['oof_sets']} "
            f"changed={ranking[0]['oof_changed_rows']} ({ranking[0]['oof_changed_ratio']:.2%})"
        )
    if dev_results:
        print(
            "best_dev "
            f"{dev_results[0]['tag']} overall={dev_results[0]['dev_overall_delta_vs_reference']:+.6f} "
            f"pos={dev_results[0]['dev_overall_positive_vs_reference']}/30 "
            f"org={dev_results[0]['dev_organization_delta_vs_reference']:+.6f} "
            f"changed={dev_results[0]['dev_changed_rows']} ({dev_results[0]['dev_changed_ratio']:.2%})"
        )
    print(f"summary_saved: {report_path}")


if __name__ == "__main__":
    main()
