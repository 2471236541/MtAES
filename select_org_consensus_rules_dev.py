#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from statistics import mean
from typing import Any

from prompt7_mapper import Prompt7Mapper
from select_and_apply_org_overlay_rules_oof import (
    compare_rows,
    load_json,
    qwk_payload,
    save_json,
    seed_csv,
    split_csv,
    stem,
)


TRAITS = ("ideas", "organization", "style", "conventions")
METRICS = ("overall", *TRAITS)


@dataclass(frozen=True)
class DonorSpec:
    name: str
    root: str
    template: str


@dataclass(frozen=True)
class ConsensusRule:
    donor_a: str
    donor_b: str
    direction: str
    prob_ge: float
    margin_ge: float
    prob_mode: str
    margin_mode: str
    word_filter: str

    @property
    def tag(self) -> str:
        p = int(round(self.prob_ge * 100))
        m = int(round(self.margin_ge * 100))
        return (
            f"{self.donor_a}__{self.donor_b}__{self.direction}_{self.word_filter}_"
            f"p{p:02d}_{self.prob_mode}_m{m:02d}_{self.margin_mode}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dev-only organization multi-donor consensus sweep. This selects no rule "
            "on holdout and never touches outer_test."
        )
    )
    p.add_argument("--base-root", required=True)
    p.add_argument("--reference-root", required=True, help="Current best dev root to compare against, usually e58 best dev.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--outer-folds", default="outer0,outer1,outer2,outer3,outer4")
    p.add_argument("--inner-splits", default="s0,s1,s2")
    p.add_argument("--seeds", required=True)
    p.add_argument("--prefix", default="v10ncv")
    p.add_argument("--split-kind", default="dev")
    p.add_argument("--top-k-save", type=int, default=8)
    p.add_argument("--bucket-word-threshold", type=int, default=200)
    p.add_argument("--summary-output", default="")
    p.add_argument(
        "--pair-mode",
        choices=("anchor_e48", "all_pairs"),
        default="anchor_e48",
        help="Which donor pairs to evaluate. Default preserves the original e59/e62 sweep.",
    )
    return p.parse_args()


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


def word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("essay_text", "")).split())


def donor_org(row: dict[str, Any]) -> int:
    if "bert_organization_pred" in row:
        return get_int(row, "bert_organization_pred")
    if "organization" in row:
        return get_int(row, "organization")
    return get_int(row, "pred_organization")


def direction_ok(direction: str, base_value: int, donor_value: int) -> bool:
    if direction == "changed":
        return donor_value != base_value
    if direction == "no_down":
        return donor_value >= base_value and donor_value != base_value
    if direction == "up":
        return donor_value > base_value
    if direction == "down":
        return donor_value < base_value
    raise ValueError(f"Unsupported direction: {direction}")


def threshold_ok(values: tuple[float, float], threshold: float, mode: str) -> bool:
    if mode == "both":
        return values[0] >= threshold and values[1] >= threshold
    if mode == "avg":
        return ((values[0] + values[1]) / 2.0) >= threshold
    raise ValueError(f"Unsupported threshold mode: {mode}")


def word_ok(word_filter: str, row: dict[str, Any]) -> bool:
    wc = word_count(row)
    if word_filter == "all":
        return True
    if word_filter.startswith("le"):
        return wc <= int(word_filter[2:])
    if word_filter.startswith("gt"):
        return wc > int(word_filter[2:])
    raise ValueError(f"Unsupported word filter: {word_filter}")


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


def load_donor_rows(spec: DonorSpec, outer: str, split: str, seed: int, stem_value: str) -> list[dict[str, Any]]:
    path = Path(spec.root) / outer / split / f"seed{seed}" / spec.template.format(stem=stem_value)
    if not path.exists():
        raise FileNotFoundError(f"Missing donor {spec.name}: {path}")
    return load_json(path)


def apply_consensus(
    base_rows: list[dict[str, Any]],
    donor_a_rows: list[dict[str, Any]],
    donor_b_rows: list[dict[str, Any]],
    rule: ConsensusRule,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a_by_id = {get_int(row, "essay_id"): row for row in donor_a_rows}
    b_by_id = {get_int(row, "essay_id"): row for row in donor_b_rows}
    out_rows: list[dict[str, Any]] = []
    counts = {
        "rows": 0,
        "agreement_rows": 0,
        "eligible_rows": 0,
        "changed_rows": 0,
        "up_rows": 0,
        "down_rows": 0,
        "base_kept_rows": 0,
        "direction_filtered_rows": 0,
        "prob_filtered_rows": 0,
        "margin_filtered_rows": 0,
        "word_filtered_rows": 0,
    }
    for base in base_rows:
        essay_id = get_int(base, "essay_id")
        if essay_id not in a_by_id or essay_id not in b_by_id:
            raise ValueError(f"Missing donor row essay_id={essay_id}")
        a = a_by_id[essay_id]
        b = b_by_id[essay_id]
        out = dict(base)
        counts["rows"] += 1
        base_org = get_int(base, "organization")
        a_org = donor_org(a)
        b_org = donor_org(b)

        if a_org != b_org:
            counts["base_kept_rows"] += 1
            out["overlay_applied"] = False
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = ["donor_disagreement"]
            out_rows.append(out)
            continue
        counts["agreement_rows"] += 1

        if not word_ok(rule.word_filter, base):
            counts["word_filtered_rows"] += 1
            counts["base_kept_rows"] += 1
            out["overlay_applied"] = False
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = ["word_filtered"]
            out_rows.append(out)
            continue

        if not direction_ok(rule.direction, base_org, a_org):
            counts["direction_filtered_rows"] += 1
            counts["base_kept_rows"] += 1
            out["overlay_applied"] = False
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = ["direction_filtered"]
            out_rows.append(out)
            continue

        probs = (
            get_float(a, "bert_organization_max_prob"),
            get_float(b, "bert_organization_max_prob"),
        )
        margins = (
            get_float(a, "bert_organization_margin"),
            get_float(b, "bert_organization_margin"),
        )
        if not threshold_ok(probs, rule.prob_ge, rule.prob_mode):
            counts["prob_filtered_rows"] += 1
            counts["base_kept_rows"] += 1
            out["overlay_applied"] = False
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = ["prob_filtered"]
            out_rows.append(out)
            continue
        if not threshold_ok(margins, rule.margin_ge, rule.margin_mode):
            counts["margin_filtered_rows"] += 1
            counts["base_kept_rows"] += 1
            out["overlay_applied"] = False
            out["overlay_name"] = rule.tag
            out["overlay_trait"] = "organization"
            out["overlay_reasons"] = ["margin_filtered"]
            out_rows.append(out)
            continue

        counts["eligible_rows"] += 1
        if a_org != base_org:
            counts["changed_rows"] += 1
            if a_org > base_org:
                counts["up_rows"] += 1
            elif a_org < base_org:
                counts["down_rows"] += 1

        out["organization"] = int(a_org)
        out["pred_organization"] = int(a_org)
        out["bert_organization_pred"] = int(a_org)
        for key in ("bert_organization_probs", "bert_organization_max_prob", "bert_organization_margin", "organization_reasoning"):
            if key in a:
                out[key] = a[key]
        out["consensus_donor_a"] = rule.donor_a
        out["consensus_donor_b"] = rule.donor_b
        out["consensus_donor_a_prob"] = probs[0]
        out["consensus_donor_b_prob"] = probs[1]
        out["consensus_donor_a_margin"] = margins[0]
        out["consensus_donor_b_margin"] = margins[1]
        out["overlay_applied"] = True
        out["overlay_name"] = rule.tag
        out["overlay_trait"] = "organization"
        out["overlay_reasons"] = [
            "donor_agreement",
            f"direction={rule.direction}",
            f"word={rule.word_filter}",
            f"prob_{rule.prob_mode}>={rule.prob_ge:.2f}",
            f"margin_{rule.margin_mode}>={rule.margin_ge:.2f}",
        ]
        if "baseline_organization" in out:
            out["sidecar_used_organization"] = bool(get_int(out, "organization") != get_int(out, "baseline_organization"))
        recompute_overall(out)
        out_rows.append(out)
    return out_rows, counts


def donor_specs(panel_root: Path) -> dict[str, DonorSpec]:
    specs = {
        "e48": DonorSpec(
            "e48",
            str(panel_root / "e48_org_highscore_devfirst_20260427" / "_org_highscore_only_dev"),
            "{stem}_org_highscore_with_bert.json",
        ),
        "e50_c3120o020": DonorSpec(
            "e50_c3120o020",
            str(panel_root / "e50_orgloss_c3120_ord020_20260427" / "_org_highscore_only_dev"),
            "{stem}_org_highscore_with_bert.json",
        ),
        "e50_c3120o035": DonorSpec(
            "e50_c3120o035",
            str(panel_root / "e50_orgloss_c3120_ord035_20260427" / "_org_highscore_only_dev"),
            "{stem}_org_highscore_with_bert.json",
        ),
        "e50_c3150o020": DonorSpec(
            "e50_c3150o020",
            str(panel_root / "e50_orgloss_c3150_ord020_20260427" / "_org_highscore_only_dev"),
            "{stem}_org_highscore_with_bert.json",
        ),
        "e52_bertbase": DonorSpec(
            "e52_bertbase",
            str(panel_root / "e52_orgbackbone_bertbase_c3135_ord020_20260427" / "_org_highscore_only_dev"),
            "{stem}_org_highscore_with_bert.json",
        ),
        "e61_len512": DonorSpec(
            "e61_len512",
            str(panel_root / "e61_org_distilbert_len512_c3135_ord020_20260427" / "_org_highscore_only_dev"),
            "{stem}_org_highscore_with_bert.json",
        ),
        "e40_orgv3": DonorSpec(
            "e40_orgv3",
            str(panel_root / "e40_org_struct_v3_compact_devfirst_20260426" / "orgv3_donor_dev" / "_org_only"),
            "{stem}_orgv3_with_bert.json",
        ),
    }
    e63_root = panel_root / "e63_org_flow_v1_devfirst_20260427" / "_org_highscore_only_dev"
    if e63_root.exists():
        specs["e63_flow_v1"] = DonorSpec(
            "e63_flow_v1",
            str(e63_root),
            "{stem}_org_highscore_with_bert.json",
        )
    return specs


def build_rules(specs: dict[str, DonorSpec], pair_mode: str = "anchor_e48") -> list[ConsensusRule]:
    if pair_mode == "anchor_e48":
        anchor = "e48"
        pairs = [(anchor, name) for name in specs if name != anchor]
    elif pair_mode == "all_pairs":
        pairs = list(combinations(specs.keys(), 2))
    else:
        raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    rules: list[ConsensusRule] = []
    for (donor_a, donor_b), direction, prob, margin, prob_mode, margin_mode, word_filter in product(
        pairs,
        ["changed", "no_down"],
        [0.50, 0.55, 0.60],
        [0.15, 0.25, 0.30],
        ["both", "avg"],
        ["both", "avg"],
        ["all", "le240"],
    ):
        rules.append(
            ConsensusRule(
                donor_a=donor_a,
                donor_b=donor_b,
                direction=direction,
                prob_ge=prob,
                margin_ge=margin,
                prob_mode=prob_mode,
                margin_mode=margin_mode,
                word_filter=word_filter,
            )
        )
    return rules


def load_cells(args: argparse.Namespace, specs: dict[str, DonorSpec]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    base_root = Path(args.base_root)
    reference_root = Path(args.reference_root)
    needed_donors = set()
    for rule in build_rules(specs, str(args.pair_mode)):
        needed_donors.add(rule.donor_a)
        needed_donors.add(rule.donor_b)

    for outer in split_csv(args.outer_folds):
        for split in split_csv(args.inner_splits):
            for seed in seed_csv(args.seeds):
                s = stem(str(args.prefix), outer, split, seed, str(args.split_kind))
                base_path = base_root / outer / split / f"seed{seed}" / f"{s}_selected_preds.json"
                reference_path = reference_root / outer / split / f"seed{seed}" / f"{s}_selected_preds.json"
                if not base_path.exists():
                    raise FileNotFoundError(f"Missing base: {base_path}")
                if not reference_path.exists():
                    raise FileNotFoundError(f"Missing reference: {reference_path}")
                donors = {
                    name: load_donor_rows(specs[name], outer, split, seed, s)
                    for name in sorted(needed_donors)
                }
                cells.append(
                    {
                        "outer": outer,
                        "split": split,
                        "seed": seed,
                        "stem": s,
                        "base_rows": load_json(base_path),
                        "reference_rows": load_json(reference_path),
                        "donors": donors,
                    }
                )
    return cells


def evaluate_rule(rule: ConsensusRule, cells: list[dict[str, Any]], bucket_threshold: int) -> dict[str, Any]:
    cell_reports: list[dict[str, Any]] = []
    total_counts = {
        "rows": 0,
        "agreement_rows": 0,
        "eligible_rows": 0,
        "changed_rows": 0,
        "up_rows": 0,
        "down_rows": 0,
        "base_kept_rows": 0,
    }
    for cell in cells:
        out_rows, counts = apply_consensus(
            cell["base_rows"],
            cell["donors"][rule.donor_a],
            cell["donors"][rule.donor_b],
            rule,
        )
        cmp = compare_rows(out_rows, cell["reference_rows"], bucket_threshold)
        cell_reports.append(
            {
                "outer": cell["outer"],
                "split": cell["split"],
                "seed": cell["seed"],
                "counts": counts,
                **cmp,
            }
        )
        for key in total_counts:
            total_counts[key] += int(counts.get(key, 0))

    def deltas(metric: str) -> list[float]:
        return [float(cell["delta_qwk"][metric]) for cell in cell_reports]

    short = [float(cell["buckets"]["short"]["delta"]["overall"]) for cell in cell_reports if "short" in cell["buckets"]]
    long = [float(cell["buckets"]["long"]["delta"]["overall"]) for cell in cell_reports if "long" in cell["buckets"]]
    result = {
        "rule": asdict(rule),
        "tag": rule.tag,
        "dev_overall_delta_vs_reference": mean(deltas("overall")),
        "dev_overall_positive_vs_reference": sum(1 for x in deltas("overall") if x > 0.0),
        "dev_organization_delta_vs_reference": mean(deltas("organization")),
        "dev_organization_positive_vs_reference": sum(1 for x in deltas("organization") if x > 0.0),
        "dev_short_overall_delta_vs_reference": mean(short) if short else 0.0,
        "dev_long_overall_delta_vs_reference": mean(long) if long else 0.0,
        "counts": total_counts,
        "changed_ratio": total_counts["changed_rows"] / total_counts["rows"] if total_counts["rows"] else 0.0,
        "agreement_ratio": total_counts["agreement_rows"] / total_counts["rows"] if total_counts["rows"] else 0.0,
        "cell_reports": cell_reports,
    }
    return result


def write_variant(
    args: argparse.Namespace,
    rule: ConsensusRule,
    cells: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    out_root = Path(args.output_root) / "dev_variants" / rule.tag
    for cell in cells:
        out_rows, counts = apply_consensus(
            cell["base_rows"],
            cell["donors"][rule.donor_a],
            cell["donors"][rule.donor_b],
            rule,
        )
        out_dir = out_root / cell["outer"] / cell["split"] / f"seed{cell['seed']}"
        pred_path = out_dir / f"{cell['stem']}_selected_preds.json"
        qwk_path = out_dir / f"{cell['stem']}_selected_qwk.json"
        report_path = out_dir / f"{cell['stem']}_selected_report.json"
        save_json(pred_path, out_rows)
        save_json(qwk_path, qwk_payload(out_rows, pred_path))
        save_json(
            report_path,
            {
                "scope": "dev-only e59 organization multi-donor consensus; no holdout; no outer_test",
                "selection_uses_gold": False,
                "rule": asdict(rule) | {"tag": rule.tag},
                "counts": counts,
            },
        )
    slim = dict(result)
    slim.pop("cell_reports", None)
    slim["output_root"] = str(out_root)
    save_json(out_root / "e59_dev_variant_summary.json", slim)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    panel_root = output_root.parents[0]
    specs = donor_specs(panel_root)
    rules = build_rules(specs, str(args.pair_mode))
    cells = load_cells(args, specs)

    results = [evaluate_rule(rule, cells, int(args.bucket_word_threshold)) for rule in rules]
    results.sort(
        key=lambda x: (
            float(x["dev_overall_delta_vs_reference"]),
            int(x["dev_overall_positive_vs_reference"]),
            float(x["dev_organization_delta_vs_reference"]),
        ),
        reverse=True,
    )

    top_k = min(int(args.top_k_save), len(results))
    rule_by_tag = {rule.tag: rule for rule in rules}
    for item in results[:top_k]:
        write_variant(args, rule_by_tag[str(item["tag"])], cells, item)

    report = {
        "scope": "dev-only e59 organization multi-donor consensus sweep; no holdout; no outer_test",
        "outer_test_touched": False,
        "holdout_touched": False,
        "base_root": str(Path(args.base_root)),
        "reference_root": str(Path(args.reference_root)),
        "output_root": str(output_root),
        "rules_evaluated": len(results),
        "pair_mode": str(args.pair_mode),
        "top_k_saved": top_k,
        "donors": {name: asdict(spec) for name, spec in specs.items()},
        "ranking": results,
        "selection_rule": "Use dev only for exploration. Only run holdout once if a fixed rule beats e58 dev by a meaningful margin.",
    }
    summary_path = Path(args.summary_output) if str(args.summary_output).strip() else output_root / "e59_org_consensus_dev_summary.json"
    save_json(summary_path, report)

    print(f"rules_evaluated={len(results)}")
    if results:
        best = results[0]
        print(
            "best_dev "
            f"{best['tag']} overall={best['dev_overall_delta_vs_reference']:+.6f} "
            f"pos={best['dev_overall_positive_vs_reference']}/30 "
            f"org={best['dev_organization_delta_vs_reference']:+.6f} "
            f"changed={best['counts']['changed_rows']} ({best['changed_ratio']:.2%})"
        )
    print(f"summary_saved: {summary_path}")


if __name__ == "__main__":
    main()
