#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prompt7_mapper import Prompt7Mapper


TRAITS = ("ideas", "organization", "style", "conventions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Overlay one trait from a donor *_selected_preds.json onto a base "
            "*_selected_preds.json, then recompute overall. Selection rules use "
            "prediction-side metadata only; no gold labels are consulted."
        )
    )
    p.add_argument("--base-preds", required=True)
    p.add_argument("--donor-preds", required=True)
    p.add_argument("--output-preds", required=True)
    p.add_argument("--report-output", required=True)
    p.add_argument("--trait", choices=TRAITS, default="organization")
    p.add_argument("--name", default="trait_overlay")
    p.add_argument("--word-le", type=int, default=None)
    p.add_argument("--word-gt", type=int, default=None)
    p.add_argument("--base-trait-le", type=int, default=None)
    p.add_argument("--base-trait-ge", type=int, default=None)
    p.add_argument("--donor-max-prob-ge", type=float, default=None)
    p.add_argument("--donor-margin-ge", type=float, default=None)
    p.add_argument(
        "--direction",
        choices=("any", "changed", "up", "down", "no_down", "no_up"),
        default="changed",
        help="Direction filter comparing donor trait value against base trait value.",
    )
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty list: {path}")
    return rows


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


def recompute_overall(row: dict[str, Any]) -> None:
    row["overall"] = int(
        Prompt7Mapper.compute_overall_score(
            get_int(row, "ideas"),
            get_int(row, "organization"),
            get_int(row, "style"),
            get_int(row, "conventions"),
        )
    )


def direction_ok(direction: str, base_value: int, donor_value: int) -> bool:
    if direction == "any":
        return True
    if direction == "changed":
        return donor_value != base_value
    if direction == "up":
        return donor_value > base_value
    if direction == "down":
        return donor_value < base_value
    if direction == "no_down":
        return donor_value >= base_value
    if direction == "no_up":
        return donor_value <= base_value
    raise ValueError(f"Unsupported direction: {direction}")


def should_overlay(args: argparse.Namespace, base_row: dict[str, Any], donor_row: dict[str, Any]) -> tuple[bool, list[str]]:
    trait = str(args.trait)
    reasons: list[str] = []
    base_value = get_int(base_row, trait)
    donor_value = get_int(donor_row, trait, base_value)
    wc = word_count(base_row)

    if args.word_le is not None:
        if wc > int(args.word_le):
            return False, reasons
        reasons.append(f"word_count<={int(args.word_le)}")

    if args.word_gt is not None:
        if wc <= int(args.word_gt):
            return False, reasons
        reasons.append(f"word_count>{int(args.word_gt)}")

    if args.base_trait_le is not None:
        if base_value > int(args.base_trait_le):
            return False, reasons
        reasons.append(f"base_{trait}<={int(args.base_trait_le)}")

    if args.base_trait_ge is not None:
        if base_value < int(args.base_trait_ge):
            return False, reasons
        reasons.append(f"base_{trait}>={int(args.base_trait_ge)}")

    if args.donor_max_prob_ge is not None:
        value = get_float(donor_row, f"bert_{trait}_max_prob")
        if value < float(args.donor_max_prob_ge):
            return False, reasons
        reasons.append(f"donor_{trait}_max_prob>={float(args.donor_max_prob_ge):.4f}")

    if args.donor_margin_ge is not None:
        value = get_float(donor_row, f"bert_{trait}_margin")
        if value < float(args.donor_margin_ge):
            return False, reasons
        reasons.append(f"donor_{trait}_margin>={float(args.donor_margin_ge):.4f}")

    if not direction_ok(str(args.direction), base_value, donor_value):
        return False, reasons
    reasons.append(f"direction={args.direction}")
    return True, reasons


def copy_trait(dst: dict[str, Any], src: dict[str, Any], trait: str) -> bool:
    old_value = get_int(dst, trait)
    new_value = get_int(src, trait, old_value)
    dst[trait] = int(new_value)
    dst[f"pred_{trait}"] = int(new_value)

    for key in (
        f"{trait}_reasoning",
        f"bert_{trait}_pred",
        f"bert_{trait}_probs",
        f"bert_{trait}_max_prob",
        f"bert_{trait}_margin",
    ):
        if key in src:
            dst[key] = src[key]

    baseline_key = f"baseline_{trait}"
    used_key = f"sidecar_used_{trait}"
    if baseline_key in dst:
        dst[used_key] = bool(get_int(dst, trait) != get_int(dst, baseline_key))
    return old_value != new_value


def main() -> None:
    args = parse_args()
    trait = str(args.trait)

    base_path = Path(args.base_preds)
    donor_path = Path(args.donor_preds)
    output_path = Path(args.output_preds)
    report_path = Path(args.report_output)

    base_rows = load_rows(base_path)
    donor_rows = load_rows(donor_path)
    base_by_id = {get_int(row, "essay_id"): row for row in base_rows}
    donor_by_id = {get_int(row, "essay_id"): row for row in donor_rows}
    if set(base_by_id) != set(donor_by_id):
        missing_base = sorted(set(donor_by_id) - set(base_by_id))[:10]
        missing_donor = sorted(set(base_by_id) - set(donor_by_id))[:10]
        raise ValueError(f"essay_id mismatch; missing_base={missing_base} missing_donor={missing_donor}")

    counts: dict[str, int] = {
        "rows": 0,
        "eligible_rows": 0,
        "changed_rows": 0,
        "unchanged_eligible_rows": 0,
        "base_kept_rows": 0,
    }
    change_direction_counts = {"up": 0, "down": 0, "same": 0}
    word_buckets = {
        "short_le200": {"eligible": 0, "changed": 0},
        "long_gt200": {"eligible": 0, "changed": 0},
        "short_le240": {"eligible": 0, "changed": 0},
        "long_gt240": {"eligible": 0, "changed": 0},
    }
    out_rows: list[dict[str, Any]] = []

    for base_row in base_rows:
        essay_id = get_int(base_row, "essay_id")
        donor_row = donor_by_id[essay_id]
        out = dict(base_row)
        counts["rows"] += 1

        ok, reasons = should_overlay(args, base_row, donor_row)
        wc = word_count(base_row)
        if ok:
            counts["eligible_rows"] += 1
            if wc <= 200:
                word_buckets["short_le200"]["eligible"] += 1
            else:
                word_buckets["long_gt200"]["eligible"] += 1
            if wc <= 240:
                word_buckets["short_le240"]["eligible"] += 1
            else:
                word_buckets["long_gt240"]["eligible"] += 1

            old_value = get_int(base_row, trait)
            new_value = get_int(donor_row, trait, old_value)
            changed = copy_trait(out, donor_row, trait)
            if changed:
                counts["changed_rows"] += 1
                if wc <= 200:
                    word_buckets["short_le200"]["changed"] += 1
                else:
                    word_buckets["long_gt200"]["changed"] += 1
                if wc <= 240:
                    word_buckets["short_le240"]["changed"] += 1
                else:
                    word_buckets["long_gt240"]["changed"] += 1
                if new_value > old_value:
                    change_direction_counts["up"] += 1
                elif new_value < old_value:
                    change_direction_counts["down"] += 1
                else:
                    change_direction_counts["same"] += 1
            else:
                counts["unchanged_eligible_rows"] += 1
        else:
            counts["base_kept_rows"] += 1

        recompute_overall(out)
        out["overlay_name"] = str(args.name)
        out["overlay_trait"] = trait
        out["overlay_applied"] = bool(ok)
        out["overlay_reasons"] = reasons
        out_rows.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "name": str(args.name),
        "trait": trait,
        "base_preds": str(base_path),
        "donor_preds": str(donor_path),
        "output_preds": str(output_path),
        "selection_uses_gold": False,
        "filters": {
            "word_le": args.word_le,
            "word_gt": args.word_gt,
            "base_trait_le": args.base_trait_le,
            "base_trait_ge": args.base_trait_ge,
            "donor_max_prob_ge": args.donor_max_prob_ge,
            "donor_margin_ge": args.donor_margin_ge,
            "direction": str(args.direction),
        },
        "counts": counts,
        "ratios": {key: float(value / counts["rows"]) if counts["rows"] else 0.0 for key, value in counts.items()},
        "change_direction_counts": change_direction_counts,
        "word_buckets": word_buckets,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={counts['rows']}")
    print(
        f"eligible={counts['eligible_rows']} changed={counts['changed_rows']} "
        f"changed_ratio={counts['changed_rows'] / counts['rows'] if counts['rows'] else 0.0:.4f}"
    )
    print(f"saved: {output_path}")
    print(f"report_saved: {report_path}")


if __name__ == "__main__":
    main()
