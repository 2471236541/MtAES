#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact outer-test result summary.")
    parser.add_argument("--outer-summary", required=True, help="Full outer summary JSON.")
    parser.add_argument("--output", required=True, help="Compact key-results JSON.")
    parser.add_argument("--note", default="", help="Optional short note.")
    return parser.parse_args()


def nested_get(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def metric_block(payload: dict[str, Any], root: str, metric: str) -> dict[str, Any]:
    return {
        "mean": nested_get(payload, ["fold_level", root, metric, "mean"]),
        "std": nested_get(payload, ["fold_level", root, metric, "std"]),
        "min": nested_get(payload, ["fold_level", root, metric, "min"]),
        "max": nested_get(payload, ["fold_level", root, metric, "max"]),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.outer_summary)
    payload = json.loads(source.read_text(encoding="utf-8"))

    fold_rows = []
    for row in payload.get("fold_rows", []):
        outer = row.get("outer")
        candidate = row.get("candidate", {})
        base = row.get("base", {})
        gain = row.get("candidate_gain_vs_base", {})
        fold_rows.append(
            {
                "outer": outer,
                "base_overall": base.get("overall"),
                "candidate_overall": candidate.get("overall"),
                "gain_overall": gain.get("overall"),
                "candidate_ideas": candidate.get("ideas"),
                "candidate_organization": candidate.get("organization"),
                "candidate_style": candidate.get("style"),
                "candidate_conventions": candidate.get("conventions"),
            }
        )

    compact = {
        "source_outer_summary": str(source),
        "note": str(args.note or ""),
        "scope": payload.get("scope"),
        "outer_test_touched": payload.get("outer_test_touched"),
        "selection_on_outer": payload.get("selection_on_outer"),
        "selection_uses_gold": payload.get("selection_uses_gold"),
        "rule": payload.get("rule"),
        "overall": {
            "base": metric_block(payload, "base", "overall"),
            "candidate": metric_block(payload, "candidate", "overall"),
            "gain_vs_base": metric_block(payload, "candidate_gain_vs_base", "overall"),
        },
        "traits": {
            trait: {
                "base": metric_block(payload, "base", trait),
                "candidate": metric_block(payload, "candidate", trait),
                "gain_vs_base": metric_block(payload, "candidate_gain_vs_base", trait),
            }
            for trait in ("ideas", "organization", "style", "conventions")
        },
        "fold_rows": fold_rows,
        "counts": payload.get("counts"),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    print(
        "outer overall",
        f"base={compact['overall']['base']['mean']}",
        f"candidate={compact['overall']['candidate']['mean']}",
        f"gain={compact['overall']['gain_vs_base']['mean']}",
    )


if __name__ == "__main__":
    main()
