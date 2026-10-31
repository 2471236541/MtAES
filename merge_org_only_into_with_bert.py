#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ORG_FIELDS = (
    "bert_organization_pred",
    "bert_organization_probs",
    "bert_organization_max_prob",
    "bert_organization_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace only bert_organization_* fields in an existing with_bert JSON."
    )
    parser.add_argument("--base-with-bert", required=True, help="Current source with_bert JSON.")
    parser.add_argument("--org-only-with-bert", required=True, help="Organization-only infer output JSON.")
    parser.add_argument("--output", required=True, help="Merged output JSON.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"JSON must be a list: {path}")
    return rows


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_with_bert)
    org_path = Path(args.org_only_with_bert)
    output_path = Path(args.output)

    base_rows = load_rows(base_path)
    org_rows = load_rows(org_path)
    if len(base_rows) != len(org_rows):
        raise ValueError(f"Row count mismatch: base={len(base_rows)} org={len(org_rows)}")

    org_by_id = {row.get("essay_id"): row for row in org_rows}
    merged: list[dict[str, Any]] = []
    replaced = 0

    for base_row in base_rows:
        essay_id = base_row.get("essay_id")
        if essay_id not in org_by_id:
            raise ValueError(f"Missing essay_id in org-only rows: {essay_id}")
        org_row = org_by_id[essay_id]
        out = dict(base_row)
        changed = False
        for key in ORG_FIELDS:
            if key not in org_row:
                raise ValueError(f"Missing field in org-only row: {key}")
            new_value = org_row[key]
            changed = changed or (out.get(key) != new_value)
            out[key] = new_value
        replaced += int(changed)
        merged.append(out)

    save_rows(output_path, merged)
    print(f"rows={len(merged)}")
    print(f"org_fields_replaced={replaced}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
