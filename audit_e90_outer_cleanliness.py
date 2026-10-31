#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("/data/rcy/rcy/LLMs-AES-IELTS")
DATA_ROOT = ROOT / "sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424"
BASE_ROOT = ROOT / "sidecar_preds/v10ncv_clean_outer_e59_sealed_20260427/e59_final_outer"
E90_ROOT = ROOT / "sidecar_preds/v10ncv_clean_outer_e90_e79_e87_sealed_20260504"
E79_ROOT = E90_ROOT / "e79_ideas_donor_outer"
E87_ROOT = E90_ROOT / "e87_ideas_donor_outer"
E90_OUT = E90_ROOT / "e90_primary_outer"
SUMMARY_PATH = E90_ROOT / "e90_outer_summary.json"
MANIFEST_PATH = E90_ROOT / "sealed_manifest.json"
OUT_JSON = E90_ROOT / "e90_outer_cleanliness_audit.json"
OUT_MD = E90_ROOT / "e90_outer_cleanliness_audit.md"
APPLY_SCRIPT = ROOT / "apply_e90_e79_e87_outer.py"

OUTERS = [f"outer{i}" for i in range(5)]
SEEDS = [20260325, 20260412]
SIGNAL_FIELDS = [
    "bert_ideas_pred",
    "bert_ideas_probs",
    "bert_ideas_max_prob",
    "bert_ideas_margin",
]


def load_e90_module() -> Any:
    spec = importlib.util.spec_from_file_location("apply_e90", APPLY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {APPLY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_gold(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("gold_")}


def ids(path: Path) -> set[int]:
    return {int(row["essay_id"]) for row in read_json(path)}


def cell_paths(outer: str, seed: int) -> dict[str, Path]:
    stem = f"v10ncv_{outer}_seed{seed}_outer"
    return {
        "base": BASE_ROOT / outer / "refit" / f"seed{seed}" / f"{stem}_selected_preds.json",
        "e79": E79_ROOT / outer / "refit" / f"seed{seed}" / f"{stem}_ideas_with_bert.json",
        "e87": E87_ROOT / outer / "refit" / f"seed{seed}" / f"{stem}_ideas_with_bert.json",
        "final": E90_OUT / outer / "refit" / f"seed{seed}" / f"{stem}_selected_preds.json",
    }


def split_overlap_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    final_sets: dict[str, set[int]] = {}
    for outer in OUTERS:
        train_ids = ids(DATA_ROOT / outer / "refit/sidecar_train.json")
        dev_ids = ids(DATA_ROOT / outer / "refit/sidecar_dev.json")
        final_ids = ids(DATA_ROOT / outer / "sidecar_final_test.json")
        outer_test_ids = ids(DATA_ROOT / outer / "sidecar_outer_test.json")
        final_sets[outer] = final_ids
        rows.append(
            {
                "outer": outer,
                "refit_train": len(train_ids),
                "refit_dev": len(dev_ids),
                "final_test": len(final_ids),
                "train_final_overlap": len(train_ids & final_ids),
                "dev_final_overlap": len(dev_ids & final_ids),
                "train_dev_overlap": len(train_ids & dev_ids),
                "outer_test_equals_final_test": outer_test_ids == final_ids,
            }
        )

    pairwise = []
    for i, left in enumerate(OUTERS):
        for right in OUTERS[i + 1 :]:
            pairwise.append({"left": left, "right": right, "final_test_overlap": len(final_sets[left] & final_sets[right])})
    return {
        "rows": rows,
        "pairwise_final_test_overlap": pairwise,
        "final_test_union": len(set().union(*final_sets.values())),
        "final_test_sum": sum(len(value) for value in final_sets.values()),
        "pass": all(
            row["train_final_overlap"] == 0 and row["dev_final_overlap"] == 0 and row["train_dev_overlap"] == 0
            for row in rows
        )
        and all(row["final_test_overlap"] == 0 for row in pairwise),
    }


def gold_stripped_replay_audit(e90: Any) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    total_rows = 0
    total_changed = 0
    for outer in OUTERS:
        for seed in SEEDS:
            paths = cell_paths(outer, seed)
            base_rows = [strip_gold(row) for row in read_json(paths["base"])]
            donor_rows = {
                "e79": [strip_gold(row) for row in read_json(paths["e79"])],
                "e87": [strip_gold(row) for row in read_json(paths["e87"])],
            }
            replay_rows, counts = e90.apply_rule(
                base_rows,
                donor_rows,
                label="e90_e79_e87_all_both_both_dir_up0m0_dn0m0_step3",
                max_step=3,
            )
            final_rows = {int(row["essay_id"]): row for row in read_json(paths["final"])}
            total_rows += int(counts["rows"])
            total_changed += int(counts["changed"])
            for row in replay_rows:
                final = final_rows[int(row["essay_id"])]
                for key in ["ideas", "overall", "pred_ideas", "pred_overall", "e90_applied", "e90_direction", "e90_ideas_from", "e90_ideas_to"]:
                    if row.get(key) != final.get(key):
                        mismatches.append(
                            {
                                "outer": outer,
                                "seed": seed,
                                "essay_id": int(row["essay_id"]),
                                "field": key,
                                "replay": row.get(key),
                                "final": final.get(key),
                            }
                        )
                        break
    return {
        "description": "Replay e90 decisions after removing all gold_* fields from base and donor rows.",
        "rows": total_rows,
        "changed": total_changed,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "pass": len(mismatches) == 0,
    }


def exactness_audit() -> dict[str, Any]:
    total = changed = 0
    old_exact = new_exact = old_overall_abs = new_overall_abs = 0
    by_direction: dict[str, dict[str, int]] = {
        "up": {"rows": 0, "old_exact": 0, "new_exact": 0},
        "down": {"rows": 0, "old_exact": 0, "new_exact": 0},
    }
    for outer in OUTERS:
        for seed in SEEDS:
            for row in read_json(cell_paths(outer, seed)["final"]):
                total += 1
                old_ideas = int(row.get("e90_ideas_from", row["ideas"]))
                new_ideas = int(row["ideas"])
                gold_ideas = int(row["gold_ideas"])
                old_overall = int(row.get("e90_overall_from", row.get("pred_overall", row["overall"])))
                new_overall = int(row["overall"])
                gold_overall = int(row["gold_overall"])
                old_overall_abs += abs(old_overall - gold_overall)
                new_overall_abs += abs(new_overall - gold_overall)
                if old_ideas == gold_ideas:
                    old_exact += 1
                if new_ideas == gold_ideas:
                    new_exact += 1
                if row.get("e90_applied"):
                    changed += 1
                    direction = str(row.get("e90_direction"))
                    if direction in by_direction:
                        by_direction[direction]["rows"] += 1
                        by_direction[direction]["old_exact"] += int(old_ideas == gold_ideas)
                        by_direction[direction]["new_exact"] += int(new_ideas == gold_ideas)
    return {
        "description": "If gold leaked into the gate, changed rows would be suspiciously near-perfect. They are not.",
        "rows": total,
        "changed": changed,
        "all_old_ideas_exact_ratio": old_exact / total,
        "all_new_ideas_exact_ratio": new_exact / total,
        "changed_by_direction": by_direction,
        "overall_abs_error_old_mean": old_overall_abs / total,
        "overall_abs_error_new_mean": new_overall_abs / total,
    }


def shuffled_donor_rows(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    shuffled_signals = [{field: row.get(field) for field in SIGNAL_FIELDS if field in row} for row in rows]
    rng.shuffle(shuffled_signals)
    out = []
    for row, signal in zip(rows, shuffled_signals):
        item = dict(row)
        for field, value in signal.items():
            item[field] = value
        out.append(item)
    return out


def shuffled_negative_control(e90: Any, n_shuffles: int = 50) -> dict[str, Any]:
    actual_summary = read_json(SUMMARY_PATH)
    actual_gain = float(actual_summary["fold_level"]["candidate_gain_vs_base"]["overall"]["mean"])
    actual_candidate = float(actual_summary["fold_level"]["candidate"]["overall"]["mean"])
    shuffle_rows: list[dict[str, Any]] = []

    for shuffle_idx in range(n_shuffles):
        rng = random.Random(910_000 + shuffle_idx)
        seed_cells: list[dict[str, Any]] = []
        for outer in OUTERS:
            for seed in SEEDS:
                paths = cell_paths(outer, seed)
                base_rows = read_json(paths["base"])
                donor_rows = {
                    "e79": shuffled_donor_rows(read_json(paths["e79"]), rng),
                    "e87": shuffled_donor_rows(read_json(paths["e87"]), rng),
                }
                cand_rows, counts = e90.apply_rule(
                    base_rows,
                    donor_rows,
                    label="shuffle_negative_control",
                    max_step=3,
                )
                base_metrics = e90.metric_block(base_rows)
                cand_metrics = e90.metric_block(cand_rows)
                seed_cells.append(
                    {
                        "outer": outer,
                        "seed": seed,
                        "base": base_metrics,
                        "candidate": cand_metrics,
                        "gain": e90.diff(cand_metrics, base_metrics),
                        "changed": int(counts["changed"]),
                    }
                )

        fold_rows = []
        for outer in OUTERS:
            rows = [row for row in seed_cells if row["outer"] == outer]
            fold_rows.append(
                {
                    "outer": outer,
                    "candidate_overall": mean(float(row["candidate"]["overall"]) for row in rows),
                    "gain_overall": mean(float(row["gain"]["overall"]) for row in rows),
                    "changed": sum(int(row["changed"]) for row in rows),
                }
            )
        shuffle_rows.append(
            {
                "shuffle": shuffle_idx,
                "candidate_overall": mean(row["candidate_overall"] for row in fold_rows),
                "gain_overall": mean(row["gain_overall"] for row in fold_rows),
                "changed": sum(row["changed"] for row in fold_rows),
            }
        )

    gains = [float(row["gain_overall"]) for row in shuffle_rows]
    candidates = [float(row["candidate_overall"]) for row in shuffle_rows]
    return {
        "description": "Donor prediction/probability/margin fields are randomly reassigned across essays. A real signal should disappear.",
        "n_shuffles": n_shuffles,
        "actual_candidate_overall": actual_candidate,
        "actual_gain_overall": actual_gain,
        "shuffle_candidate_overall_mean": mean(candidates),
        "shuffle_candidate_overall_std": pstdev(candidates),
        "shuffle_gain_overall_mean": mean(gains),
        "shuffle_gain_overall_std": pstdev(gains),
        "shuffle_gain_overall_min": min(gains),
        "shuffle_gain_overall_max": max(gains),
        "shuffle_gain_ge_actual_count": sum(gain >= actual_gain for gain in gains),
        "shuffle_rows": shuffle_rows,
        "pass": all(gain < actual_gain for gain in gains),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    split = payload["split_overlap"]
    replay = payload["gold_stripped_replay"]
    exact = payload["exactness"]
    shuffle = payload["shuffled_negative_control"]
    lines = [
        "# E90 sealed outer cleanliness audit",
        "",
        "This audit is diagnostic only. It does not tune or change the frozen e90 rule.",
        "",
        "## Verdict",
        "",
        f"- data split overlap pass: {split['pass']}",
        f"- gold-stripped replay pass: {replay['pass']}",
        f"- shuffled donor negative control pass: {shuffle['pass']}",
        "",
        "## Split Overlap",
        "",
        "| Outer | train | dev | final_test | train-final overlap | dev-final overlap | train-dev overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split["rows"]:
        lines.append(
            f"| {row['outer']} | {row['refit_train']} | {row['refit_dev']} | {row['final_test']} | "
            f"{row['train_final_overlap']} | {row['dev_final_overlap']} | {row['train_dev_overlap']} |"
        )
    lines += [
        "",
        f"- final_test union: {split['final_test_union']}",
        f"- final_test sum: {split['final_test_sum']}",
        "",
        "## Gold-Stripped Replay",
        "",
        f"- rows replayed: {replay['rows']}",
        f"- changed replayed: {replay['changed']}",
        f"- mismatches vs saved e90 outputs: {replay['mismatch_count']}",
        "",
        "## Exactness Check",
        "",
        f"- all rows old ideas exact ratio: {exact['all_old_ideas_exact_ratio']:.4f}",
        f"- all rows new ideas exact ratio: {exact['all_new_ideas_exact_ratio']:.4f}",
        f"- old overall abs error mean: {exact['overall_abs_error_old_mean']:.4f}",
        f"- new overall abs error mean: {exact['overall_abs_error_new_mean']:.4f}",
        "",
        "## Shuffled Donor Negative Control",
        "",
        f"- actual overall gain: {shuffle['actual_gain_overall']:+.6f}",
        f"- shuffled gain mean: {shuffle['shuffle_gain_overall_mean']:+.6f}",
        f"- shuffled gain std: {shuffle['shuffle_gain_overall_std']:.6f}",
        f"- shuffled gain min/max: {shuffle['shuffle_gain_overall_min']:+.6f} / {shuffle['shuffle_gain_overall_max']:+.6f}",
        f"- shuffled runs >= actual gain: {shuffle['shuffle_gain_ge_actual_count']}/{shuffle['n_shuffles']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    e90 = load_e90_module()
    manifest = read_json(MANIFEST_PATH)
    outer_summary = read_json(SUMMARY_PATH)
    payload = {
        "scope": "e90 sealed outer cleanliness audit; diagnostic only; no tuning",
        "paths": {
            "data_root": str(DATA_ROOT),
            "base_root": str(BASE_ROOT),
            "e90_root": str(E90_ROOT),
            "summary": str(SUMMARY_PATH),
            "manifest": str(MANIFEST_PATH),
        },
        "manifest_flags": {
            "outer_test_touched": manifest.get("outer_test_touched"),
            "selection_on_outer": manifest.get("selection_on_outer"),
            "selection_uses_gold": manifest.get("selection_uses_gold"),
            "policy": manifest.get("policy"),
        },
        "outer_result": {
            "base_overall": outer_summary["fold_level"]["base"]["overall"]["mean"],
            "candidate_overall": outer_summary["fold_level"]["candidate"]["overall"]["mean"],
            "gain_overall": outer_summary["fold_level"]["candidate_gain_vs_base"]["overall"]["mean"],
            "base_ideas": outer_summary["fold_level"]["base"]["ideas"]["mean"],
            "candidate_ideas": outer_summary["fold_level"]["candidate"]["ideas"]["mean"],
            "gain_ideas": outer_summary["fold_level"]["candidate_gain_vs_base"]["ideas"]["mean"],
        },
        "split_overlap": split_overlap_audit(),
        "gold_stripped_replay": gold_stripped_replay_audit(e90),
        "exactness": exactness_audit(),
        "shuffled_negative_control": shuffled_negative_control(e90, n_shuffles=50),
    }
    write_json(OUT_JSON, payload)
    write_markdown(OUT_MD, payload)
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    print("split_overlap_pass", payload["split_overlap"]["pass"])
    print("gold_stripped_replay_pass", payload["gold_stripped_replay"]["pass"])
    print("shuffle_negative_control_pass", payload["shuffled_negative_control"]["pass"])
    print("actual_gain", payload["shuffled_negative_control"]["actual_gain_overall"])
    print("shuffle_gain_mean", payload["shuffled_negative_control"]["shuffle_gain_overall_mean"])
    print("shuffle_gain_max", payload["shuffled_negative_control"]["shuffle_gain_overall_max"])


if __name__ == "__main__":
    main()
