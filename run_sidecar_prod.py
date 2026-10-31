#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt7 BERT sidecar 一键生产流程。

流程：
1) sidecar 推理（ideas/organization）
2) 从 pred_* 字段构建 baseline 预测
3) 分别按 gA/gE 做门控融合
4) 可选：style sidecar 融合
5) 可选：conventions 共识融合（双模型边界门控）
6) 可选：QWK 评估
7) 按回退规则自动选择最终输出

默认策略：
- 主门控：gA
- 回退门控：gE
- 最后回退：baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

from evaluate_qwk import qwk as compute_qwk
from prompt7_mapper import Prompt7Mapper


GATE_PRESETS = {
    "gA": {"max_prob": 0.72, "margin": 0.18, "min_delta": 1, "max_replace_ratio": 0.25},
    "gE": {"max_prob": 0.80, "margin": 0.24, "min_delta": 1, "max_replace_ratio": 0.20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", required=True, help="Input cache JSON.")
    parser.add_argument("--model-dir", default="sidecar_models/v3", help="Sidecar model dir (ideas/organization subdirs).")
    parser.add_argument("--output-dir", default="sidecar_preds/prod", help="Output directory.")
    parser.add_argument("--prefix", default="batch", help="Output file prefix.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python binary used to call helper scripts.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--ideas-ensemble-dirs",
        default="",
        help="Optional comma-separated ideas trait model dirs for logits ensemble.",
    )
    parser.add_argument(
        "--ideas-agreement-dirs",
        default="",
        help="Optional comma-separated ideas trait model dirs for agreement fallback inference.",
    )
    parser.add_argument(
        "--gpu",
        default="auto",
        help="GPU selector forwarded to infer_sidecar_bert.py: auto | cpu | <physical_gpu_id>.",
    )
    parser.add_argument("--auto-min-free-gb", type=float, default=8.0, help="Min free GPU memory for --gpu auto.")
    parser.add_argument("--auto-wait-seconds", type=float, default=600.0, help="Wait time for an idle GPU.")
    parser.add_argument("--gpu-lock-dir", default="/tmp/llms_aes_gpu_locks", help="Directory for GPU lock files.")

    parser.add_argument("--primary-gate", choices=("gA", "gE"), default="gA")
    parser.add_argument("--fallback-gate", choices=("gA", "gE"), default="gE")
    parser.add_argument(
        "--gate-selection-mode",
        choices=("auto", "force_primary"),
        default="auto",
        help="How to choose the IO gate result before style/conventions branches.",
    )
    parser.add_argument("--org-max-prob", type=float, default=-1.0, help="Optional organization-only gate override.")
    parser.add_argument("--org-margin", type=float, default=-1.0, help="Optional organization-only gate override.")
    parser.add_argument("--org-min-delta", type=int, default=-1, help="Optional organization-only gate override.")
    parser.add_argument(
        "--ideas-max-replace-ratio",
        type=float,
        default=-1.0,
        help="Optional ideas-only gate override.",
    )
    parser.add_argument(
        "--ideas-score-floor",
        type=float,
        default=None,
        help="Optional ideas-only score floor forwarded to merge_sidecar_gate.py.",
    )
    parser.add_argument(
        "--org-max-replace-ratio",
        type=float,
        default=-1.0,
        help="Optional organization-only gate override.",
    )
    parser.add_argument(
        "--org-directional-preset",
        choices=("none", "o1"),
        default="none",
        help="Optional organization directional gate preset passed to merge_sidecar_gate.py.",
    )
    parser.add_argument("--ideas-replace-limit", type=float, default=0.30, help="Fallback to fallback-gate if primary exceeds this.")
    parser.add_argument("--org-replace-limit", type=float, default=0.25, help="Fallback to fallback-gate if primary exceeds this.")
    parser.add_argument("--min-overall-advantage", type=float, default=0.01, help="Switch if fallback overall beats primary by >= this.")
    parser.add_argument(
        "--enable-io-trait-decoupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Search baseline/gA/gE combinations for ideas/org separately and pick the best feasible combo.",
    )
    parser.add_argument(
        "--io-decouple-min-overall-advantage",
        type=float,
        default=0.0,
        help="Apply the best ideas/org decoupled combo only if it beats the current selected IO source by at least this.",
    )

    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--selection-use-gold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow current input gold/QWK to drive gate and branch selection. Disable for frozen final test.",
    )
    parser.add_argument("--asap-data", default="data/training_set_rel3.tsv")
    parser.add_argument("--prompt-id", type=int, default=7)

    # 可选 style 分支（在选中的 gA/gE/baseline 结果上继续融合）。
    parser.add_argument(
        "--enable-style-sidecar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable style sidecar merge with confidence gate.",
    )
    parser.add_argument("--style-model-dir", default="", help="Style sidecar model dir.")
    parser.add_argument("--style-batch-size", type=int, default=32)
    parser.add_argument("--style-max-length", type=int, default=256)
    parser.add_argument("--style-max-prob", type=float, default=0.72)
    parser.add_argument("--style-margin", type=float, default=0.16)
    parser.add_argument("--style-min-delta", type=int, default=1)
    parser.add_argument("--style-max-replace-ratio", type=float, default=0.12)
    parser.add_argument("--style-allow-upward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--style-replace-limit",
        type=float,
        default=0.18,
        help="Rollback style branch if replacement ratio exceeds this.",
    )
    parser.add_argument(
        "--style-min-overall-advantage",
        type=float,
        default=0.0,
        help="Apply style branch only if overall gain >= this (when evaluate enabled).",
    )
    parser.add_argument(
        "--style-min-style-advantage",
        type=float,
        default=0.0,
        help="Apply style branch only if style trait gain >= this (when evaluate enabled).",
    )

    # 可选 conventions 共识分支（在选中的 gA/gE/baseline 结果上继续融合）。
    parser.add_argument(
        "--enable-conv-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable conventions consensus merge with two sidecar models.",
    )
    parser.add_argument("--conv-model-a-dir", default="", help="Conventions sidecar model A dir.")
    parser.add_argument("--conv-model-b-dir", default="", help="Conventions sidecar model B dir.")
    parser.add_argument("--conv-batch-size", type=int, default=32)
    parser.add_argument("--conv-max-length", type=int, default=256)
    parser.add_argument("--conv-max-prob", type=float, default=0.72)
    parser.add_argument("--conv-margin", type=float, default=0.16)
    parser.add_argument("--conv-min-delta", type=int, default=1)
    parser.add_argument("--conv-max-replace-ratio", type=float, default=0.08)
    parser.add_argument("--conv-allow-upward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--conv-boundary-mode",
        choices=("none", "23_down", "23_bi"),
        default="23_bi",
    )
    parser.add_argument(
        "--conv-replace-limit",
        type=float,
        default=0.12,
        help="Rollback conventions branch if replacement ratio exceeds this.",
    )
    parser.add_argument(
        "--conv-min-overall-advantage",
        type=float,
        default=0.0,
        help="Apply conventions branch only if overall gain >= this (when evaluate enabled).",
    )
    parser.add_argument(
        "--conv-min-conventions-advantage",
        type=float,
        default=0.0,
        help="Apply conventions branch only if conventions gain >= this (when evaluate enabled).",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_int(row: dict[str, Any], keys: tuple[str, ...], default: int | None = None) -> int | None:
    for key in keys:
        if key in row:
            try:
                return int(row[key])
            except Exception:
                continue
    return default


def build_baseline_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        i = int(row.get("pred_ideas", row.get("ideas", 0)))
        o = int(row.get("pred_organization", row.get("organization", 0)))
        s = int(row.get("pred_style", row.get("style", 0)))
        c = int(row.get("pred_conventions", row.get("conventions", 0)))
        merged = dict(row)
        merged["ideas"] = i
        merged["organization"] = o
        merged["style"] = s
        merged["conventions"] = c
        merged["overall"] = int(Prompt7Mapper.compute_overall_score(i, o, s, c))
        out.append(merged)
    return out


def evaluate_if_enabled(
    enabled: bool,
    python_bin: str,
    repo_root: Path,
    predictions: Path,
    output: Path,
    asap_data: str,
    prompt_id: int,
) -> bool:
    if not enabled:
        return False
    cmd = [
        python_bin,
        str(repo_root / "evaluate_qwk.py"),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
        "--asap-data",
        asap_data,
        "--prompt-id",
        str(prompt_id),
    ]
    try:
        run_cmd(cmd, cwd=repo_root)
        return True
    except subprocess.CalledProcessError:
        return False


def evaluate_rows_in_memory(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    required_gold = ("gold_ideas", "gold_organization", "gold_style", "gold_conventions", "gold_overall")
    if not all(all(key in row for key in required_gold) for row in rows):
        return None

    trait_qwk = {}
    for trait in ("ideas", "organization", "style", "conventions"):
        y_true = [int(row[f"gold_{trait}"]) for row in rows]
        y_pred = [int(get_int(row, (trait, f"pred_{trait}"), default=0) or 0) for row in rows]
        trait_qwk[trait] = float(compute_qwk(y_true, y_pred, 0, 3))

    y_true_overall = [int(row["gold_overall"]) for row in rows]
    y_pred_overall = [int(get_int(row, ("overall", "pred_overall"), default=0) or 0) for row in rows]
    overall_qwk = float(compute_qwk(y_true_overall, y_pred_overall, 0, 30))
    return {
        "trait_qwk": trait_qwk,
        "overall_qwk": overall_qwk,
        "num_samples": len(rows),
        "skipped_rows": 0,
    }


def build_decoupled_io_rows(
    baseline_rows: list[dict[str, Any]],
    ideas_rows: list[dict[str, Any]],
    organization_rows: list[dict[str, Any]],
    ideas_source: str,
    organization_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (len(baseline_rows) == len(ideas_rows) == len(organization_rows)):
        raise ValueError("Row counts mismatch for IO decoupling.")

    merged_rows: list[dict[str, Any]] = []
    ideas_used = 0
    org_used = 0

    for base_row, ideas_row, org_row in zip(baseline_rows, ideas_rows, organization_rows):
        base_ideas = int(get_int(base_row, ("ideas", "pred_ideas"), default=0) or 0)
        base_org = int(get_int(base_row, ("organization", "pred_organization"), default=0) or 0)
        style = int(get_int(base_row, ("style", "pred_style"), default=0) or 0)
        conventions = int(get_int(base_row, ("conventions", "pred_conventions"), default=0) or 0)

        ideas_val = get_int(ideas_row, ("ideas", "pred_ideas"), default=base_ideas)
        org_val = get_int(org_row, ("organization", "pred_organization"), default=base_org)
        ideas = int(base_ideas if ideas_val is None else ideas_val)
        organization = int(base_org if org_val is None else org_val)

        ideas_used += int(ideas != base_ideas)
        org_used += int(organization != base_org)

        out = dict(base_row)
        out["ideas"] = ideas
        out["organization"] = organization
        out["style"] = style
        out["conventions"] = conventions
        out["overall"] = int(Prompt7Mapper.compute_overall_score(ideas, organization, style, conventions))
        out["io_selected_ideas_source"] = ideas_source
        out["io_selected_organization_source"] = organization_source
        merged_rows.append(out)

    rows_n = max(1, len(baseline_rows))
    report = {
        "ideas_source": ideas_source,
        "organization_source": organization_source,
        "ideas_used_count": int(ideas_used),
        "organization_used_count": int(org_used),
        "ideas_used_ratio": float(ideas_used / rows_n),
        "organization_used_ratio": float(org_used / rows_n),
    }
    return merged_rows, report


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    input_path = Path(args.input_cache)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input cache not found: {input_path}")

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if str(args.ideas_ensemble_dirs).strip() and str(args.ideas_agreement_dirs).strip():
        raise ValueError("Use either --ideas-ensemble-dirs or --ideas-agreement-dirs, not both.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    with_bert = output_dir / f"{prefix}_with_bert.json"
    baseline_preds = output_dir / f"{prefix}_baseline_preds.json"
    baseline_qwk = output_dir / f"{prefix}_baseline_qwk.json"
    selected_preds = output_dir / f"{prefix}_selected_preds.json"
    selected_report = output_dir / f"{prefix}_selected_report.json"
    summary_path = output_dir / f"{prefix}_run_summary.json"
    selected_qwk = output_dir / f"{prefix}_selected_qwk.json"
    io_decoupled_preds = output_dir / f"{prefix}_io_decoupled_preds.json"
    io_decoupled_report = output_dir / f"{prefix}_io_decoupled_report.json"
    io_decoupled_qwk = output_dir / f"{prefix}_io_decoupled_qwk.json"

    gate_files: dict[str, dict[str, Path]] = {}
    for gate_name in ("gA", "gE"):
        gate_files[gate_name] = {
            "preds": output_dir / f"{prefix}_merged_preds_{gate_name}.json",
            "report": output_dir / f"{prefix}_report_{gate_name}.json",
            "qwk": output_dir / f"{prefix}_qwk_{gate_name}.json",
        }

    infer_gpu_args = [
        "--gpu",
        str(args.gpu),
        "--auto-min-free-gb",
        str(args.auto_min_free_gb),
        "--auto-wait-seconds",
        str(args.auto_wait_seconds),
        "--gpu-lock-dir",
        str(args.gpu_lock_dir),
    ]

    # 第 1 步：sidecar 推理输出。
    run_cmd(
        [
            args.python_bin,
            str(repo_root / "infer_sidecar_bert.py"),
            "--input-cache",
            str(input_path),
            "--model-dir",
            str(model_dir),
            "--output",
            str(with_bert),
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            *infer_gpu_args,
        ]
        + (
            ["--ideas-agreement-dirs", str(args.ideas_agreement_dirs)]
            if str(args.ideas_agreement_dirs).strip()
            else []
        )
        + (
            ["--ideas-ensemble-dirs", str(args.ideas_ensemble_dirs)]
            if str(args.ideas_ensemble_dirs).strip()
            else []
        ),
        cwd=repo_root,
    )

    # 第 2 步：构建 baseline 预测。
    rows = load_json(with_bert)
    baseline_rows = build_baseline_predictions(rows)
    save_json(baseline_preds, baseline_rows)

    baseline_eval_ok = evaluate_if_enabled(
        enabled=args.evaluate,
        python_bin=args.python_bin,
        repo_root=repo_root,
        predictions=baseline_preds,
        output=baseline_qwk,
        asap_data=args.asap_data,
        prompt_id=args.prompt_id,
    )

    # 第 3 步：分别按 gA/gE 融合。
    for gate_name in ("gA", "gE"):
        gate = GATE_PRESETS[gate_name]
        out = gate_files[gate_name]
        merge_cmd = [
            args.python_bin,
            str(repo_root / "merge_sidecar_gate.py"),
            "--input",
            str(with_bert),
            "--output",
            str(out["preds"]),
            "--report-output",
            str(out["report"]),
            "--max-prob",
            str(gate["max_prob"]),
            "--margin",
            str(gate["margin"]),
            "--min-delta",
            str(gate["min_delta"]),
            "--max-replace-ratio",
            str(gate["max_replace_ratio"]),
            "--ideas-max-replace-ratio",
            str(args.ideas_max_replace_ratio),
            "--org-max-prob",
            str(args.org_max_prob),
            "--org-margin",
            str(args.org_margin),
            "--org-min-delta",
            str(args.org_min_delta),
            "--org-max-replace-ratio",
            str(args.org_max_replace_ratio),
            "--org-directional-preset",
            str(args.org_directional_preset),
            "--no-distribution-guard",
        ]
        if args.ideas_score_floor is not None:
            merge_cmd.extend(["--ideas-score-floor", str(args.ideas_score_floor)])
        run_cmd(merge_cmd, cwd=repo_root)
        evaluate_if_enabled(
            enabled=args.evaluate,
            python_bin=args.python_bin,
            repo_root=repo_root,
            predictions=out["preds"],
            output=out["qwk"],
            asap_data=args.asap_data,
            prompt_id=args.prompt_id,
        )

    # 第 4 步：自动选择门控结果。
    selected_gate = args.primary_gate
    reasons: list[str] = []

    primary_report = load_json(gate_files[args.primary_gate]["report"])
    primary_ideas_ratio = float(primary_report["traits"]["ideas"]["final_used_ratio"])
    primary_org_ratio = float(primary_report["traits"]["organization"]["final_used_ratio"])

    if args.gate_selection_mode == "force_primary":
        reasons.append(f"gate_selection_mode=force_primary -> selected_gate={args.primary_gate}")
    else:
        if primary_ideas_ratio > args.ideas_replace_limit:
            reasons.append(
                f"{args.primary_gate}.ideas_replace_ratio={primary_ideas_ratio:.4f} > limit={args.ideas_replace_limit:.4f}"
            )
            selected_gate = args.fallback_gate
        if primary_org_ratio > args.org_replace_limit:
            reasons.append(
                f"{args.primary_gate}.org_replace_ratio={primary_org_ratio:.4f} > limit={args.org_replace_limit:.4f}"
            )
            selected_gate = args.fallback_gate

        if args.evaluate and args.selection_use_gold:
            primary_qwk_path = gate_files[args.primary_gate]["qwk"]
            fallback_qwk_path = gate_files[args.fallback_gate]["qwk"]
            if primary_qwk_path.exists() and fallback_qwk_path.exists():
                primary_overall = float(load_json(primary_qwk_path)["overall_qwk"])
                fallback_overall = float(load_json(fallback_qwk_path)["overall_qwk"])
                if (fallback_overall - primary_overall) >= args.min_overall_advantage:
                    selected_gate = args.fallback_gate
                    reasons.append(
                        f"{args.fallback_gate}.overall={fallback_overall:.4f} beats "
                        f"{args.primary_gate}.overall={primary_overall:.4f} by >= {args.min_overall_advantage:.4f}"
                    )

                if baseline_eval_ok and baseline_qwk.exists():
                    baseline_overall = float(load_json(baseline_qwk)["overall_qwk"])
                    if (baseline_overall - max(primary_overall, fallback_overall)) >= args.min_overall_advantage:
                        selected_gate = "baseline"
                        reasons.append(
                            f"baseline.overall={baseline_overall:.4f} beats sidecar by >= {args.min_overall_advantage:.4f}"
                        )

    selected_io_rows = baseline_rows if selected_gate == "baseline" else load_json(gate_files[selected_gate]["preds"])
    source_pred = baseline_preds if selected_gate == "baseline" else gate_files[selected_gate]["preds"]
    source_report: Path | None = None if selected_gate == "baseline" else gate_files[selected_gate]["report"]
    source_qwk: Path | None = baseline_qwk if selected_gate == "baseline" and baseline_qwk.exists() else None
    if selected_gate != "baseline" and gate_files[selected_gate]["qwk"].exists():
        source_qwk = gate_files[selected_gate]["qwk"]
    selected_io_source = {
        "ideas": "baseline" if selected_gate == "baseline" else selected_gate,
        "organization": "baseline" if selected_gate == "baseline" else selected_gate,
    }

    io_decoupling_summary: dict[str, Any] | None = None
    if args.enable_io_trait_decoupling:
        io_reasons: list[str] = []
        io_decoupling_summary = {
            "enabled": True,
            "applied": False,
            "reasons": io_reasons,
        }
        if not args.evaluate:
            io_reasons.append("evaluate_required_for_io_trait_decoupling")
        elif not args.selection_use_gold:
            io_reasons.append("selection_use_gold_disabled_for_io_trait_decoupling")
        else:
            selected_io_qwk_payload = evaluate_rows_in_memory(selected_io_rows)
            if selected_io_qwk_payload is None:
                io_reasons.append("missing_gold_labels_for_io_trait_decoupling")
            else:
                source_rows = {
                    "baseline": baseline_rows,
                    "gA": load_json(gate_files["gA"]["preds"]),
                    "gE": load_json(gate_files["gE"]["preds"]),
                }
                candidate_rows_by_key: dict[str, list[dict[str, Any]]] = {}
                candidates: list[dict[str, Any]] = []

                for ideas_source, org_source in product(("baseline", "gA", "gE"), repeat=2):
                    merged_rows, candidate_meta = build_decoupled_io_rows(
                        baseline_rows=baseline_rows,
                        ideas_rows=source_rows[ideas_source],
                        organization_rows=source_rows[org_source],
                        ideas_source=ideas_source,
                        organization_source=org_source,
                    )
                    qwk_payload = evaluate_rows_in_memory(merged_rows)
                    if qwk_payload is None:
                        continue
                    candidate_key = f"{ideas_source}__{org_source}"
                    candidate_rows_by_key[candidate_key] = merged_rows
                    candidates.append(
                        {
                            "candidate_key": candidate_key,
                            "ideas_source": ideas_source,
                            "organization_source": org_source,
                            "ideas_replace_ratio": candidate_meta["ideas_used_ratio"],
                            "organization_replace_ratio": candidate_meta["organization_used_ratio"],
                            "trait_qwk": qwk_payload["trait_qwk"],
                            "overall_qwk": qwk_payload["overall_qwk"],
                            "valid_by_limits": (
                                candidate_meta["ideas_used_ratio"] <= float(args.ideas_replace_limit)
                                and candidate_meta["organization_used_ratio"] <= float(args.org_replace_limit)
                            ),
                        }
                    )

                candidates.sort(
                    key=lambda item: (
                        float(item["valid_by_limits"]),
                        float(item["overall_qwk"]),
                    ),
                    reverse=True,
                )

                best_candidate = next((item for item in candidates if bool(item["valid_by_limits"])), None)
                current_overall = float(selected_io_qwk_payload["overall_qwk"])
                io_decoupling_summary["current_selection"] = {
                    "ideas_source": selected_io_source["ideas"],
                    "organization_source": selected_io_source["organization"],
                    "overall_qwk": current_overall,
                }
                io_decoupling_summary["candidates"] = candidates

                if best_candidate is None:
                    io_reasons.append("no_feasible_io_decoupled_candidate")
                else:
                    same_as_current = (
                        best_candidate["ideas_source"] == selected_io_source["ideas"]
                        and best_candidate["organization_source"] == selected_io_source["organization"]
                    )
                    if same_as_current:
                        io_reasons.append("best_io_decoupled_candidate_matches_current_selection")
                    elif (float(best_candidate["overall_qwk"]) - current_overall) < float(args.io_decouple_min_overall_advantage):
                        io_reasons.append(
                            f"io_decoupled_overall_gain={float(best_candidate['overall_qwk'])-current_overall:.4f} "
                            f"< min={args.io_decouple_min_overall_advantage:.4f}"
                        )
                    else:
                        selected_gate = "io_decoupled"
                        selected_io_source = {
                            "ideas": str(best_candidate["ideas_source"]),
                            "organization": str(best_candidate["organization_source"]),
                        }
                        selected_io_rows = candidate_rows_by_key[str(best_candidate["candidate_key"])]
                        save_json(io_decoupled_preds, selected_io_rows)
                        io_qwk_payload = evaluate_rows_in_memory(selected_io_rows)
                        if io_qwk_payload is None:
                            raise ValueError("Failed to evaluate selected IO decoupled candidate.")
                        io_qwk_payload["prompt_id"] = int(args.prompt_id)
                        io_qwk_payload["predictions_path"] = str(io_decoupled_preds)
                        save_json(io_decoupled_qwk, io_qwk_payload)
                        report_payload = {
                            "selected_candidate": best_candidate,
                            "current_selection": io_decoupling_summary["current_selection"],
                            "limits": {
                                "ideas_replace_limit": float(args.ideas_replace_limit),
                                "org_replace_limit": float(args.org_replace_limit),
                                "io_decouple_min_overall_advantage": float(args.io_decouple_min_overall_advantage),
                            },
                            "candidates": candidates,
                        }
                        save_json(io_decoupled_report, report_payload)
                        source_pred = io_decoupled_preds
                        source_report = io_decoupled_report
                        source_qwk = io_decoupled_qwk
                        io_decoupling_summary["applied"] = True
                        io_decoupling_summary["selected_candidate"] = best_candidate
                        reasons.append(
                            f"io_decoupled ideas={best_candidate['ideas_source']} org={best_candidate['organization_source']} "
                            f"overall={best_candidate['overall_qwk']:.4f}"
                        )
    else:
        io_decoupling_summary = {"enabled": False}

    save_json(selected_preds, selected_io_rows)
    if source_report and source_report.exists():
        shutil.copy2(source_report, selected_report)
    if source_qwk and Path(source_qwk).exists():
        shutil.copy2(source_qwk, selected_qwk)

    style_summary: dict[str, Any] | None = None
    style_applied = False
    style_reasons: list[str] = []

    # 第 5 步：可选 style sidecar 分支。
    if args.enable_style_sidecar:
        style_model_dir = Path(args.style_model_dir)
        if not style_model_dir.is_absolute():
            style_model_dir = repo_root / style_model_dir

        if not style_model_dir.exists():
            style_reasons.append("missing_style_model_dir")
        else:
            style_pred = output_dir / f"{prefix}_style_with_bert.json"
            style_merged = output_dir / f"{prefix}_style_merged_preds.json"
            style_report = output_dir / f"{prefix}_style_report.json"
            style_qwk = output_dir / f"{prefix}_style_qwk.json"

            run_cmd(
                [
                    args.python_bin,
                    str(repo_root / "infer_sidecar_bert.py"),
                    "--input-cache",
                    str(selected_preds),
                    "--model-dir",
                    str(style_model_dir),
                    "--traits",
                    "style",
                    "--output",
                    str(style_pred),
                    "--batch-size",
                    str(args.style_batch_size),
                    "--max-length",
                    str(args.style_max_length),
                    *infer_gpu_args,
                ],
                cwd=repo_root,
            )

            merge_cmd = [
                args.python_bin,
                str(repo_root / "merge_style_sidecar.py"),
                "--base-input",
                str(selected_preds),
                "--model-input",
                str(style_pred),
                "--output",
                str(style_merged),
                "--report-output",
                str(style_report),
                "--max-prob",
                str(args.style_max_prob),
                "--margin",
                str(args.style_margin),
                "--min-delta",
                str(args.style_min_delta),
                "--max-replace-ratio",
                str(args.style_max_replace_ratio),
                "--no-distribution-guard",
            ]
            merge_cmd.append("--allow-upward" if args.style_allow_upward else "--no-allow-upward")
            run_cmd(merge_cmd, cwd=repo_root)

            rep = load_json(style_report)
            rep_ratio = float(rep.get("final_used_ratio", 0.0))
            if rep_ratio > float(args.style_replace_limit):
                style_reasons.append(
                    f"style_replace_ratio={rep_ratio:.4f} > style_replace_limit={args.style_replace_limit:.4f}"
                )

            style_eval_ok = evaluate_if_enabled(
                enabled=args.evaluate,
                python_bin=args.python_bin,
                repo_root=repo_root,
                predictions=style_merged,
                output=style_qwk,
                asap_data=args.asap_data,
                prompt_id=args.prompt_id,
            )
            if args.evaluate and args.selection_use_gold and style_eval_ok and selected_qwk.exists():
                base_sel_qwk = load_json(selected_qwk)
                style_sel_qwk = load_json(style_qwk)
                base_overall = float(base_sel_qwk["overall_qwk"])
                style_overall = float(style_sel_qwk["overall_qwk"])
                base_style = float(base_sel_qwk["trait_qwk"]["style"])
                style_style = float(style_sel_qwk["trait_qwk"]["style"])
                if (style_overall - base_overall) < float(args.style_min_overall_advantage):
                    style_reasons.append(
                        f"style_overall_gain={style_overall-base_overall:.4f} < min={args.style_min_overall_advantage:.4f}"
                    )
                if (style_style - base_style) < float(args.style_min_style_advantage):
                    style_reasons.append(
                        f"style_trait_gain={style_style-base_style:.4f} < min={args.style_min_style_advantage:.4f}"
                    )

            style_applied = len(style_reasons) == 0
            if style_applied:
                shutil.copy2(style_merged, selected_preds)
                shutil.copy2(style_report, selected_report)
                if style_qwk.exists():
                    shutil.copy2(style_qwk, selected_qwk)

            style_summary = {
                "enabled": True,
                "applied": bool(style_applied),
                "reasons": style_reasons,
                "config": {
                    "max_prob": float(args.style_max_prob),
                    "margin": float(args.style_margin),
                    "min_delta": int(args.style_min_delta),
                    "max_replace_ratio": float(args.style_max_replace_ratio),
                    "allow_upward": bool(args.style_allow_upward),
                    "distribution_guard": False,
                    "style_replace_limit": float(args.style_replace_limit),
                    "style_min_overall_advantage": float(args.style_min_overall_advantage),
                    "style_min_style_advantage": float(args.style_min_style_advantage),
                },
                "artifacts": {
                    "style_model_dir": str(style_model_dir),
                    "style_with_bert": str(style_pred),
                    "style_merged_preds": str(style_merged),
                    "style_report": str(style_report),
                    "style_qwk": str(style_qwk) if style_qwk.exists() else "",
                },
            }
    else:
        style_summary = {"enabled": False}

    conventions_summary: dict[str, Any] | None = None
    conv_applied = False
    conv_reasons: list[str] = []

    # 第 6 步：可选 conventions 共识分支。
    if args.enable_conv_consensus:
        conv_model_a = Path(args.conv_model_a_dir)
        conv_model_b = Path(args.conv_model_b_dir)
        if not conv_model_a.is_absolute():
            conv_model_a = repo_root / conv_model_a
        if not conv_model_b.is_absolute():
            conv_model_b = repo_root / conv_model_b

        if not conv_model_a.exists() or not conv_model_b.exists():
            conv_reasons.append("missing_conventions_model_dir")
        else:
            conv_a_pred = output_dir / f"{prefix}_conv_a_with_bert.json"
            conv_b_pred = output_dir / f"{prefix}_conv_b_with_bert.json"
            conv_merged = output_dir / f"{prefix}_conv_merged_preds.json"
            conv_report = output_dir / f"{prefix}_conv_report.json"
            conv_qwk = output_dir / f"{prefix}_conv_qwk.json"

            run_cmd(
                [
                    args.python_bin,
                    str(repo_root / "infer_sidecar_bert.py"),
                    "--input-cache",
                    str(selected_preds),
                    "--model-dir",
                    str(conv_model_a),
                    "--traits",
                    "conventions",
                    "--output",
                    str(conv_a_pred),
                    "--batch-size",
                    str(args.conv_batch_size),
                    "--max-length",
                    str(args.conv_max_length),
                    *infer_gpu_args,
                ],
                cwd=repo_root,
            )
            run_cmd(
                [
                    args.python_bin,
                    str(repo_root / "infer_sidecar_bert.py"),
                    "--input-cache",
                    str(selected_preds),
                    "--model-dir",
                    str(conv_model_b),
                    "--traits",
                    "conventions",
                    "--output",
                    str(conv_b_pred),
                    "--batch-size",
                    str(args.conv_batch_size),
                    "--max-length",
                    str(args.conv_max_length),
                    *infer_gpu_args,
                ],
                cwd=repo_root,
            )

            merge_cmd = [
                args.python_bin,
                str(repo_root / "merge_conventions_consensus.py"),
                "--base-input",
                str(selected_preds),
                "--model-a-input",
                str(conv_a_pred),
                "--model-b-input",
                str(conv_b_pred),
                "--output",
                str(conv_merged),
                "--report-output",
                str(conv_report),
                "--max-prob",
                str(args.conv_max_prob),
                "--margin",
                str(args.conv_margin),
                "--min-delta",
                str(args.conv_min_delta),
                "--max-replace-ratio",
                str(args.conv_max_replace_ratio),
                "--boundary-mode",
                str(args.conv_boundary_mode),
                "--no-distribution-guard",
            ]
            merge_cmd.append("--allow-upward" if args.conv_allow_upward else "--no-allow-upward")
            run_cmd(merge_cmd, cwd=repo_root)

            rep = load_json(conv_report)
            rep_ratio = float(rep.get("final_used_ratio", 0.0))
            if rep_ratio > float(args.conv_replace_limit):
                conv_reasons.append(
                    f"conv_replace_ratio={rep_ratio:.4f} > conv_replace_limit={args.conv_replace_limit:.4f}"
                )

            conv_eval_ok = evaluate_if_enabled(
                enabled=args.evaluate,
                python_bin=args.python_bin,
                repo_root=repo_root,
                predictions=conv_merged,
                output=conv_qwk,
                asap_data=args.asap_data,
                prompt_id=args.prompt_id,
            )
            if args.evaluate and args.selection_use_gold and conv_eval_ok and selected_qwk.exists():
                base_sel_qwk = load_json(selected_qwk)
                conv_sel_qwk = load_json(conv_qwk)
                base_overall = float(base_sel_qwk["overall_qwk"])
                conv_overall = float(conv_sel_qwk["overall_qwk"])
                base_conv = float(base_sel_qwk["trait_qwk"]["conventions"])
                conv_conv = float(conv_sel_qwk["trait_qwk"]["conventions"])
                if (conv_overall - base_overall) < float(args.conv_min_overall_advantage):
                    conv_reasons.append(
                        f"conv_overall_gain={conv_overall-base_overall:.4f} < min={args.conv_min_overall_advantage:.4f}"
                    )
                if (conv_conv - base_conv) < float(args.conv_min_conventions_advantage):
                    conv_reasons.append(
                        f"conv_trait_gain={conv_conv-base_conv:.4f} < min={args.conv_min_conventions_advantage:.4f}"
                    )

            conv_applied = len(conv_reasons) == 0
            if conv_applied:
                shutil.copy2(conv_merged, selected_preds)
                shutil.copy2(conv_report, selected_report)
                if conv_qwk.exists():
                    shutil.copy2(conv_qwk, selected_qwk)

            conventions_summary = {
                "enabled": True,
                "applied": bool(conv_applied),
                "reasons": conv_reasons,
                "config": {
                    "max_prob": float(args.conv_max_prob),
                    "margin": float(args.conv_margin),
                    "min_delta": int(args.conv_min_delta),
                    "max_replace_ratio": float(args.conv_max_replace_ratio),
                    "allow_upward": bool(args.conv_allow_upward),
                    "distribution_guard": False,
                    "boundary_mode": str(args.conv_boundary_mode),
                    "conv_replace_limit": float(args.conv_replace_limit),
                    "conv_min_overall_advantage": float(args.conv_min_overall_advantage),
                    "conv_min_conventions_advantage": float(args.conv_min_conventions_advantage),
                },
                "artifacts": {
                    "conv_model_a_dir": str(conv_model_a),
                    "conv_model_b_dir": str(conv_model_b),
                    "conv_a_pred": str(conv_a_pred),
                    "conv_b_pred": str(conv_b_pred),
                    "conv_merged_preds": str(conv_merged),
                    "conv_report": str(conv_report),
                    "conv_qwk": str(conv_qwk) if conv_qwk.exists() else "",
                },
            }
    else:
        conventions_summary = {"enabled": False}

    summary = {
        "input_cache": str(input_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "python_bin": args.python_bin,
        "gpu": {
            "requested": str(args.gpu),
            "auto_min_free_gb": float(args.auto_min_free_gb),
            "auto_wait_seconds": float(args.auto_wait_seconds),
            "gpu_lock_dir": str(args.gpu_lock_dir),
        },
        "gate_presets": GATE_PRESETS,
        "evaluate": bool(args.evaluate),
        "selection_use_gold": bool(args.selection_use_gold),
        "primary_gate": args.primary_gate,
        "fallback_gate": args.fallback_gate,
        "gate_selection_mode": str(args.gate_selection_mode),
        "ideas_agreement_dirs": [x.strip() for x in str(args.ideas_agreement_dirs).split(",") if x.strip()],
        "ideas_ensemble_dirs": [x.strip() for x in str(args.ideas_ensemble_dirs).split(",") if x.strip()],
        "org_gate_overrides": {
            "org_max_prob": float(args.org_max_prob) if args.org_max_prob >= 0 else None,
            "org_margin": float(args.org_margin) if args.org_margin >= 0 else None,
            "org_min_delta": int(args.org_min_delta) if args.org_min_delta >= 0 else None,
            "org_max_replace_ratio": float(args.org_max_replace_ratio) if args.org_max_replace_ratio >= 0 else None,
        },
        "org_directional_preset": str(args.org_directional_preset),
        "limits": {
            "ideas_replace_limit": float(args.ideas_replace_limit),
            "org_replace_limit": float(args.org_replace_limit),
            "min_overall_advantage": float(args.min_overall_advantage),
            "ideas_max_replace_ratio_override": float(args.ideas_max_replace_ratio)
            if args.ideas_max_replace_ratio >= 0
            else None,
        },
        "io_distribution_guard": False,
        "io_trait_selection": selected_io_source,
        "io_trait_decoupling": io_decoupling_summary,
        "selected_gate": selected_gate,
        "selection_reasons": reasons,
        "style_sidecar": style_summary,
        "conventions_consensus": conventions_summary,
        "artifacts": {
            "with_bert": str(with_bert),
            "baseline_preds": str(baseline_preds),
            "baseline_qwk": str(baseline_qwk) if baseline_qwk.exists() else "",
            "gA_preds": str(gate_files["gA"]["preds"]),
            "gA_report": str(gate_files["gA"]["report"]),
            "gA_qwk": str(gate_files["gA"]["qwk"]) if gate_files["gA"]["qwk"].exists() else "",
            "gE_preds": str(gate_files["gE"]["preds"]),
            "gE_report": str(gate_files["gE"]["report"]),
            "gE_qwk": str(gate_files["gE"]["qwk"]) if gate_files["gE"]["qwk"].exists() else "",
            "io_decoupled_preds": str(io_decoupled_preds) if io_decoupled_preds.exists() else "",
            "io_decoupled_report": str(io_decoupled_report) if io_decoupled_report.exists() else "",
            "io_decoupled_qwk": str(io_decoupled_qwk) if io_decoupled_qwk.exists() else "",
            "selected_preds": str(selected_preds),
            "selected_report": str(selected_report) if selected_report.exists() else "",
            "selected_qwk": str(selected_qwk) if selected_qwk.exists() else "",
            "selected_source_preds": str(source_pred),
            "selected_source_report": str(source_report) if source_report else "",
        },
    }
    save_json(summary_path, summary)

    print(f"selected_gate={selected_gate}")
    print(f"saved: {selected_preds}")
    if selected_report.exists():
        print(f"saved: {selected_report}")
    print(f"summary_saved: {summary_path}")


if __name__ == "__main__":
    main()
