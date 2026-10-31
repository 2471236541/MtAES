#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from run_sidecar_prod import GATE_PRESETS, build_baseline_predictions, evaluate_if_enabled, load_json, save_json, run_cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay mainline pipeline from existing artifacts with a stricter ideas gate.")
    p.add_argument("--source-dir", required=True, help="Directory containing existing source artifacts for one cell.")
    p.add_argument("--source-prefix", required=True, help="Existing source prefix, e.g. v10s53_s0_seed20260325_holdout.")
    p.add_argument("--output-dir", required=True, help="Output directory for replayed artifacts.")
    p.add_argument("--output-prefix", required=True, help="Output prefix for replayed artifacts.")
    p.add_argument("--ideas-margin-threshold", type=float, required=True)
    p.add_argument("--ideas-score-floor", type=float, default=None)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--selection-use-gold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow current input gold/QWK to drive gate and branch selection. Disable for deployable replay.",
    )
    p.add_argument("--asap-data", default="data/training_set_rel3.tsv")
    p.add_argument("--prompt-id", type=int, default=7)
    p.add_argument("--primary-gate", choices=("gA", "gE"), default="gA")
    p.add_argument("--fallback-gate", choices=("gA", "gE"), default="gE")
    p.add_argument("--ideas-replace-limit", type=float, default=0.30)
    p.add_argument("--ideas-max-replace-ratio", type=float, default=-1.0)
    p.add_argument("--org-replace-limit", type=float, default=0.25)
    p.add_argument("--min-overall-advantage", type=float, default=0.0)
    p.add_argument("--org-max-prob", type=float, default=0.60)
    p.add_argument("--org-margin", type=float, default=0.10)
    p.add_argument("--org-min-delta", type=int, default=-1)
    p.add_argument("--org-max-replace-ratio", type=float, default=0.24)
    p.add_argument("--org-directional-preset", choices=("none", "o1"), default="none")
    p.add_argument("--enable-style-sidecar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--style-max-prob", type=float, default=0.75)
    p.add_argument("--style-margin", type=float, default=0.20)
    p.add_argument("--style-min-delta", type=int, default=1)
    p.add_argument("--style-max-replace-ratio", type=float, default=0.10)
    p.add_argument("--style-replace-limit", type=float, default=0.12)
    p.add_argument("--style-min-overall-advantage", type=float, default=0.0)
    p.add_argument("--style-min-style-advantage", type=float, default=0.0)
    p.add_argument("--style-allow-upward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-conv-consensus", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--conv-max-prob", type=float, default=0.72)
    p.add_argument("--conv-margin", type=float, default=0.16)
    p.add_argument("--conv-min-delta", type=int, default=1)
    p.add_argument("--conv-max-replace-ratio", type=float, default=0.08)
    p.add_argument("--conv-boundary-mode", choices=("none", "23_down", "23_bi"), default="23_down")
    p.add_argument("--conv-replace-limit", type=float, default=0.12)
    p.add_argument("--conv-min-overall-advantage", type=float, default=0.0)
    p.add_argument("--conv-min-conventions-advantage", type=float, default=0.0)
    p.add_argument("--conv-allow-upward", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def copy_if_exists(src: Path | None, dst: Path) -> None:
    if src and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_minimal_selected_report(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_prefix = args.source_prefix
    out_prefix = args.output_prefix

    source_with_bert = source_dir / f"{src_prefix}_with_bert.json"
    source_style_pred = source_dir / f"{src_prefix}_style_with_bert.json"
    source_conv_a_pred = source_dir / f"{src_prefix}_conv_a_with_bert.json"
    source_conv_b_pred = source_dir / f"{src_prefix}_conv_b_with_bert.json"
    ensure_exists(source_with_bert)

    with_bert = output_dir / f"{out_prefix}_with_bert.json"
    baseline_preds = output_dir / f"{out_prefix}_baseline_preds.json"
    baseline_qwk = output_dir / f"{out_prefix}_baseline_qwk.json"
    selected_preds = output_dir / f"{out_prefix}_selected_preds.json"
    selected_report = output_dir / f"{out_prefix}_selected_report.json"
    selected_qwk = output_dir / f"{out_prefix}_selected_qwk.json"
    summary_path = output_dir / f"{out_prefix}_run_summary.json"

    shutil.copy2(source_with_bert, with_bert)
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

    gate_files: dict[str, dict[str, Path]] = {}
    for gate_name in ("gA", "gE"):
        gate_files[gate_name] = {
            "preds": output_dir / f"{out_prefix}_merged_preds_{gate_name}.json",
            "report": output_dir / f"{out_prefix}_report_{gate_name}.json",
            "qwk": output_dir / f"{out_prefix}_qwk_{gate_name}.json",
        }

    for gate_name in ("gA", "gE"):
        gate = GATE_PRESETS[gate_name]
        out = gate_files[gate_name]
        gate_cmd = [
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
            "--ideas-margin-threshold",
            str(args.ideas_margin_threshold),
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
            gate_cmd.extend(["--ideas-score-floor", str(args.ideas_score_floor)])

        run_cmd(gate_cmd, cwd=repo_root)
        evaluate_if_enabled(
            enabled=args.evaluate,
            python_bin=args.python_bin,
            repo_root=repo_root,
            predictions=out["preds"],
            output=out["qwk"],
            asap_data=args.asap_data,
            prompt_id=args.prompt_id,
        )

    selected_gate = args.primary_gate
    selection_reasons: list[str] = []

    primary_report = load_json(gate_files[args.primary_gate]["report"])
    primary_ideas_ratio = float(primary_report["traits"]["ideas"]["final_used_ratio"])
    primary_org_ratio = float(primary_report["traits"]["organization"]["final_used_ratio"])

    if primary_ideas_ratio > args.ideas_replace_limit:
        selection_reasons.append(
            f"{args.primary_gate}.ideas_replace_ratio={primary_ideas_ratio:.4f} > limit={args.ideas_replace_limit:.4f}"
        )
        selected_gate = args.fallback_gate
    if primary_org_ratio > args.org_replace_limit:
        selection_reasons.append(
            f"{args.primary_gate}.org_replace_ratio={primary_org_ratio:.4f} > limit={args.org_replace_limit:.4f}"
        )
        selected_gate = args.fallback_gate

    if args.evaluate and args.selection_use_gold:
        primary_overall = float(load_json(gate_files[args.primary_gate]["qwk"])["overall_qwk"])
        fallback_overall = float(load_json(gate_files[args.fallback_gate]["qwk"])["overall_qwk"])
        if (fallback_overall - primary_overall) >= args.min_overall_advantage:
            selected_gate = args.fallback_gate
            selection_reasons.append(
                f"{args.fallback_gate}.overall={fallback_overall:.4f} beats "
                f"{args.primary_gate}.overall={primary_overall:.4f} by >= {args.min_overall_advantage:.4f}"
            )
        if baseline_eval_ok and baseline_qwk.exists():
            baseline_overall = float(load_json(baseline_qwk)["overall_qwk"])
            if (baseline_overall - max(primary_overall, fallback_overall)) >= args.min_overall_advantage:
                selected_gate = "baseline"
                selection_reasons.append(
                    f"baseline.overall={baseline_overall:.4f} beats sidecar by >= {args.min_overall_advantage:.4f}"
                )

    if selected_gate == "baseline":
        selected_rows = baseline_rows
        save_json(selected_preds, selected_rows)
        write_minimal_selected_report(
            selected_report,
            {
                "selected_gate": "baseline",
                "selection_reasons": selection_reasons,
                "ideas_margin_threshold": float(args.ideas_margin_threshold),
            },
        )
        copy_if_exists(baseline_qwk if baseline_qwk.exists() else None, selected_qwk)
    else:
        shutil.copy2(gate_files[selected_gate]["preds"], selected_preds)
        shutil.copy2(gate_files[selected_gate]["report"], selected_report)
        copy_if_exists(gate_files[selected_gate]["qwk"], selected_qwk)

    style_summary: dict[str, Any] = {"enabled": bool(args.enable_style_sidecar), "applied": False, "reasons": []}
    if args.enable_style_sidecar and source_style_pred.exists():
        style_merged = output_dir / f"{out_prefix}_style_merged_preds.json"
        style_report = output_dir / f"{out_prefix}_style_report.json"
        style_qwk = output_dir / f"{out_prefix}_style_qwk.json"
        shutil.copy2(source_style_pred, output_dir / f"{out_prefix}_style_with_bert.json")
        run_cmd(
            [
                args.python_bin,
                str(repo_root / "merge_style_sidecar.py"),
                "--base-input",
                str(selected_preds),
                "--model-input",
                str(source_style_pred),
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
                "--allow-upward" if args.style_allow_upward else "--no-allow-upward",
            ],
            cwd=repo_root,
        )
        rep = load_json(style_report)
        rep_ratio = float(rep.get("final_used_ratio", 0.0))
        if rep_ratio > float(args.style_replace_limit):
            style_summary["reasons"].append(
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
                style_summary["reasons"].append(
                    f"style_overall_gain={style_overall-base_overall:.4f} < min={args.style_min_overall_advantage:.4f}"
                )
            if (style_style - base_style) < float(args.style_min_style_advantage):
                style_summary["reasons"].append(
                    f"style_trait_gain={style_style-base_style:.4f} < min={args.style_min_style_advantage:.4f}"
                )
        style_summary["applied"] = len(style_summary["reasons"]) == 0
        if style_summary["applied"]:
            shutil.copy2(style_merged, selected_preds)
            shutil.copy2(style_report, selected_report)
            copy_if_exists(style_qwk if style_qwk.exists() else None, selected_qwk)

    conventions_summary: dict[str, Any] = {"enabled": bool(args.enable_conv_consensus), "applied": False, "reasons": []}
    if args.enable_conv_consensus and source_conv_a_pred.exists() and source_conv_b_pred.exists():
        conv_merged = output_dir / f"{out_prefix}_conv_merged_preds.json"
        conv_report = output_dir / f"{out_prefix}_conv_report.json"
        conv_qwk = output_dir / f"{out_prefix}_conv_qwk.json"
        shutil.copy2(source_conv_a_pred, output_dir / f"{out_prefix}_conv_a_with_bert.json")
        shutil.copy2(source_conv_b_pred, output_dir / f"{out_prefix}_conv_b_with_bert.json")
        run_cmd(
            [
                args.python_bin,
                str(repo_root / "merge_conventions_consensus.py"),
                "--base-input",
                str(selected_preds),
                "--model-a-input",
                str(source_conv_a_pred),
                "--model-b-input",
                str(source_conv_b_pred),
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
                "--allow-upward" if args.conv_allow_upward else "--no-allow-upward",
            ],
            cwd=repo_root,
        )
        rep = load_json(conv_report)
        rep_ratio = float(rep.get("final_used_ratio", 0.0))
        if rep_ratio > float(args.conv_replace_limit):
            conventions_summary["reasons"].append(
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
                conventions_summary["reasons"].append(
                    f"conv_overall_gain={conv_overall-base_overall:.4f} < min={args.conv_min_overall_advantage:.4f}"
                )
            if (conv_conv - base_conv) < float(args.conv_min_conventions_advantage):
                conventions_summary["reasons"].append(
                    f"conv_trait_gain={conv_conv-base_conv:.4f} < min={args.conv_min_conventions_advantage:.4f}"
                )
        conventions_summary["applied"] = len(conventions_summary["reasons"]) == 0
        if conventions_summary["applied"]:
            shutil.copy2(conv_merged, selected_preds)
            shutil.copy2(conv_report, selected_report)
            copy_if_exists(conv_qwk if conv_qwk.exists() else None, selected_qwk)

    summary = {
        "source_dir": str(source_dir),
        "source_prefix": src_prefix,
        "output_dir": str(output_dir),
        "output_prefix": out_prefix,
        "ideas_margin_threshold": float(args.ideas_margin_threshold),
        "ideas_score_floor": float(args.ideas_score_floor) if args.ideas_score_floor is not None else None,
        "ideas_replace_limit": float(args.ideas_replace_limit),
        "ideas_max_replace_ratio_override": float(args.ideas_max_replace_ratio)
        if args.ideas_max_replace_ratio >= 0
        else None,
        "org_replace_limit": float(args.org_replace_limit),
        "min_overall_advantage": float(args.min_overall_advantage),
        "selection_use_gold": bool(args.selection_use_gold),
        "primary_gate": args.primary_gate,
        "fallback_gate": args.fallback_gate,
        "selected_gate": selected_gate,
        "selection_reasons": selection_reasons,
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
            "selected_preds": str(selected_preds),
            "selected_report": str(selected_report) if selected_report.exists() else "",
            "selected_qwk": str(selected_qwk) if selected_qwk.exists() else "",
        },
    }
    save_json(summary_path, summary)
    print(f"summary_saved: {summary_path}")


if __name__ == "__main__":
    main()
