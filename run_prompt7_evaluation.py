#!/usr/bin/env python3
# -*- coding: utf-8 -*-  test
"""
Prompt 7 批量评估与 QWK 计算脚本。
对全部 Prompt 7 作文打分，并计算与人工评分的一致性（QWK）。
使用固定预定义问题，不做动态生成。
"""

from trait_analysis_pipeline import TraitAnalysisPipeline
from prompt7_mapper import Prompt7Mapper
from qwk_evaluator import QWKEvaluator
import pandas as pd
import json
import os
import sys
import argparse
import subprocess
import shutil
from collections import Counter
import torch
import torch.distributed as dist
from datetime import datetime


def compute_dual_objective(qwk_results, trait_weight=0.6, overall_weight=0.4):
    trait_qwk = qwk_results.get("trait_qwk", {})
    trait_values = list(trait_qwk.values())
    mean_trait_qwk = (sum(trait_values) / len(trait_values)) if trait_values else 0.0
    overall_qwk = float(qwk_results.get("overall_qwk", 0.0))
    objective = (trait_weight * mean_trait_qwk) + (overall_weight * overall_qwk)
    return mean_trait_qwk, objective


def build_distribution_dict(items, keys):
    dist = {}
    for key in keys:
        counter = Counter(int(x[key]) for x in items if key in x)
        dist[key] = {str(score): int(counter[score]) for score in sorted(counter)}
    return dist


def _annotate_gold_fields(rows, gold_map):
    for item in rows:
        gold = gold_map.get(item.get("essay_id"))
        if not gold:
            continue
        item["gold_ideas"] = int(gold["ideas"])
        item["gold_organization"] = int(gold["organization"])
        item["gold_style"] = int(gold["style"])
        item["gold_conventions"] = int(gold["conventions"])
        item["gold_overall"] = int(gold["overall"])


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_essay_id_list(raw):
    if not raw:
        return []
    out = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def build_prompt7_detailed_rows(
    predictions,
    gold_labels,
    prompt7_df,
    threshold_cache,
    sidecar_cache,
    include_essay_text=True,
    include_reasoning=True,
):
    gold_map = {int(x["essay_id"]): x for x in gold_labels}
    essay_text_map = {
        int(eid): str(text)
        for eid, text in zip(prompt7_df["essay_id"].tolist(), prompt7_df["essay"].tolist())
    }

    cache_map = {}
    for row in sidecar_cache:
        eid = _to_int_or_none(row.get("essay_id"))
        if eid is not None:
            cache_map[eid] = dict(row)
    for row in threshold_cache:
        eid = _to_int_or_none(row.get("essay_id"))
        if eid is not None:
            cache_map[eid] = dict(row)

    detailed = []
    for pred in predictions:
        eid = int(pred["essay_id"])
        gold = gold_map.get(eid, {})
        cache = cache_map.get(eid, {})

        pred_i = int(pred["ideas"])
        pred_o = int(pred["organization"])
        pred_s = int(pred["style"])
        pred_c = int(pred["conventions"])
        pred_overall = int(pred["overall"])

        base_i = _to_int_or_none(pred.get("baseline_ideas"))
        if base_i is None:
            base_i = _to_int_or_none(cache.get("pred_ideas"))
        if base_i is None:
            base_i = pred_i

        base_o = _to_int_or_none(pred.get("baseline_organization"))
        if base_o is None:
            base_o = _to_int_or_none(cache.get("pred_organization"))
        if base_o is None:
            base_o = pred_o

        base_s = _to_int_or_none(cache.get("pred_style"))
        if base_s is None:
            base_s = pred_s

        base_c = _to_int_or_none(cache.get("pred_conventions"))
        if base_c is None:
            base_c = pred_c

        base_overall = int(Prompt7Mapper.compute_overall_score(base_i, base_o, base_s, base_c))

        gold_i = _to_int_or_none(gold.get("ideas"))
        gold_o = _to_int_or_none(gold.get("organization"))
        gold_s = _to_int_or_none(gold.get("style"))
        gold_c = _to_int_or_none(gold.get("conventions"))
        gold_overall = _to_int_or_none(gold.get("overall"))

        sidecar_used_ideas = pred.get("sidecar_used_ideas")
        if sidecar_used_ideas is None:
            sidecar_used_ideas = bool(pred_i != base_i)
        sidecar_used_org = pred.get("sidecar_used_organization")
        if sidecar_used_org is None:
            sidecar_used_org = bool(pred_o != base_o)

        item = {
            "essay_id": eid,
            "prediction": {
                "ideas": pred_i,
                "organization": pred_o,
                "style": pred_s,
                "conventions": pred_c,
                "overall": pred_overall,
            },
            "gold": {
                "ideas": gold_i,
                "organization": gold_o,
                "style": gold_s,
                "conventions": gold_c,
                "overall": gold_overall,
            },
            "baseline": {
                "ideas": base_i,
                "organization": base_o,
                "style": base_s,
                "conventions": base_c,
                "overall": base_overall,
            },
            "delta_vs_gold": {
                "ideas": (pred_i - gold_i) if gold_i is not None else None,
                "organization": (pred_o - gold_o) if gold_o is not None else None,
                "style": (pred_s - gold_s) if gold_s is not None else None,
                "conventions": (pred_c - gold_c) if gold_c is not None else None,
                "overall": (pred_overall - gold_overall) if gold_overall is not None else None,
            },
            "delta_vs_baseline": {
                "ideas": pred_i - base_i,
                "organization": pred_o - base_o,
                "style": pred_s - base_s,
                "conventions": pred_c - base_c,
                "overall": pred_overall - base_overall,
            },
            "sidecar": {
                "used_ideas": bool(sidecar_used_ideas),
                "used_organization": bool(sidecar_used_org),
                "bert_ideas_pred": _to_int_or_none(pred.get("bert_ideas_pred")),
                "bert_ideas_max_prob": _to_float_or_none(pred.get("bert_ideas_max_prob")),
                "bert_ideas_margin": _to_float_or_none(pred.get("bert_ideas_margin")),
                "bert_organization_pred": _to_int_or_none(pred.get("bert_organization_pred")),
                "bert_organization_max_prob": _to_float_or_none(pred.get("bert_organization_max_prob")),
                "bert_organization_margin": _to_float_or_none(pred.get("bert_organization_margin")),
            },
        }

        if include_essay_text:
            item["essay_text"] = str(cache.get("essay_text") or essay_text_map.get(eid, ""))

        if include_reasoning:
            item["reasoning"] = {
                "ideas": str(cache.get("ideas_reasoning", "")),
                "organization": str(cache.get("organization_reasoning", "")),
                "style": str(cache.get("style_reasoning", "")),
                "conventions": str(cache.get("conventions_reasoning", "")),
            }

        detailed.append(item)

    detailed.sort(key=lambda x: x["essay_id"])
    return detailed


def maybe_apply_sidecar_predictions(args, repo_root, data_path, predictions, sidecar_cache):
    if not args.enable_sidecar:
        return predictions, None
    if not sidecar_cache:
        print("WARNING: sidecar enabled but sidecar cache is empty; fallback to baseline predictions.")
        return predictions, {"enabled": True, "applied": False, "reason": "empty_sidecar_cache"}

    sidecar_output_dir = (
        args.sidecar_output_dir
        if os.path.isabs(args.sidecar_output_dir)
        else os.path.join(repo_root, args.sidecar_output_dir)
    )
    os.makedirs(sidecar_output_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"prompt7_eval_{run_tag}"
    input_cache_path = os.path.join(sidecar_output_dir, f"{prefix}_input_cache.json")

    with open(input_cache_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_cache, f, indent=2, ensure_ascii=False)

    model_dir = (
        args.sidecar_model_dir
        if os.path.isabs(args.sidecar_model_dir)
        else os.path.join(repo_root, args.sidecar_model_dir)
    )
    python_bin = args.sidecar_python_bin or sys.executable
    cmd = [
        python_bin,
        os.path.join(repo_root, "run_sidecar_prod.py"),
        "--input-cache",
        input_cache_path,
        "--model-dir",
        model_dir,
        "--output-dir",
        sidecar_output_dir,
        "--prefix",
        prefix,
        "--python-bin",
        python_bin,
        "--batch-size",
        str(args.sidecar_batch_size),
        "--max-length",
        str(args.sidecar_max_length),
        "--primary-gate",
        args.sidecar_primary_gate,
        "--fallback-gate",
        args.sidecar_fallback_gate,
        "--ideas-replace-limit",
        str(args.sidecar_ideas_replace_limit),
        "--org-replace-limit",
        str(args.sidecar_org_replace_limit),
        "--min-overall-advantage",
        str(args.sidecar_min_overall_advantage),
        "--style-model-dir",
        (
            args.sidecar_style_model_dir
            if os.path.isabs(args.sidecar_style_model_dir)
            else os.path.join(repo_root, args.sidecar_style_model_dir)
        ),
        "--style-batch-size",
        str(args.sidecar_style_batch_size),
        "--style-max-length",
        str(args.sidecar_style_max_length),
        "--style-max-prob",
        str(args.sidecar_style_max_prob),
        "--style-margin",
        str(args.sidecar_style_margin),
        "--style-min-delta",
        str(args.sidecar_style_min_delta),
        "--style-max-replace-ratio",
        str(args.sidecar_style_max_replace_ratio),
        "--style-replace-limit",
        str(args.sidecar_style_replace_limit),
        "--style-min-overall-advantage",
        str(args.sidecar_style_min_overall_advantage),
        "--style-min-style-advantage",
        str(args.sidecar_style_min_style_advantage),
        "--conv-model-a-dir",
        (
            args.sidecar_conv_model_a_dir
            if os.path.isabs(args.sidecar_conv_model_a_dir)
            else os.path.join(repo_root, args.sidecar_conv_model_a_dir)
        ),
        "--conv-model-b-dir",
        (
            args.sidecar_conv_model_b_dir
            if os.path.isabs(args.sidecar_conv_model_b_dir)
            else os.path.join(repo_root, args.sidecar_conv_model_b_dir)
        ),
        "--conv-batch-size",
        str(args.sidecar_conv_batch_size),
        "--conv-max-length",
        str(args.sidecar_conv_max_length),
        "--conv-max-prob",
        str(args.sidecar_conv_max_prob),
        "--conv-margin",
        str(args.sidecar_conv_margin),
        "--conv-min-delta",
        str(args.sidecar_conv_min_delta),
        "--conv-max-replace-ratio",
        str(args.sidecar_conv_max_replace_ratio),
        "--conv-boundary-mode",
        str(args.sidecar_conv_boundary_mode),
        "--conv-replace-limit",
        str(args.sidecar_conv_replace_limit),
        "--conv-min-overall-advantage",
        str(args.sidecar_conv_min_overall_advantage),
        "--conv-min-conventions-advantage",
        str(args.sidecar_conv_min_conventions_advantage),
        "--asap-data",
        data_path,
        "--prompt-id",
        "7",
    ]
    if args.sidecar_evaluate:
        cmd.append("--evaluate")
    else:
        cmd.append("--no-evaluate")
    if args.sidecar_enable_style_sidecar:
        cmd.append("--enable-style-sidecar")
    else:
        cmd.append("--no-enable-style-sidecar")
    if args.sidecar_style_allow_upward:
        cmd.append("--style-allow-upward")
    else:
        cmd.append("--no-style-allow-upward")
    if args.sidecar_enable_conv_consensus:
        cmd.append("--enable-conv-consensus")
    else:
        cmd.append("--no-enable-conv-consensus")
    if args.sidecar_conv_allow_upward:
        cmd.append("--conv-allow-upward")
    else:
        cmd.append("--no-conv-allow-upward")

    try:
        print("\n[Sidecar] Running V3 sidecar gating pipeline...")
        subprocess.run(cmd, check=True)
        summary_path = os.path.join(sidecar_output_dir, f"{prefix}_run_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            sidecar_summary = json.load(f)
        selected_preds_path = sidecar_summary["artifacts"]["selected_preds"]
        with open(selected_preds_path, "r", encoding="utf-8") as f:
            selected_rows = json.load(f)

        final_predictions = []
        for row in selected_rows:
            final_predictions.append(
                {
                    "essay_id": int(row["essay_id"]),
                    "ideas": int(row["ideas"]),
                    "organization": int(row["organization"]),
                    "style": int(row["style"]),
                    "conventions": int(row["conventions"]),
                    "overall": int(row["overall"]),
                    "shadow_ideas": row.get("shadow_ideas"),
                    "shadow_organization": row.get("shadow_organization"),
                    "baseline_ideas": _to_int_or_none(row.get("baseline_ideas")),
                    "baseline_organization": _to_int_or_none(row.get("baseline_organization")),
                    "sidecar_used_ideas": bool(row.get("sidecar_used_ideas", False)),
                    "sidecar_used_organization": bool(row.get("sidecar_used_organization", False)),
                    "bert_ideas_pred": _to_int_or_none(row.get("bert_ideas_pred")),
                    "bert_ideas_max_prob": _to_float_or_none(row.get("bert_ideas_max_prob")),
                    "bert_ideas_margin": _to_float_or_none(row.get("bert_ideas_margin")),
                    "bert_organization_pred": _to_int_or_none(row.get("bert_organization_pred")),
                    "bert_organization_max_prob": _to_float_or_none(row.get("bert_organization_max_prob")),
                    "bert_organization_margin": _to_float_or_none(row.get("bert_organization_margin")),
                }
            )
        final_predictions.sort(key=lambda x: x["essay_id"])
        if len(final_predictions) != len(predictions):
            print(
                "WARNING: sidecar output count != baseline count "
                f"({len(final_predictions)} vs {len(predictions)}). Fallback to baseline."
            )
            return predictions, {
                "enabled": True,
                "applied": False,
                "reason": "count_mismatch",
                "sidecar_summary_path": summary_path,
            }
        return final_predictions, {
            "enabled": True,
            "applied": True,
            "summary_path": summary_path,
            "selected_gate": sidecar_summary.get("selected_gate", ""),
            "artifacts": sidecar_summary.get("artifacts", {}),
            "selection_reasons": sidecar_summary.get("selection_reasons", []),
            "limits": sidecar_summary.get("limits", {}),
        }
    except Exception as e:
        print(f"WARNING: sidecar pipeline failed ({e}); fallback to baseline predictions.")
        return predictions, {"enabled": True, "applied": False, "reason": str(e)}


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return distributed, rank, world_size, local_rank


def detect_available_gpus(min_free_gb=20.0, max_gpus=8):
    if not torch.cuda.is_available():
        return []

    min_free_mb = int(min_free_gb * 1024)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True
        )
    except Exception:
        return []

    candidates = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(parts[1])
        except ValueError:
            continue
        if free_mb >= min_free_mb:
            candidates.append((idx, free_mb))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates[:max_gpus]]


def maybe_auto_launch(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.auto_launch or world_size > 1:
        return

    gpu_ids = detect_available_gpus(min_free_gb=args.min_free_gb, max_gpus=args.max_gpus)
    if not gpu_ids:
        print(
            f"Auto-launch found no GPU with free memory >= {args.min_free_gb} GB. "
            "Run with --no-auto-launch for manual mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    visible = ",".join(str(x) for x in gpu_ids)
    nproc = len(gpu_ids)
    # 稳健解析 torchrun：
    # 1) 当前 Python 可执行文件同目录
    # 2) PATH 查找
    # 3) 回退到 python -m torch.distributed.run
    py_bin = os.path.abspath(sys.executable)
    py_dir = os.path.dirname(py_bin)
    sibling_torchrun = os.path.join(py_dir, "torchrun")
    path_torchrun = shutil.which("torchrun")

    if os.path.isfile(sibling_torchrun) and os.access(sibling_torchrun, os.X_OK):
        launcher = [sibling_torchrun]
    elif path_torchrun:
        launcher = [path_torchrun]
    else:
        launcher = [py_bin, "-m", "torch.distributed.run"]

    cmd = launcher + [
        "--standalone",
        f"--nproc_per_node={nproc}",
        os.path.abspath(__file__),
        "--no-auto-launch",
        "--min-free-gb",
        str(args.min_free_gb),
        "--max-gpus",
        str(args.max_gpus),
        "--debug-first-n",
        str(args.debug_first_n),
        "--trait-weight",
        str(args.trait_weight),
        "--overall-weight",
        str(args.overall_weight),
    ]
    if args.max_essays is not None:
        cmd.extend(["--max-essays", str(args.max_essays)])
    if args.save_threshold_cache:
        cmd.extend(["--save-threshold-cache", str(args.save_threshold_cache)])
    if args.save_detailed_report:
        cmd.extend(["--save-detailed-report", str(args.save_detailed_report)])
    if args.detailed_report_essay_ids:
        cmd.extend(["--detailed-report-essay-ids", str(args.detailed_report_essay_ids)])
    if args.enable_ideas_org_shadow:
        cmd.append("--enable-ideas-org-shadow")
    if args.enable_ideas_org_structured_main:
        cmd.append("--enable-ideas-org-structured-main")
    if args.enable_sidecar:
        cmd.append("--enable-sidecar")
    else:
        cmd.append("--no-enable-sidecar")
    if args.sidecar_evaluate:
        cmd.append("--sidecar-evaluate")
    else:
        cmd.append("--no-sidecar-evaluate")
    if args.sidecar_enable_conv_consensus:
        cmd.append("--sidecar-enable-conv-consensus")
    else:
        cmd.append("--no-sidecar-enable-conv-consensus")
    if args.sidecar_enable_style_sidecar:
        cmd.append("--sidecar-enable-style-sidecar")
    else:
        cmd.append("--no-sidecar-enable-style-sidecar")
    if args.sidecar_style_allow_upward:
        cmd.append("--sidecar-style-allow-upward")
    else:
        cmd.append("--no-sidecar-style-allow-upward")
    if args.sidecar_conv_allow_upward:
        cmd.append("--sidecar-conv-allow-upward")
    else:
        cmd.append("--no-sidecar-conv-allow-upward")
    if args.detailed_report_include_essay_text:
        cmd.append("--detailed-report-include-essay-text")
    else:
        cmd.append("--no-detailed-report-include-essay-text")
    if args.detailed_report_include_reasoning:
        cmd.append("--detailed-report-include-reasoning")
    else:
        cmd.append("--no-detailed-report-include-reasoning")

    cmd.extend(
        [
            "--sidecar-model-dir",
            str(args.sidecar_model_dir),
            "--sidecar-output-dir",
            str(args.sidecar_output_dir),
            "--sidecar-python-bin",
            str(args.sidecar_python_bin),
            "--sidecar-primary-gate",
            str(args.sidecar_primary_gate),
            "--sidecar-fallback-gate",
            str(args.sidecar_fallback_gate),
            "--sidecar-ideas-replace-limit",
            str(args.sidecar_ideas_replace_limit),
            "--sidecar-org-replace-limit",
            str(args.sidecar_org_replace_limit),
            "--sidecar-min-overall-advantage",
            str(args.sidecar_min_overall_advantage),
            "--sidecar-style-model-dir",
            str(args.sidecar_style_model_dir),
            "--sidecar-style-batch-size",
            str(args.sidecar_style_batch_size),
            "--sidecar-style-max-length",
            str(args.sidecar_style_max_length),
            "--sidecar-style-max-prob",
            str(args.sidecar_style_max_prob),
            "--sidecar-style-margin",
            str(args.sidecar_style_margin),
            "--sidecar-style-min-delta",
            str(args.sidecar_style_min_delta),
            "--sidecar-style-max-replace-ratio",
            str(args.sidecar_style_max_replace_ratio),
            "--sidecar-style-replace-limit",
            str(args.sidecar_style_replace_limit),
            "--sidecar-style-min-overall-advantage",
            str(args.sidecar_style_min_overall_advantage),
            "--sidecar-style-min-style-advantage",
            str(args.sidecar_style_min_style_advantage),
            "--sidecar-batch-size",
            str(args.sidecar_batch_size),
            "--sidecar-max-length",
            str(args.sidecar_max_length),
            "--sidecar-conv-model-a-dir",
            str(args.sidecar_conv_model_a_dir),
            "--sidecar-conv-model-b-dir",
            str(args.sidecar_conv_model_b_dir),
            "--sidecar-conv-batch-size",
            str(args.sidecar_conv_batch_size),
            "--sidecar-conv-max-length",
            str(args.sidecar_conv_max_length),
            "--sidecar-conv-max-prob",
            str(args.sidecar_conv_max_prob),
            "--sidecar-conv-margin",
            str(args.sidecar_conv_margin),
            "--sidecar-conv-min-delta",
            str(args.sidecar_conv_min_delta),
            "--sidecar-conv-max-replace-ratio",
            str(args.sidecar_conv_max_replace_ratio),
            "--sidecar-conv-boundary-mode",
            str(args.sidecar_conv_boundary_mode),
            "--sidecar-conv-replace-limit",
            str(args.sidecar_conv_replace_limit),
            "--sidecar-conv-min-overall-advantage",
            str(args.sidecar_conv_min_overall_advantage),
            "--sidecar-conv-min-conventions-advantage",
            str(args.sidecar_conv_min_conventions_advantage),
            "--detailed-report-top-k",
            str(args.detailed_report_top_k),
            "--detailed-report-sort-by",
            str(args.detailed_report_sort_by),
        ]
    )

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = visible
    child_env.setdefault("OMP_NUM_THREADS", "1")

    print(
        f"Auto-launch using GPUs [{visible}] with nproc={nproc}; "
        f"launcher={' '.join(launcher)}"
    )
    os.execvpe(cmd[0], cmd, child_env)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--max-essays", type=int, default=None)
    parser.add_argument("--debug-first-n", type=int, default=5)
    parser.add_argument("--trait-weight", type=float, default=0.6)
    parser.add_argument("--overall-weight", type=float, default=0.4)
    parser.add_argument(
        "--enable-ideas-org-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable shadow-mode side scoring for ideas/organization (does not affect main scoring).",
    )
    parser.add_argument(
        "--enable-ideas-org-structured-main",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use structured banded scoring for ideas/organization in main scoring "
            "(A/B switch; default keeps baseline yes/no behavior)."
        ),
    )
    parser.add_argument(
        "--print-aggregate-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print Mean Trait QWK and Dual Objective in console output.",
    )
    parser.add_argument(
        "--save-threshold-cache",
        type=str,
        default="",
        help="Optional path to save style/conventions reasoning cache for offline threshold search.",
    )
    parser.add_argument(
        "--enable-sidecar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sidecar (ideas/org) merge pipeline via run_sidecar_prod.py.",
    )
    parser.add_argument(
        "--sidecar-model-dir",
        type=str,
        default="sidecar_models/v3",
        help="Sidecar model dir containing ideas/organization subfolders.",
    )
    parser.add_argument(
        "--sidecar-output-dir",
        type=str,
        default="sidecar_preds/prod",
        help="Output directory for sidecar artifacts.",
    )
    parser.add_argument(
        "--sidecar-python-bin",
        type=str,
        default=sys.executable,
        help="Python binary used to call sidecar scripts.",
    )
    parser.add_argument(
        "--sidecar-primary-gate",
        type=str,
        choices=("gA", "gE"),
        default="gA",
        help="Primary gate preset.",
    )
    parser.add_argument(
        "--sidecar-fallback-gate",
        type=str,
        choices=("gA", "gE"),
        default="gE",
        help="Fallback gate preset.",
    )
    parser.add_argument(
        "--sidecar-ideas-replace-limit",
        type=float,
        default=0.30,
        help="Fallback if primary ideas replace ratio exceeds this.",
    )
    parser.add_argument(
        "--sidecar-org-replace-limit",
        type=float,
        default=0.25,
        help="Fallback if primary organization replace ratio exceeds this.",
    )
    parser.add_argument(
        "--sidecar-min-overall-advantage",
        type=float,
        default=0.01,
        help="Fallback if fallback gate overall beats primary by at least this.",
    )
    parser.add_argument(
        "--sidecar-enable-style-sidecar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable style sidecar branch in run_sidecar_prod.py.",
    )
    parser.add_argument(
        "--sidecar-style-model-dir",
        type=str,
        default="sidecar_models/style_v1",
        help="Style sidecar model dir.",
    )
    parser.add_argument("--sidecar-style-batch-size", type=int, default=32)
    parser.add_argument("--sidecar-style-max-length", type=int, default=256)
    parser.add_argument("--sidecar-style-max-prob", type=float, default=0.72)
    parser.add_argument("--sidecar-style-margin", type=float, default=0.16)
    parser.add_argument("--sidecar-style-min-delta", type=int, default=1)
    parser.add_argument("--sidecar-style-max-replace-ratio", type=float, default=0.12)
    parser.add_argument(
        "--sidecar-style-allow-upward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sidecar-style-replace-limit", type=float, default=0.18)
    parser.add_argument("--sidecar-style-min-overall-advantage", type=float, default=0.0)
    parser.add_argument("--sidecar-style-min-style-advantage", type=float, default=0.0)
    parser.add_argument(
        "--sidecar-evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether sidecar runner evaluates baseline/gates with QWK for selection.",
    )
    parser.add_argument(
        "--sidecar-batch-size",
        type=int,
        default=32,
        help="Batch size for sidecar inference.",
    )
    parser.add_argument(
        "--sidecar-max-length",
        type=int,
        default=256,
        help="Max token length for sidecar inference.",
    )
    parser.add_argument(
        "--sidecar-enable-conv-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable conventions consensus branch in run_sidecar_prod.py.",
    )
    parser.add_argument(
        "--sidecar-conv-model-a-dir",
        type=str,
        default="sidecar_models/v6_conv_a",
        help="Conventions model A dir.",
    )
    parser.add_argument(
        "--sidecar-conv-model-b-dir",
        type=str,
        default="sidecar_models/v6_conv_b",
        help="Conventions model B dir.",
    )
    parser.add_argument("--sidecar-conv-batch-size", type=int, default=32)
    parser.add_argument("--sidecar-conv-max-length", type=int, default=256)
    parser.add_argument("--sidecar-conv-max-prob", type=float, default=0.72)
    parser.add_argument("--sidecar-conv-margin", type=float, default=0.16)
    parser.add_argument("--sidecar-conv-min-delta", type=int, default=1)
    parser.add_argument("--sidecar-conv-max-replace-ratio", type=float, default=0.08)
    parser.add_argument(
        "--sidecar-conv-allow-upward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sidecar-conv-boundary-mode",
        type=str,
        choices=("none", "23_down", "23_bi"),
        default="23_bi",
    )
    parser.add_argument("--sidecar-conv-replace-limit", type=float, default=0.12)
    parser.add_argument("--sidecar-conv-min-overall-advantage", type=float, default=0.0)
    parser.add_argument("--sidecar-conv-min-conventions-advantage", type=float, default=0.0)
    parser.add_argument(
        "--save-detailed-report",
        type=str,
        default="",
        help="Optional path to save per-essay detailed score report (essay text, reasoning, pred/gold deltas).",
    )
    parser.add_argument(
        "--detailed-report-top-k",
        type=int,
        default=30,
        help="When no essay id filter is set, keep top-K rows after sorting. <=0 means keep all.",
    )
    parser.add_argument(
        "--detailed-report-sort-by",
        type=str,
        choices=("abs_overall_delta_vs_gold", "overall_delta_vs_gold", "essay_id"),
        default="abs_overall_delta_vs_gold",
        help="Sort key for detailed report when essay id filter is not set.",
    )
    parser.add_argument(
        "--detailed-report-essay-ids",
        type=str,
        default="",
        help="Optional comma-separated essay IDs to export explicitly (order preserved).",
    )
    parser.add_argument(
        "--detailed-report-include-essay-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--detailed-report-include-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    maybe_auto_launch(args)
    distributed, rank, world_size, local_rank = init_distributed()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(repo_root, "data", "training_set_rel3.tsv")
    model_path = os.path.join(repo_root, "models", "Qwen-7B-Chat")

    if rank == 0:
        print("="*80)
        print("Prompt 7 Baseline Reasoning Evaluation with QWK")
        print("="*80)
    
    # 加载数据
    if rank == 0:
        print("\n[1/5] Loading ASAP dataset...")
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    
    # 过滤 Prompt 7
    prompt7_df = df[df['essay_set'] == 7].copy()
    if rank == 0:
        print(f"✓ Loaded {len(prompt7_df)} Prompt 7 essays")
    
    # 初始化评分流水线
    if rank == 0:
        print("\n[2/5] Initializing scoring pipeline...")
        print(f"Using data_path: {data_path}")
        print(f"Using model_path: {model_path}")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    pipeline = TraitAnalysisPipeline(model_path=model_path, device=device)
    Prompt7Mapper.ENABLE_IDEAS_ORG_STRUCTURED_MAIN = bool(args.enable_ideas_org_structured_main)
    if rank == 0:
        print("✓ Pipeline initialized")
        mapper = pipeline.mapper
        print(
            "✓ Calibration layer: "
            f"enabled={mapper.ENABLE_TRAIT_CALIBRATION}, "
            f"style_map={mapper.STYLE_CALIBRATION_MAP}, "
            f"conventions_map={mapper.CONVENTIONS_CALIBRATION_MAP}"
        )
        print(
            "✓ Ideas/Org main mode: "
            f"structured={mapper.ENABLE_IDEAS_ORG_STRUCTURED_MAIN}"
        )
    
    # 作文评分
    if rank == 0:
        print("\n[3/5] Scoring essays...")
        print("Note: Baseline minimal pipeline (fixed questions -> LLM reasoning -> rule-based scores)")
    predictions = []
    threshold_cache = []
    sidecar_cache = []
    
    # 测试时可限制前 N 篇（全量评估时不限制）
    max_essays = args.max_essays
    essays_to_score = prompt7_df.head(max_essays) if max_essays else prompt7_df
    local_essays = essays_to_score.iloc[rank::world_size]
    
    for idx, row in local_essays.iterrows():
        essay_id = row['essay_id']
        essay_text = row['essay']
        
        print(f"\n[rank {rank}] Scoring essay {essay_id} ({len(predictions)+1}/{len(local_essays)})...")
        
        try:
            debug_enabled = (rank == 0 and len(predictions) < args.debug_first_n)
            result = pipeline.score_prompt7_essay(
                text=essay_text,
                essay_id=int(essay_id),
                debug=debug_enabled,
                enable_ideas_org_shadow=args.enable_ideas_org_shadow,
            )
            shadow_ideas = result.get("shadow", {}).get("ideas", {}).get("score")
            shadow_organization = result.get("shadow", {}).get("organization", {}).get("score")
            
            predictions.append({
                "essay_id": int(essay_id),
                "ideas": result["ideas"],
                "organization": result["organization"],
                "style": result["style"],
                "conventions": result["conventions"],
                "overall": result["overall"],
                "shadow_ideas": int(shadow_ideas) if shadow_ideas is not None else None,
                "shadow_organization": int(shadow_organization) if shadow_organization is not None else None,
            })
            cache_item = {
                "essay_id": int(essay_id),
                "essay_text": essay_text,
                "pred_ideas": int(result["ideas"]),
                "pred_organization": int(result["organization"]),
                "pred_style": int(result["style"]),
                "pred_conventions": int(result["conventions"]),
                "ideas_reasoning": result["reasoning"]["ideas"]["raw_reasoning"],
                "organization_reasoning": result["reasoning"]["organization"]["raw_reasoning"],
                "style_reasoning": result["reasoning"]["style"]["raw_reasoning"],
                "conventions_reasoning": result["reasoning"]["conventions"]["raw_reasoning"],
                "shadow_ideas_reasoning": result.get("shadow", {}).get("ideas", {}).get("raw_reasoning", ""),
                "shadow_organization_reasoning": result.get("shadow", {}).get("organization", {}).get("raw_reasoning", ""),
                "shadow_ideas_score": int(shadow_ideas) if shadow_ideas is not None else None,
                "shadow_organization_score": int(shadow_organization) if shadow_organization is not None else None,
                "shadow_ideas": int(shadow_ideas) if shadow_ideas is not None else None,
                "shadow_organization": int(shadow_organization) if shadow_organization is not None else None,
            }
            if args.save_threshold_cache:
                threshold_cache.append(dict(cache_item))
            if args.enable_sidecar:
                sidecar_cache.append(dict(cache_item))
            
            print(f"✓ Essay {essay_id}: Overall={result['overall']}/30")
            
        except Exception as e:
            print(f"✗ Error scoring essay {essay_id}: {e}")
            continue
    
    if distributed:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, predictions)
        gathered_cache = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_cache, threshold_cache)
        gathered_sidecar = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_sidecar, sidecar_cache)
        if rank == 0:
            predictions = [item for sublist in gathered for item in sublist]
            predictions.sort(key=lambda x: x["essay_id"])
            threshold_cache = [item for sublist in gathered_cache for item in sublist]
            threshold_cache.sort(key=lambda x: x["essay_id"])
            sidecar_cache = [item for sublist in gathered_sidecar for item in sublist]
            sidecar_cache.sort(key=lambda x: x["essay_id"])
        else:
            predictions = []
            threshold_cache = []
            sidecar_cache = []

    if rank != 0:
        if distributed:
            dist.destroy_process_group()
        return

    sidecar_run_info = None
    if args.enable_sidecar:
        predictions, sidecar_run_info = maybe_apply_sidecar_predictions(
            args=args,
            repo_root=repo_root,
            data_path=data_path,
            predictions=predictions,
            sidecar_cache=sidecar_cache,
        )

    print(f"\n✓ Scored {len(predictions)} essays")
    if sidecar_run_info is not None:
        print(
            "[Sidecar] "
            f"enabled={sidecar_run_info.get('enabled', False)} "
            f"applied={sidecar_run_info.get('applied', False)} "
            f"selected_gate={sidecar_run_info.get('selected_gate', '')}"
        )
        if sidecar_run_info.get("summary_path"):
            print(f"[Sidecar] summary: {sidecar_run_info['summary_path']}")
        if sidecar_run_info.get("selection_reasons"):
            print(f"[Sidecar] reasons: {sidecar_run_info['selection_reasons']}")

    print("\nPrediction distribution (collapse check):")
    trait_keys = ("ideas", "organization", "style", "conventions")
    prediction_distribution = build_distribution_dict(predictions, trait_keys + ("overall",))
    for trait in trait_keys:
        dist_counter = Counter(int(p[trait]) for p in predictions)
        sorted_items = sorted(dist_counter.items())
        print(f"  {trait}: {sorted_items}")
        max_bucket = max(dist_counter.values()) if dist_counter else 0
        if predictions and max_bucket / len(predictions) >= 0.8:
            print(
                f"  WARNING: {trait} has dominant bucket "
                f"{max_bucket}/{len(predictions)} ({max_bucket/len(predictions):.1%})"
            )

    # 文风维度专项防坍缩保护
    style_counter = Counter(p["style"] for p in predictions)
    if predictions and style_counter:
        style_max = max(style_counter.values())
        style_ratio = style_max / len(predictions)
        if style_ratio >= 0.8:
            print(
                "  STYLE_COLLAPSE_GUARD: style distribution is overly concentrated. "
                f"dominant={style_max}/{len(predictions)} ({style_ratio:.1%}). "
                "Consider rerunning style with stricter prompt/audit."
            )
    
    # 准备金标
    print("\n[4/5] Preparing gold labels...")
    gold_labels = []
    
    for pred in predictions:
        essay_id = pred["essay_id"]
        row = prompt7_df[prompt7_df['essay_id'] == essay_id].iloc[0]
        
        gold_labels.append({
            "essay_id": int(essay_id),
            "prompt_id": 7,
            "ideas": int(row['rater1_trait1']),
            "organization": int(row['rater1_trait2']),
            "style": int(row['rater1_trait3']),
            "conventions": int(row['rater1_trait4']),
            "overall": int(row['domain1_score'])
        })
    
    print(f"✓ Prepared {len(gold_labels)} gold labels")
    gold_distribution = build_distribution_dict(gold_labels, trait_keys + ("overall",))
    gold_map = {x["essay_id"]: x for x in gold_labels}
    if args.save_threshold_cache:
        _annotate_gold_fields(threshold_cache, gold_map)
    if args.enable_sidecar and sidecar_cache:
        _annotate_gold_fields(sidecar_cache, gold_map)
    
    # 计算 QWK
    print("\n[5/5] Computing QWK...")
    
    # 组装预测与金标为评估样本格式
    samples = []
    for pred, gold in zip(predictions, gold_labels):
        samples.append({
            "prompt_id": gold["prompt_id"],
            "gold_trait_scores": {
                "ideas": gold["ideas"],
                "organization": gold["organization"],
                "style": gold["style"],
                "conventions": gold["conventions"]
            },
            "pred_trait_scores": {
                "ideas": pred["ideas"],
                "organization": pred["organization"],
                "style": pred["style"],
                "conventions": pred["conventions"]
            },
            "gold_overall_score": gold["overall"],
            "pred_overall_score": pred["overall"]
        })
    
    evaluator = QWKEvaluator()
    qwk_results = evaluator.evaluate(samples)
    
    # 附加元数据
    qwk_results["prompt_id"] = 7
    qwk_results["num_samples"] = len(samples)
    qwk_results["ideas_org_main_mode"] = {
        "structured_enabled": bool(Prompt7Mapper.ENABLE_IDEAS_ORG_STRUCTURED_MAIN)
    }
    mean_trait_qwk, dual_objective = compute_dual_objective(
        qwk_results,
        trait_weight=args.trait_weight,
        overall_weight=args.overall_weight,
    )
    qwk_results["mean_trait_qwk"] = float(mean_trait_qwk)
    qwk_results["dual_objective"] = float(dual_objective)
    qwk_results["dual_objective_weights"] = {
        "trait_weight": float(args.trait_weight),
        "overall_weight": float(args.overall_weight),
    }
    qwk_results["prediction_distribution"] = prediction_distribution
    qwk_results["gold_distribution"] = gold_distribution
    if sidecar_run_info is not None:
        qwk_results["sidecar"] = sidecar_run_info

    if args.enable_ideas_org_shadow:
        shadow_samples = []
        for pred, gold in zip(predictions, gold_labels):
            s_ideas = pred.get("shadow_ideas")
            s_org = pred.get("shadow_organization")
            if s_ideas is None or s_org is None:
                continue
            shadow_overall = pipeline.mapper.compute_overall_score(
                int(s_ideas),
                int(s_org),
                int(pred["style"]),
                int(pred["conventions"]),
            )
            shadow_samples.append(
                {
                    "prompt_id": gold["prompt_id"],
                    "gold_trait_scores": {
                        "ideas": gold["ideas"],
                        "organization": gold["organization"],
                    },
                    "pred_trait_scores": {
                        "ideas": int(s_ideas),
                        "organization": int(s_org),
                    },
                    "gold_overall_score": gold["overall"],
                    "pred_overall_score": shadow_overall,
                }
            )

        shadow_payload = {"enabled": True, "num_samples": len(shadow_samples)}
        if shadow_samples:
            shadow_eval = evaluator.evaluate(shadow_samples)
            shadow_ideas_counter = Counter(
                int(x["shadow_ideas"]) for x in predictions if x.get("shadow_ideas") is not None
            )
            shadow_org_counter = Counter(
                int(x["shadow_organization"]) for x in predictions if x.get("shadow_organization") is not None
            )
            shadow_payload["trait_qwk"] = shadow_eval.get("trait_qwk", {})
            shadow_payload["overall_qwk_with_shadow_ideas_org"] = float(shadow_eval.get("overall_qwk", 0.0))
            shadow_payload["prediction_distribution"] = {
                "shadow_ideas": {str(k): int(v) for k, v in sorted(shadow_ideas_counter.items())},
                "shadow_organization": {str(k): int(v) for k, v in sorted(shadow_org_counter.items())},
            }
        qwk_results["shadow_mode"] = shadow_payload
    
    # 打印结果
    print("\n" + "="*80)
    print("QWK EVALUATION RESULTS")
    print("="*80)
    print(f"Prompt: {qwk_results['prompt_id']}")
    print(f"Samples: {qwk_results['num_samples']}")
    print(f"\nTrait QWK:")
    for trait, qwk in qwk_results['trait_qwk'].items():
        print(f"  {trait.capitalize():15s}: {qwk:.4f}")
    print(f"\nOverall QWK: {qwk_results['overall_qwk']:.4f}")
    if args.enable_ideas_org_shadow:
        shadow_info = qwk_results.get("shadow_mode", {})
        print(
            "Shadow Mode (ideas/org only, non-scoring): "
            f"samples={shadow_info.get('num_samples', 0)}"
        )
        if shadow_info.get("trait_qwk"):
            for trait, sqwk in shadow_info["trait_qwk"].items():
                print(f"  Shadow {trait.capitalize():9s}: {sqwk:.4f}")
            print(
                "  Shadow Overall QWK (replace ideas/org only): "
                f"{shadow_info.get('overall_qwk_with_shadow_ideas_org', 0.0):.4f}"
            )
    if args.print_aggregate_metrics:
        print(f"Mean Trait QWK: {qwk_results['mean_trait_qwk']:.4f}")
        print(
            "Dual Objective: "
            f"{qwk_results['dual_objective']:.4f} "
            f"(trait_weight={args.trait_weight}, overall_weight={args.overall_weight})"
        )
    print("="*80)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存预测
    pred_file = f"prompt7_predictions_{timestamp}.json"
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Predictions saved to: {pred_file}")
    
    # 保存 QWK 结果
    qwk_file = f"prompt7_qwk_results_{timestamp}.json"
    with open(qwk_file, 'w', encoding='utf-8') as f:
        json.dump(qwk_results, f, indent=2, ensure_ascii=False)
    print(f"✓ QWK results saved to: {qwk_file}")

    if args.save_threshold_cache:
        cache_path = args.save_threshold_cache
        if not cache_path.strip():
            cache_path = f"prompt7_threshold_cache_{timestamp}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(threshold_cache, f, indent=2, ensure_ascii=False)
        print(f"✓ Threshold cache saved to: {cache_path}")

    if args.save_detailed_report:
        detail_rows = build_prompt7_detailed_rows(
            predictions=predictions,
            gold_labels=gold_labels,
            prompt7_df=prompt7_df,
            threshold_cache=threshold_cache,
            sidecar_cache=sidecar_cache,
            include_essay_text=bool(args.detailed_report_include_essay_text),
            include_reasoning=bool(args.detailed_report_include_reasoning),
        )

        selected_rows = detail_rows
        requested_ids = _parse_essay_id_list(args.detailed_report_essay_ids)
        if requested_ids:
            row_map = {int(x["essay_id"]): x for x in detail_rows}
            selected_rows = [row_map[eid] for eid in requested_ids if eid in row_map]
        else:
            if args.detailed_report_sort_by == "abs_overall_delta_vs_gold":
                selected_rows = sorted(
                    detail_rows,
                    key=lambda x: (
                        abs(x["delta_vs_gold"]["overall"])
                        if x["delta_vs_gold"]["overall"] is not None
                        else -1
                    ),
                    reverse=True,
                )
            elif args.detailed_report_sort_by == "overall_delta_vs_gold":
                selected_rows = sorted(
                    detail_rows,
                    key=lambda x: (
                        x["delta_vs_gold"]["overall"]
                        if x["delta_vs_gold"]["overall"] is not None
                        else -10**9
                    ),
                    reverse=True,
                )
            else:
                selected_rows = sorted(detail_rows, key=lambda x: int(x["essay_id"]))

            if args.detailed_report_top_k > 0:
                selected_rows = selected_rows[: int(args.detailed_report_top_k)]

        abs_deltas = [
            abs(int(x["delta_vs_gold"]["overall"]))
            for x in detail_rows
            if x["delta_vs_gold"]["overall"] is not None
        ]
        report_payload = {
            "meta": {
                "prompt_id": 7,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_rows_total": len(detail_rows),
                "num_rows_selected": len(selected_rows),
                "sort_by": args.detailed_report_sort_by,
                "top_k": int(args.detailed_report_top_k),
                "requested_essay_ids": requested_ids,
                "include_essay_text": bool(args.detailed_report_include_essay_text),
                "include_reasoning": bool(args.detailed_report_include_reasoning),
            },
            "summary": {
                "mean_abs_overall_delta_vs_gold": (
                    float(sum(abs_deltas) / len(abs_deltas)) if abs_deltas else None
                ),
                "max_abs_overall_delta_vs_gold": (max(abs_deltas) if abs_deltas else None),
            },
            "rows": selected_rows,
        }

        report_path = args.save_detailed_report
        if not report_path.strip():
            report_path = f"prompt7_detailed_report_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed score report saved to: {report_path}")
        preview = selected_rows[: min(3, len(selected_rows))]
        for row in preview:
            print(
                "  "
                f"essay_id={row['essay_id']} "
                f"pred_overall={row['prediction']['overall']} "
                f"gold_overall={row['gold']['overall']} "
                f"delta={row['delta_vs_gold']['overall']}"
            )


if __name__ == "__main__":
    main()
