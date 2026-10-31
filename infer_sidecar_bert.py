#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for Prompt7 BERT sidecar models (selected traits).

Input:
- JSON rows containing at least essay_id, essay_text (and usually baseline preds).

Output:
- Same rows + bert_* fields per trait:
  bert_<trait>_pred, bert_<trait>_probs, bert_<trait>_max_prob, bert_<trait>_margin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpu_auto_select import acquire_gpu_lease
from ideas_agreement import run_ideas_agreement_inference
from ideas_ensemble import resolve_model_dirs, run_ideas_ensemble_inference
from sidecar_text_preset import DEFAULT_TEXT_PRESET, build_trait_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", required=True, help="Input JSON rows.")
    parser.add_argument("--model-dir", required=True, help="Directory containing trait subfolders.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--traits",
        default="ideas,organization",
        help="Comma-separated traits to infer. Example: ideas,organization or conventions",
    )
    parser.add_argument(
        "--gpu",
        default="auto",
        help="GPU selector: auto | cpu | <physical_gpu_id>. Default: auto.",
    )
    parser.add_argument("--auto-min-free-gb", type=float, default=8.0, help="Min free GPU memory for --gpu auto.")
    parser.add_argument("--auto-wait-seconds", type=float, default=600.0, help="Wait time for an idle GPU.")
    parser.add_argument("--gpu-lock-dir", default="/tmp/llms_aes_gpu_locks", help="Directory for GPU lock files.")
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
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list.")
    return rows


def load_temperature(trait_dir: Path) -> float:
    calib_path = trait_dir / "calibration.json"
    if not calib_path.exists():
        return 1.0
    try:
        payload = json.loads(calib_path.read_text(encoding="utf-8"))
        t = float(payload.get("temperature", 1.0))
        return max(0.05, min(10.0, t))
    except Exception:
        return 1.0


def load_input_config(trait_dir: Path) -> dict[str, Any]:
    config_path = trait_dir / "input_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        return {}
    except Exception:
        return {}


def load_text_preset(trait_dir: Path) -> str:
    payload = load_input_config(trait_dir)
    return str(payload.get("text_preset", DEFAULT_TEXT_PRESET))


def load_max_length(trait_dir: Path, fallback: int) -> int:
    payload = load_input_config(trait_dir)
    try:
        value = int(payload.get("max_length", fallback))
        return max(8, value)
    except Exception:
        return int(fallback)


def run_trait_inference(
    rows: list[dict[str, Any]],
    trait: str,
    model_root: Path,
    batch_size: int,
    max_length: int,
    device_name: str,
) -> dict[str, list]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    trait_dir = model_root / trait
    if not trait_dir.exists():
        raise FileNotFoundError(f"Trait model dir not found: {trait_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(trait_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(trait_dir))
    model.eval()

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    temperature = load_temperature(trait_dir)
    text_preset = load_text_preset(trait_dir)
    trait_max_length = load_max_length(trait_dir, max_length)

    texts = []
    for row in rows:
        text = build_trait_text(row, trait_name=trait, preset=text_preset)
        if text is None:
            text = str(row.get("essay_text", "")).strip()
        texts.append(text)
    preds: list[int] = []
    probs_all: list[list[float]] = []
    max_probs: list[float] = []
    margins: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=trait_max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            logits = logits / float(temperature)
            probs = torch.softmax(logits, dim=-1)
            top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)

            batch_preds = top2_idx[:, 0].detach().cpu().numpy().tolist()
            batch_probs = probs.detach().cpu().numpy().tolist()
            batch_max = top2_vals[:, 0].detach().cpu().numpy().tolist()
            batch_margin = (top2_vals[:, 0] - top2_vals[:, 1]).detach().cpu().numpy().tolist()

            preds.extend([int(x) for x in batch_preds])
            probs_all.extend([[float(y) for y in x] for x in batch_probs])
            max_probs.extend([float(x) for x in batch_max])
            margins.extend([float(x) for x in batch_margin])

    return {
        "preds": preds,
        "probs": probs_all,
        "max_probs": max_probs,
        "margins": margins,
        "temperature": float(temperature),
        "text_preset": str(text_preset),
        "max_length": int(trait_max_length),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_cache)
    if not rows:
        raise ValueError("Input rows are empty.")
    if str(args.ideas_ensemble_dirs).strip() and str(args.ideas_agreement_dirs).strip():
        raise ValueError("Use either --ideas-ensemble-dirs or --ideas-agreement-dirs, not both.")

    repo_root = Path(__file__).resolve().parent
    model_root = Path(args.model_dir)
    if not model_root.is_absolute():
        model_root = repo_root / model_root
    trait_names = [x.strip() for x in str(args.traits).split(",") if x.strip()]
    if not trait_names:
        raise ValueError("No traits selected for inference.")
    for t in trait_names:
        if t not in {"ideas", "organization", "conventions", "style"}:
            raise ValueError(f"Unsupported trait: {t}")

    with acquire_gpu_lease(
        requested=args.gpu,
        min_free_gb=args.auto_min_free_gb,
        wait_seconds=args.auto_wait_seconds,
        lock_dir=args.gpu_lock_dir,
    ) as gpu_lease:
        trait_outputs: dict[str, dict[str, list]] = {}
        ideas_ensemble_dirs = resolve_model_dirs(args.ideas_ensemble_dirs, repo_root) if args.ideas_ensemble_dirs else []
        ideas_agreement_dirs = resolve_model_dirs(args.ideas_agreement_dirs, repo_root) if args.ideas_agreement_dirs else []
        for trait in trait_names:
            if trait == "ideas" and ideas_agreement_dirs:
                trait_outputs[trait] = run_ideas_agreement_inference(
                    rows=rows,
                    model_dirs=ideas_agreement_dirs,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device_name=gpu_lease.local_device,
                )
            elif trait == "ideas" and ideas_ensemble_dirs:
                trait_outputs[trait] = run_ideas_ensemble_inference(
                    rows=rows,
                    model_dirs=ideas_ensemble_dirs,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device_name=gpu_lease.local_device,
                )
            else:
                trait_outputs[trait] = run_trait_inference(
                    rows,
                    trait,
                    model_root,
                    args.batch_size,
                    args.max_length,
                    gpu_lease.local_device,
                )

        merged = []
        for idx, row in enumerate(rows):
            out = dict(row)
            for trait in trait_names:
                t_out = trait_outputs[trait]
                out[f"bert_{trait}_pred"] = int(t_out["preds"][idx])
                out[f"bert_{trait}_probs"] = [round(float(x), 8) for x in t_out["probs"][idx]]
                out[f"bert_{trait}_max_prob"] = float(t_out["max_probs"][idx])
                out[f"bert_{trait}_margin"] = float(t_out["margins"][idx])
                if trait == "ideas" and "agreement_used" in t_out:
                    out["bert_ideas_pred_a"] = int(t_out["preds_a"][idx])
                    out["bert_ideas_pred_b"] = int(t_out["preds_b"][idx])
                    out["bert_ideas_max_prob_a"] = float(t_out["max_probs_a"][idx])
                    out["bert_ideas_max_prob_b"] = float(t_out["max_probs_b"][idx])
                    out["bert_ideas_agreement_used"] = bool(t_out["agreement_used"][idx])
                    out["bert_ideas_agreement_reason"] = str(t_out["agreement_reason"][idx])
            merged.append(out)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"rows={len(merged)}")
        def format_temp_summary(trait_name: str) -> str:
            t_out = trait_outputs[trait_name]
            if "agreement_meta" in t_out:
                temps = ",".join(f"{float(item['temperature']):.4f}" for item in t_out["agreement_meta"])
                return f"{trait_name}:agreement[{temps}]"
            if "ensemble_meta" in t_out:
                temps = ",".join(f"{float(item['temperature']):.4f}" for item in t_out["ensemble_meta"])
                return f"{trait_name}:ensemble[{temps}]"
            return f"{trait_name}:{t_out['temperature']:.4f}"

        def format_preset_summary(trait_name: str) -> str:
            t_out = trait_outputs[trait_name]
            if "agreement_meta" in t_out:
                presets = sorted({str(item["text_preset"]) for item in t_out["agreement_meta"]})
                return f"{trait_name}:agreement[{'+'.join(presets)}]"
            if "ensemble_meta" in t_out:
                presets = sorted({str(item["text_preset"]) for item in t_out["ensemble_meta"]})
                return f"{trait_name}:ensemble[{'+'.join(presets)}]"
            return f"{trait_name}:{t_out['text_preset']}"

        def format_length_summary(trait_name: str) -> str:
            t_out = trait_outputs[trait_name]
            if "agreement_meta" in t_out:
                lengths = ",".join(str(int(item["max_length"])) for item in t_out["agreement_meta"])
                return f"{trait_name}:agreement[{lengths}]"
            if "ensemble_meta" in t_out:
                lengths = ",".join(str(int(item["max_length"])) for item in t_out["ensemble_meta"])
                return f"{trait_name}:ensemble[{lengths}]"
            return f"{trait_name}:{int(t_out['max_length'])}"

        temps = ",".join(format_temp_summary(t) for t in trait_names)
        presets = ",".join(format_preset_summary(t) for t in trait_names)
        lengths = ",".join(format_length_summary(t) for t in trait_names)
        print(f"traits={','.join(trait_names)}")
        print(f"gpu={gpu_lease.physical_gpu} local_device={gpu_lease.local_device} source={gpu_lease.source}")
        print(f"temperatures={temps}")
        print(f"text_presets={presets}")
        print(f"max_lengths={lengths}")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
