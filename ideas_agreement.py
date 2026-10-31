#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for conservative ideas agreement inference.

Rule:
- if model A and model B agree on the ideas label, use the averaged logits
- otherwise fall back to the baseline ideas prediction already present in input rows

The exported contract stays compatible with the existing sidecar pipeline:
- pred / probs / max_prob / margin
- optional diagnostic arrays for agreement tracing
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ideas_ensemble import load_max_length, load_temperature, load_text_preset
from sidecar_text_preset import build_trait_text


def _build_texts(rows: list[dict[str, Any]], preset: str) -> list[str]:
    texts: list[str] = []
    for row in rows:
        text = build_trait_text(row, trait_name="ideas", preset=preset)
        if text is None:
            text = str(row.get("essay_text", "")).strip()
        texts.append(text)
    return texts


def _get_baseline_ideas(row: dict[str, Any], num_labels: int) -> int:
    for key in ("pred_ideas", "ideas", "baseline_ideas"):
        if key in row:
            try:
                value = int(row[key])
                return max(0, min(num_labels - 1, value))
            except Exception:
                continue
    return 0


def _build_fallback_probs(label: int, num_labels: int) -> list[float]:
    weights = [math.exp(-0.7 * abs(idx - label)) for idx in range(num_labels)]
    total = sum(weights) or 1.0
    return [float(weight / total) for weight in weights]


def run_ideas_agreement_inference(
    rows: list[dict[str, Any]],
    model_dirs: list[Path],
    batch_size: int,
    max_length: int,
    device_name: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not rows:
        raise ValueError("Input rows are empty.")
    if len(model_dirs) != 2:
        raise ValueError("Ideas agreement requires exactly 2 model dirs.")

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cached_texts: dict[str, list[str]] = {}
    model_outputs: list[dict[str, Any]] = []

    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"Ideas agreement model dir not found: {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        model.to(device)

        temperature = load_temperature(model_dir)
        text_preset = load_text_preset(model_dir)
        model_max_length = load_max_length(model_dir, max_length)
        texts = cached_texts.setdefault(text_preset, _build_texts(rows, text_preset))

        batch_logits_cpu: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=model_max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits / float(temperature)
                batch_logits_cpu.append(logits.detach().cpu())

        logits_cpu = torch.cat(batch_logits_cpu, dim=0)
        probs = torch.softmax(logits_cpu, dim=-1)
        preds = probs.argmax(dim=-1).tolist()
        model_outputs.append(
            {
                "model_dir": str(model_dir),
                "temperature": float(temperature),
                "text_preset": str(text_preset),
                "max_length": int(model_max_length),
                "logits": logits_cpu,
                "preds": [int(x) for x in preds],
            }
        )

        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logits_a = model_outputs[0]["logits"]
    logits_b = model_outputs[1]["logits"]
    num_labels = int(logits_a.shape[-1])

    preds: list[int] = []
    probs_all: list[list[float]] = []
    max_probs: list[float] = []
    margins: list[float] = []
    pred_a_list: list[int] = []
    pred_b_list: list[int] = []
    max_prob_a_list: list[float] = []
    max_prob_b_list: list[float] = []
    agreement_used: list[bool] = []
    agreement_reason: list[str] = []

    for idx, row in enumerate(rows):
        pred_a = int(model_outputs[0]["preds"][idx])
        pred_b = int(model_outputs[1]["preds"][idx])
        pred_a_list.append(pred_a)
        pred_b_list.append(pred_b)
        probs_a_tensor = torch.softmax(logits_a[idx], dim=-1)
        probs_b_tensor = torch.softmax(logits_b[idx], dim=-1)
        max_prob_a_list.append(float(probs_a_tensor[pred_a].item()))
        max_prob_b_list.append(float(probs_b_tensor[pred_b].item()))

        if pred_a == pred_b:
            avg_logits = (logits_a[idx] + logits_b[idx]) / 2.0
            probs_tensor = torch.softmax(avg_logits, dim=-1)
            top2_vals, top2_idx = torch.topk(probs_tensor, k=min(2, num_labels), dim=-1)
            pred = int(top2_idx[0].item())
            probs = [float(x) for x in probs_tensor.tolist()]
            max_prob = float(top2_vals[0].item())
            margin = float((top2_vals[0] - top2_vals[1]).item()) if num_labels >= 2 else float(top2_vals[0].item())
            used = True
            reason = "agree_use_avg_logits"
        else:
            pred = _get_baseline_ideas(row, num_labels=num_labels)
            probs = _build_fallback_probs(pred, num_labels=num_labels)
            sorted_probs = sorted(probs, reverse=True)
            max_prob = float(sorted_probs[0])
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else float(sorted_probs[0])
            used = False
            reason = "disagree_fallback_baseline"

        preds.append(int(pred))
        probs_all.append([float(x) for x in probs])
        max_probs.append(float(max_prob))
        margins.append(float(margin))
        agreement_used.append(bool(used))
        agreement_reason.append(reason)

    used_count = sum(1 for x in agreement_used if x)
    return {
        "preds": preds,
        "probs": probs_all,
        "max_probs": max_probs,
        "margins": margins,
        "temperature": float(sum(float(item["temperature"]) for item in model_outputs) / len(model_outputs)),
        "text_preset": "agreement",
        "agreement_meta": [
            {
                "model_dir": str(item["model_dir"]),
                "temperature": float(item["temperature"]),
                "text_preset": str(item["text_preset"]),
                "max_length": int(item["max_length"]),
            }
            for item in model_outputs
        ],
        "preds_a": pred_a_list,
        "preds_b": pred_b_list,
        "max_probs_a": max_prob_a_list,
        "max_probs_b": max_prob_b_list,
        "agreement_used": agreement_used,
        "agreement_reason": agreement_reason,
        "agreement_used_count": int(used_count),
        "agreement_used_ratio": float(used_count / len(rows)),
    }
