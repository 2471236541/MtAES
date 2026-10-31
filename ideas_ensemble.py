#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for ideas-side model ensembling.

The ensemble keeps the existing sidecar contract unchanged:
- average temperature-scaled logits across models
- derive pred/probs/max_prob/margin from averaged logits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sidecar_text_preset import DEFAULT_TEXT_PRESET, build_trait_text


def resolve_model_dirs(spec: str, repo_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for raw_part in str(spec or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        path = Path(part)
        if not path.is_absolute():
            path = repo_root / path
        dirs.append(path)
    if not dirs:
        raise ValueError("No ensemble model dirs provided.")
    return dirs


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


def _build_texts(rows: list[dict[str, Any]], preset: str) -> list[str]:
    texts: list[str] = []
    for row in rows:
        text = build_trait_text(row, trait_name="ideas", preset=preset)
        if text is None:
            text = str(row.get("essay_text", "")).strip()
        texts.append(text)
    return texts


def run_ideas_ensemble_inference(
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
    if not model_dirs:
        raise ValueError("No ideas ensemble model dirs provided.")

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cached_texts: dict[str, list[str]] = {}
    logits_list: list[torch.Tensor] = []
    ensemble_meta: list[dict[str, Any]] = []

    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"Ideas ensemble model dir not found: {model_dir}")

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

        model_logits = torch.cat(batch_logits_cpu, dim=0)
        logits_list.append(model_logits)
        ensemble_meta.append(
            {
                "model_dir": str(model_dir),
                "temperature": float(temperature),
                "text_preset": str(text_preset),
                "max_length": int(model_max_length),
            }
        )

        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    probs = torch.softmax(avg_logits, dim=-1)
    top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)

    preds = top2_idx[:, 0].detach().cpu().numpy().tolist()
    probs_all = probs.detach().cpu().numpy().tolist()
    max_probs = top2_vals[:, 0].detach().cpu().numpy().tolist()
    margins = (top2_vals[:, 0] - top2_vals[:, 1]).detach().cpu().numpy().tolist()

    return {
        "preds": [int(x) for x in preds],
        "probs": [[float(y) for y in row] for row in probs_all],
        "max_probs": [float(x) for x in max_probs],
        "margins": [float(x) for x in margins],
        "temperature": float(sum(item["temperature"] for item in ensemble_meta) / len(ensemble_meta)),
        "text_preset": ensemble_meta[0]["text_preset"],
        "max_length": int(max(item["max_length"] for item in ensemble_meta)),
        "ensemble_meta": ensemble_meta,
    }
