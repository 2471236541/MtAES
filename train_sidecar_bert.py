#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Prompt7 BERT sidecar models for selected traits (labels 0-3).

Outputs:
- <output-dir>/ideas/* (HF model + tokenizer + calibration.json)
- <output-dir>/organization/* (HF model + tokenizer + calibration.json)
- <output-dir>/style/* (HF model + tokenizer + calibration.json)
- <output-dir>/conventions/* (HF model + tokenizer + calibration.json)
- <output-dir>/sidecar_train_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gpu_auto_select import acquire_gpu_lease
from sidecar_text_preset import build_trait_text, parse_trait_text_presets, trait_input_config_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True, help="Train split JSON.")
    parser.add_argument("--dev-cache", required=True, help="Dev split JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--loss-mode",
        choices=("ce", "ce_ordinal", "ce_ordinal_weighted"),
        default="ce",
        help="Training loss mode. Default keeps the original cross-entropy behavior.",
    )
    parser.add_argument(
        "--ordinal-loss-weight",
        type=float,
        default=0.35,
        help="Weight for the ordinal distance penalty when enabled.",
    )
    parser.add_argument(
        "--class-weights",
        default="",
        help=(
            "Optional comma-separated class weights for ce_ordinal_weighted, "
            "for example '1.0,1.05,1.15,1.25'."
        ),
    )
    parser.add_argument(
        "--traits",
        default="ideas,organization",
        help="Comma-separated traits to train. Supported: ideas,organization,style,conventions",
    )
    parser.add_argument(
        "--trait-text-presets",
        default="",
        help=(
            "Optional trait-specific input preset list, e.g. "
            "'organization=essay_plus_trait_reasoning'."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="auto",
        help="GPU selector: auto | cpu | <physical_gpu_id>. Default: auto.",
    )
    parser.add_argument("--auto-min-free-gb", type=float, default=8.0, help="Min free GPU memory for --gpu auto.")
    parser.add_argument("--auto-wait-seconds", type=float, default=600.0, help="Wait time for an idle GPU.")
    parser.add_argument("--gpu-lock-dir", default="/tmp/llms_aes_gpu_locks", help="Directory for GPU lock files.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"JSON must be list: {p}")
    return rows


def prepare_rows_for_trait(
    rows: list[dict[str, Any]],
    trait_name: str,
    label_key: str,
    text_preset: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if label_key not in row:
            continue
        try:
            _ = int(row[label_key])
            text = build_trait_text(row, trait_name=trait_name, preset=text_preset)
            if not text:
                continue
        except Exception:
            continue
        merged = dict(row)
        merged["_sidecar_input_text"] = text
        out.append(merged)
    return out


@dataclass
class TraitSpec:
    name: str
    label_key: str


class EssayClsDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, text_key: str, label_key: str, max_length: int):
        import torch

        self._torch = torch
        texts = [str(r[text_key]) for r in rows]
        labels = [int(r[label_key]) for r in rows]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: self._torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self._torch.tensor(self.labels[idx], dtype=self._torch.long)
        return item


def summarize_token_lengths(texts: list[str], tokenizer, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=False,
        )
        lengths.append(len(encoded["input_ids"]))

    if not lengths:
        return {
            "n": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "over_limit_count": 0,
            "over_limit_ratio": 0.0,
        }

    sorted_lengths = sorted(int(x) for x in lengths)

    def pct(q: float) -> int:
        idx = int(round(q * (len(sorted_lengths) - 1)))
        idx = max(0, min(len(sorted_lengths) - 1, idx))
        return int(sorted_lengths[idx])

    over_limit_count = sum(1 for x in sorted_lengths if int(x) > int(max_length))
    return {
        "n": int(len(sorted_lengths)),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": int(sorted_lengths[-1]),
        "over_limit_count": int(over_limit_count),
        "over_limit_ratio": float(over_limit_count / len(sorted_lengths)),
    }


def build_compute_metrics_fn() -> Callable:
    import numpy as np

    def _qwk(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 0, max_rating: int = 3) -> float:
        n = max_rating - min_rating + 1
        N = len(y_true)
        if N == 0:
            return 0.0
        O = np.zeros((n, n), dtype=np.float64)
        for a, b in zip(y_true, y_pred):
            ai = int(max(min_rating, min(max_rating, int(a)))) - min_rating
            bi = int(max(min_rating, min(max_rating, int(b)))) - min_rating
            O[ai, bi] += 1.0
        hist_true = np.zeros(n, dtype=np.float64)
        hist_pred = np.zeros(n, dtype=np.float64)
        for a in y_true:
            hist_true[int(max(min_rating, min(max_rating, int(a)))) - min_rating] += 1.0
        for b in y_pred:
            hist_pred[int(max(min_rating, min(max_rating, int(b)))) - min_rating] += 1.0
        E = np.outer(hist_true, hist_pred) / float(N)
        W = np.zeros((n, n), dtype=np.float64)
        denom = float((n - 1) ** 2) if n > 1 else 1.0
        for i in range(n):
            for j in range(n):
                W[i, j] = ((i - j) ** 2) / denom
        num = float(np.sum(W * O))
        den = float(np.sum(W * E))
        if den == 0.0:
            return 0.0
        return 1.0 - (num / den)

    def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...] = (0, 1, 2, 3)) -> float:
        f1s = []
        for lb in labels:
            tp = int(np.sum((y_true == lb) & (y_pred == lb)))
            fp = int(np.sum((y_true != lb) & (y_pred == lb)))
            fn = int(np.sum((y_true == lb) & (y_pred != lb)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)
        return float(sum(f1s) / len(f1s)) if f1s else 0.0

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        labels = np.asarray(labels, dtype=np.int64)
        preds = np.asarray(preds, dtype=np.int64)
        acc = float(np.mean(labels == preds)) if len(labels) > 0 else 0.0
        macro_f1 = _macro_f1(labels, preds)
        qwk = _qwk(labels, preds, min_rating=0, max_rating=3)
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "qwk": float(qwk),
        }

    return _compute_metrics


def fit_temperature(logits, labels) -> dict[str, float]:
    import torch

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    nll_criterion = torch.nn.CrossEntropyLoss()
    before_nll = float(nll_criterion(logits_t, labels_t).item())

    temperature = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        temp = torch.clamp(temperature, min=0.05, max=10.0)
        loss = nll_criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(_closure)
    with torch.no_grad():
        t_value = float(torch.clamp(temperature, min=0.05, max=10.0).item())
        after_nll = float(nll_criterion(logits_t / t_value, labels_t).item())

    return {
        "temperature": t_value,
        "nll_before": before_nll,
        "nll_after": after_nll,
    }


DEFAULT_ORDINAL_CLASS_WEIGHTS = [1.0, 1.05, 1.15, 1.25]


def resolve_class_weights(loss_mode: str, spec: str) -> tuple[list[float] | None, str]:
    if loss_mode != "ce_ordinal_weighted":
        return None, "unused"

    raw = str(spec or "").strip()
    if not raw:
        return list(DEFAULT_ORDINAL_CLASS_WEIGHTS), "default_conservative"

    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError(f"--class-weights expects 4 comma-separated numbers, got: {spec}")

    values = [float(x) for x in parts]
    if any(x <= 0.0 for x in values):
        raise ValueError(f"--class-weights must be positive, got: {values}")
    return values, "user"


def train_single_trait(
    trait: TraitSpec,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    text_preset: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    trait_dir = Path(args.output_dir) / trait.name
    trait_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    train_texts = [str(row["_sidecar_input_text"]) for row in train_rows]
    dev_texts = [str(row["_sidecar_input_text"]) for row in dev_rows]
    train_token_stats = summarize_token_lengths(train_texts, tokenizer, args.max_length)
    dev_token_stats = summarize_token_lengths(dev_texts, tokenizer, args.max_length)
    print(
        f"token_lengths train_p50={train_token_stats['p50']} "
        f"train_p95={train_token_stats['p95']} train_max={train_token_stats['max']} "
        f"train_over={train_token_stats['over_limit_count']}/{train_token_stats['n']} "
        f"({train_token_stats['over_limit_ratio']:.2%})"
    )
    print(
        f"token_lengths dev_p50={dev_token_stats['p50']} "
        f"dev_p95={dev_token_stats['p95']} dev_max={dev_token_stats['max']} "
        f"dev_over={dev_token_stats['over_limit_count']}/{dev_token_stats['n']} "
        f"({dev_token_stats['over_limit_ratio']:.2%})"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        problem_type="single_label_classification",
    )
    model.config.label2id = {"0": 0, "1": 1, "2": 2, "3": 3}
    model.config.id2label = {0: "0", 1: "1", 2: "2", 3: "3"}

    train_ds = EssayClsDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        text_key="_sidecar_input_text",
        label_key=trait.label_key,
        max_length=args.max_length,
    )
    dev_ds = EssayClsDataset(
        rows=dev_rows,
        tokenizer=tokenizer,
        text_key="_sidecar_input_text",
        label_key=trait.label_key,
        max_length=args.max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    compute_metrics = build_compute_metrics_fn()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    epochs_int = max(1, int(round(float(args.epochs))))
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, steps_per_epoch * epochs_int)
    warmup_steps = int(round(float(args.warmup_ratio) * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    score_values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    class_weight_values, class_weight_source = resolve_class_weights(args.loss_mode, args.class_weights)
    class_weights_t = (
        torch.tensor(class_weight_values, dtype=torch.float32, device=device)
        if class_weight_values is not None
        else None
    )

    def build_loss(logits, labels):
        ce_loss = F.cross_entropy(logits.float(), labels, weight=class_weights_t)
        if args.loss_mode == "ce":
            return ce_loss

        probs = torch.softmax(logits.float(), dim=-1)
        label_positions = labels.float().unsqueeze(1)
        dist2 = ((score_values.unsqueeze(0) - label_positions) ** 2) / 9.0
        ord_loss = (probs * dist2).sum(dim=-1).mean()

        return ce_loss + float(args.ordinal_loss_weight) * ord_loss

    def evaluate_current_model() -> tuple[dict[str, float], Any, Any]:
        model.eval()
        all_logits = []
        all_labels = []
        loss_sum = 0.0
        n_items = 0
        with torch.no_grad():
            for batch in dev_loader:
                labels = batch["labels"].to(device)
                model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(**model_inputs)
                    loss = build_loss(outputs.logits, labels)
                batch_size = int(labels.shape[0])
                loss_sum += float(loss.item()) * batch_size
                n_items += batch_size
                all_logits.append(outputs.logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        logits_np = torch.cat(all_logits, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        metrics = compute_metrics((logits_np, labels_np))
        metrics = {
            "eval_accuracy": float(metrics["accuracy"]),
            "eval_macro_f1": float(metrics["macro_f1"]),
            "eval_qwk": float(metrics["qwk"]),
            "eval_loss": float(loss_sum / max(1, n_items)),
        }
        return metrics, logits_np, labels_np

    best_qwk = -1e9
    best_state = None
    global_step = 0
    train_loss_sum = 0.0
    train_loss_count = 0

    for epoch in range(epochs_int):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**model_inputs)
                loss = build_loss(outputs.logits, labels) / float(grad_accum)
            scaler.scale(loss).backward()
            train_loss_sum += float(loss.item()) * float(grad_accum)
            train_loss_count += 1

            if (step_idx % grad_accum == 0) or (step_idx == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        eval_metrics, _, _ = evaluate_current_model()
        cur_qwk = float(eval_metrics.get("eval_qwk", 0.0))
        print(
            f"trait={trait.name} epoch={epoch+1}/{epochs_int} "
            f"eval_qwk={cur_qwk:.4f} eval_loss={eval_metrics['eval_loss']:.4f}"
        )
        if cur_qwk > best_qwk:
            best_qwk = cur_qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    eval_metrics, logits, labels = evaluate_current_model()
    calibration = fit_temperature(logits, labels)

    model.save_pretrained(str(trait_dir))
    tokenizer.save_pretrained(str(trait_dir))
    (trait_dir / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    (trait_dir / "input_config.json").write_text(
        json.dumps(
            trait_input_config_payload(
                trait_name=trait.name,
                text_preset=text_preset,
                max_length=args.max_length,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "trait": trait.name,
        "label_key": trait.label_key,
        "text_preset": text_preset,
        "max_length": int(args.max_length),
        "train_rows": int(len(train_rows)),
        "dev_rows": int(len(dev_rows)),
        "token_length_stats": {
            "train": train_token_stats,
            "dev": dev_token_stats,
        },
        "loss_config": {
            "loss_mode": args.loss_mode,
            "ordinal_loss_weight": float(args.ordinal_loss_weight),
            "class_weights": class_weight_values,
            "class_weight_source": class_weight_source,
        },
        "optimization_config": {
            "model_name": str(args.model_name),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "epochs": int(epochs_int),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "fp16": bool(args.fp16),
        },
        "train_metrics": {
            "train_loss": float(train_loss_sum / max(1, train_loss_count)),
            "train_steps": int(global_step),
            "epochs": int(epochs_int),
        },
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
        "calibration": calibration,
        "output_dir": str(trait_dir),
    }
    (trait_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    with acquire_gpu_lease(
        requested=args.gpu,
        min_free_gb=args.auto_min_free_gb,
        wait_seconds=args.auto_wait_seconds,
        lock_dir=args.gpu_lock_dir,
    ) as gpu_lease:
        set_seed(args.seed)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_rows_all = load_rows(args.train_cache)
        dev_rows_all = load_rows(args.dev_cache)

        trait_map = {
            "ideas": TraitSpec(name="ideas", label_key="gold_ideas"),
            "organization": TraitSpec(name="organization", label_key="gold_organization"),
            "style": TraitSpec(name="style", label_key="gold_style"),
            "conventions": TraitSpec(name="conventions", label_key="gold_conventions"),
        }
        trait_names = [x.strip() for x in args.traits.split(",") if x.strip()]
        trait_specs = []
        for t in trait_names:
            if t not in trait_map:
                raise ValueError(f"Unsupported trait: {t}. Supported: {sorted(trait_map.keys())}")
            trait_specs.append(trait_map[t])
        if not trait_specs:
            raise ValueError("No traits selected.")

        text_presets = parse_trait_text_presets(args.trait_text_presets, trait_names)

        all_summaries = []
        for idx, trait in enumerate(trait_specs):
            text_preset = text_presets.get(trait.name, "essay_only")
            train_rows = prepare_rows_for_trait(train_rows_all, trait_name=trait.name, label_key=trait.label_key, text_preset=text_preset)
            dev_rows = prepare_rows_for_trait(dev_rows_all, trait_name=trait.name, label_key=trait.label_key, text_preset=text_preset)
            if len(train_rows) < 50 or len(dev_rows) < 20:
                raise ValueError(
                    f"Not enough rows for trait={trait.name}. train={len(train_rows)} dev={len(dev_rows)}"
                )

            print(
                f"[{idx+1}/{len(trait_specs)}] training trait={trait.name} "
                f"train={len(train_rows)} dev={len(dev_rows)} text_preset={text_preset}"
            )
            summary = train_single_trait(trait, train_rows, dev_rows, text_preset, args)
            all_summaries.append(summary)
            print(
                f"done trait={trait.name} eval_qwk={summary['eval_metrics'].get('eval_qwk', 0.0):.4f} "
                f"temperature={summary['calibration']['temperature']:.4f}"
            )

        top_summary = {
            "train_cache": args.train_cache,
            "dev_cache": args.dev_cache,
            "model_name": args.model_name,
            "seed": int(args.seed),
            "max_length": int(args.max_length),
            "optimization_config": {
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "epochs": int(round(float(args.epochs))),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "warmup_ratio": float(args.warmup_ratio),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "fp16": bool(args.fp16),
            },
            "loss_config": {
                "loss_mode": args.loss_mode,
                "ordinal_loss_weight": float(args.ordinal_loss_weight),
                "class_weights": args.class_weights,
            },
            "gpu": gpu_lease.to_dict(),
            "traits": trait_names,
            "trait_text_presets": text_presets,
            "summaries": all_summaries,
        }
        summary_path = output_dir / "sidecar_train_summary.json"
        summary_path.write_text(json.dumps(top_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"gpu={gpu_lease.physical_gpu} local_device={gpu_lease.local_device} "
            f"source={gpu_lease.source}"
        )
        print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
