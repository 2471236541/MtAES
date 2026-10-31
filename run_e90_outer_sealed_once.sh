#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/rcy/rcy/LLMs-AES-IELTS}"
PY="${PY:-/home/rcy/miniconda3/envs/llm/bin/python}"

DATA_ROOT="${DATA_ROOT:-$ROOT/sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424}"
BASE_OUTER_ROOT="${BASE_OUTER_ROOT:-$ROOT/sidecar_preds/v10ncv_clean_outer_e59_sealed_20260427/e59_final_outer}"

RUN_ID="${RUN_ID:-v10ncv_clean_outer_e90_e79_e87_sealed_20260504}"
PRED_ROOT="${PRED_ROOT:-$ROOT/sidecar_preds/$RUN_ID}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/sidecar_models/$RUN_ID}"
LOG_ROOT="$PRED_ROOT/logs"

E79_DONOR_ROOT="$PRED_ROOT/e79_ideas_donor_outer"
E87_DONOR_ROOT="$PRED_ROOT/e87_ideas_donor_outer"
E90_OUT_ROOT="$PRED_ROOT/e90_primary_outer"
E90_SUMMARY_OUT="$PRED_ROOT/e90_outer_summary.json"
E90_KEY_OUT="$PRED_ROOT/e90_outer_key_results.json"
MANIFEST_OUT="$PRED_ROOT/sealed_manifest.json"

OUTER_FOLDS="${OUTER_FOLDS:-outer0 outer1 outer2 outer3 outer4}"
SEEDS="${SEEDS:-20260325 20260412}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MAX_JOBS="${MAX_JOBS:-8}"

cd "$ROOT"
mkdir -p "$PRED_ROOT" "$MODEL_ROOT" "$LOG_ROOT"

GPU_LIST=( $(echo "$GPUS" | tr ',' ' ') )
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  GPU_LIST=(0)
fi

PIDS=()
TAGS=()

gpu_for_index() {
  local idx="$1"
  echo "${GPU_LIST[$(( idx % ${#GPU_LIST[@]} ))]}"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

model_ready() {
  local trait_dir="$1"
  [[ -f "$trait_dir/config.json" ]] || return 1
  [[ -f "$trait_dir/pytorch_model.bin" ]] || return 1
  [[ -f "$trait_dir/tokenizer_config.json" ]] || return 1
  if [[ -f "$trait_dir/tokenizer.json" || -f "$trait_dir/vocab.txt" || -f "$trait_dir/spm.model" ]]; then
    return 0
  fi
  return 1
}

wait_for_slot() {
  while [[ "${#PIDS[@]}" -ge "$MAX_JOBS" ]]; do
    local pid="${PIDS[0]}"
    local tag="${TAGS[0]}"
    if wait "$pid"; then
      echo "[e90-outer] done   $tag"
    else
      echo "[e90-outer] FAIL   $tag (see $LOG_ROOT/${tag}.log)" >&2
      exit 1
    fi
    PIDS=("${PIDS[@]:1}")
    TAGS=("${TAGS[@]:1}")
  done
}

wait_all_jobs() {
  local status=0
  local i pid tag
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    tag="${TAGS[$i]}"
    if wait "$pid"; then
      echo "[e90-outer] done   $tag"
    else
      echo "[e90-outer] FAIL   $tag (see $LOG_ROOT/${tag}.log)" >&2
      status=1
    fi
  done
  PIDS=()
  TAGS=()
  return "$status"
}

train_and_infer_donor() {
  local donor="$1"
  local model_name="$2"
  local preset="$3"
  local max_length="$4"
  local outer="$5"
  local seed="$6"
  local gpu="$7"
  local donor_root="$8"

  local train_cache="$DATA_ROOT/$outer/refit/sidecar_train.json"
  local dev_cache="$DATA_ROOT/$outer/refit/sidecar_dev.json"
  local test_cache="$DATA_ROOT/$outer/sidecar_final_test.json"
  local model_dir="$MODEL_ROOT/$donor/$outer/refit/seed${seed}"
  local out_dir="$donor_root/$outer/refit/seed${seed}"
  local prefix="v10ncv_${outer}_seed${seed}_outer"
  local infer_out="$out_dir/${prefix}_ideas_with_bert.json"

  require_path "$train_cache"
  require_path "$dev_cache"
  require_path "$test_cache"
  mkdir -p "$model_dir" "$out_dir"

  if ! model_ready "$model_dir/ideas"; then
    "$PY" "$ROOT/train_sidecar_bert.py" \
      --train-cache "$train_cache" \
      --dev-cache "$dev_cache" \
      --output-dir "$model_dir" \
      --gpu "$gpu" \
      --model-name "$model_name" \
      --max-length "$max_length" \
      --epochs 5 \
      --batch-size 16 \
      --eval-batch-size 32 \
      --learning-rate 3e-5 \
      --weight-decay 0.01 \
      --seed "$seed" \
      --traits ideas \
      --trait-text-presets "ideas=$preset" \
      --loss-mode ce_ordinal_weighted \
      --ordinal-loss-weight 0.20 \
      --class-weights 1.05,1.45,0.92,1.15
  fi

  if [[ ! -f "$infer_out" ]]; then
    "$PY" "$ROOT/infer_sidecar_bert.py" \
      --input-cache "$test_cache" \
      --model-dir "$model_dir" \
      --output "$infer_out" \
      --traits ideas \
      --batch-size 32 \
      --max-length "$max_length" \
      --gpu "$gpu"
  fi
}

run_donor_cell() {
  local outer="$1"
  local seed="$2"
  local gpu="$3"
  train_and_infer_donor e79 microsoft/deberta-v3-base essay_head_tail 320 "$outer" "$seed" "$gpu" "$E79_DONOR_ROOT"
  train_and_infer_donor e87 bert-base-uncased essay_head_tail 320 "$outer" "$seed" "$gpu" "$E87_DONOR_ROOT"
}

check_base_complete() {
  local missing=0
  local outer seed prefix base_dir
  for outer in $OUTER_FOLDS; do
    for seed in $SEEDS; do
      prefix="v10ncv_${outer}_seed${seed}_outer"
      base_dir="$BASE_OUTER_ROOT/$outer/refit/seed${seed}"
      for suffix in _selected_preds.json _selected_qwk.json; do
        if [[ ! -f "$base_dir/${prefix}${suffix}" ]]; then
          echo "Missing base artifact: $base_dir/${prefix}${suffix}" >&2
          missing=1
        fi
      done
    done
  done
  if [[ "$missing" != "0" ]]; then
    exit 1
  fi
}

echo "[e90-outer] sealed outer confirmation"
echo "[e90-outer] frozen primary=e90_e79_e87_both_dir"
echo "[e90-outer] rule=all_both_both_dir_up0m0_dn0m0_step3"
echo "[e90-outer] no outer-based tuning; do not change rules after this run"
check_base_complete

echo "[e90-outer] phase 1/3 train fixed refit ideas donors and infer outer_test"
idx=0
for outer in $OUTER_FOLDS; do
  for seed in $SEEDS; do
    wait_for_slot
    tag="donors_${outer}_seed${seed}"
    (
      run_donor_cell "$outer" "$seed" "$(gpu_for_index "$idx")"
    ) >"$LOG_ROOT/${tag}.log" 2>&1 &
    PIDS+=("$!")
    TAGS+=("$tag")
    echo "[e90-outer] started $tag gpu=$(gpu_for_index "$idx") pid=${PIDS[-1]}"
    idx=$((idx + 1))
  done
done
wait_all_jobs

echo "[e90-outer] phase 2/3 apply frozen e90 primary"
"$PY" "$ROOT/apply_e90_e79_e87_outer.py" \
  --base-root "$BASE_OUTER_ROOT" \
  --donor-roots "$E79_DONOR_ROOT,$E87_DONOR_ROOT" \
  --output-root "$E90_OUT_ROOT" \
  --summary-output "$E90_SUMMARY_OUT" \
  --outer-folds "$(echo "$OUTER_FOLDS" | tr ' ' ',')" \
  --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
  --label e90_e79_e87_all_both_both_dir_up0m0_dn0m0_step3 \
  --max-step 3 \
  --scope-note "sealed outer primary: frozen e90 e79+e87 both-direction ideas donor rule selected by inner-only e89/e90; no outer-based tuning"

echo "[e90-outer] phase 3/3 write compact key results and sealed manifest"
"$PY" "$ROOT/write_outer_key_results.py" \
  --outer-summary "$E90_SUMMARY_OUT" \
  --output "$E90_KEY_OUT" \
  --note "sealed primary e90 e79+e87; selected by inner-only e89/e90; no outer-based tuning"

RUN_ID="$RUN_ID" DATA_ROOT="$DATA_ROOT" BASE_OUTER_ROOT="$BASE_OUTER_ROOT" PRED_ROOT="$PRED_ROOT" MODEL_ROOT="$MODEL_ROOT" E90_SUMMARY_OUT="$E90_SUMMARY_OUT" E90_KEY_OUT="$E90_KEY_OUT" MANIFEST_OUT="$MANIFEST_OUT" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "type": "sealed_outer_once",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": "Primary is frozen e90 e79+e87 before running outer. Do not use this result for tuning or post-hoc selection.",
    "outer_test_touched": True,
    "selection_on_outer": False,
    "selection_uses_gold": False,
    "selection_source": {
        "inner_only_ablation": "/data/rcy/rcy/LLMs-AES-IELTS/sidecar_preds/v10ncv_clean_confirm30_20260425_r1/e89_e88_donor_ablation_inneronly_20260504/e89_e88_donor_ablation_summary.md",
        "inner_only_audit": "/data/rcy/rcy/LLMs-AES-IELTS/sidecar_preds/v10ncv_clean_confirm30_20260425_r1/e90_e79_e87_robust_audit_inneronly_20260504/e90_e79_e87_robust_audit_summary.md",
    },
    "run_id": os.environ["RUN_ID"],
    "data_root": os.environ["DATA_ROOT"],
    "base_outer_root": os.environ["BASE_OUTER_ROOT"],
    "pred_root": os.environ["PRED_ROOT"],
    "model_root": os.environ["MODEL_ROOT"],
    "primary": {
        "name": "e90",
        "summary": os.environ["E90_SUMMARY_OUT"],
        "key_results": os.environ["E90_KEY_OUT"],
        "rule": {
            "donors": ["e79", "e87"],
            "donor_configs": {
                "e79": {
                    "model_name": "microsoft/deberta-v3-base",
                    "text_preset": "essay_head_tail",
                    "max_length": 320,
                    "epochs": 5,
                    "learning_rate": 3e-5,
                    "loss_mode": "ce_ordinal_weighted",
                    "ordinal_loss_weight": 0.20,
                    "class_weights": "1.05,1.45,0.92,1.15",
                },
                "e87": {
                    "model_name": "bert-base-uncased",
                    "text_preset": "essay_head_tail",
                    "max_length": 320,
                    "epochs": 5,
                    "learning_rate": 3e-5,
                    "loss_mode": "ce_ordinal_weighted",
                    "ordinal_loss_weight": 0.20,
                    "class_weights": "1.05,1.45,0.92,1.15",
                },
            },
            "merge_rule": {
                "tag": "all_both_both_dir_up0m0_dn0m0_step3",
                "all_lengths": True,
                "up_enabled": True,
                "down_enabled": True,
                "agreement": "both donors must predict the same direction relative to base ideas",
                "prob_threshold": 0.0,
                "margin_threshold": 0.0,
                "max_step": 3,
            },
        },
    },
    "outer_folds": ["outer0", "outer1", "outer2", "outer3", "outer4"],
    "seeds": [20260325, 20260412],
}
Path(os.environ["MANIFEST_OUT"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"manifest_saved: {os.environ['MANIFEST_OUT']}")
PY

echo "[e90-outer] finished sealed outer confirmation"
echo "primary summary: $E90_SUMMARY_OUT"
echo "primary key results: $E90_KEY_OUT"
