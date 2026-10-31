#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/rcy/rcy/LLMs-AES-IELTS}"
PY="${PY:-/home/rcy/miniconda3/envs/llm/bin/python}"

DATA_ROOT="${DATA_ROOT:-$ROOT/sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424}"

# Reuse immutable source outer artifacts from the prior sealed source run.
# These are raw outer/refit artifacts; this script does not select parameters from them.
SOURCE_PRED_ROOT="${SOURCE_PRED_ROOT:-$ROOT/sidecar_preds/v10ncv_clean_e30_outer_source_default}"
SOURCE_MODEL_ROOT="${SOURCE_MODEL_ROOT:-$ROOT/sidecar_models/v10ncv_clean_e30_outer_source_default}"

RUN_ID="${RUN_ID:-v10ncv_clean_outer_e59_sealed_20260427}"
PRED_ROOT="${PRED_ROOT:-$ROOT/sidecar_preds/$RUN_ID}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/sidecar_models/$RUN_ID}"
LOG_ROOT="$PRED_ROOT/logs"

E31_ROOT="$PRED_ROOT/e31_outer"
E35_ROOT="$PRED_ROOT/e35_org_struct_outer"
E36_ROOT="$PRED_ROOT/e36_org_overlay_outer"
E38_ROOT="$PRED_ROOT/e38_conv_struct_outer"
E48_DONOR_ROOT="$PRED_ROOT/e48_org_highscore_donor_outer"
E40_DONOR_ROOT="$PRED_ROOT/e40_orgv3_donor_outer"
E59_ROOT="$PRED_ROOT/e59_final_outer"
SUMMARY_OUT="$PRED_ROOT/e59_outer_summary.json"
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

gpu_for_index() {
  local idx="$1"
  echo "${GPU_LIST[$(( idx % ${#GPU_LIST[@]} ))]}"
}

outer_index_of() {
  case "$1" in
    outer0) echo 0 ;;
    outer1) echo 1 ;;
    outer2) echo 2 ;;
    outer3) echo 3 ;;
    outer4) echo 4 ;;
    *) echo "Unknown outer fold: $1" >&2; exit 1 ;;
  esac
}

refit_seed_of() {
  local outer="$1"
  local offset="$2"
  echo "$(( 20260422 + $(outer_index_of "$outer") * 100 + offset ))"
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
  [[ -f "$trait_dir/tokenizer.json" ]] || return 1
  [[ -f "$trait_dir/tokenizer_config.json" ]] || return 1
  [[ -f "$trait_dir/vocab.txt" ]] || return 1
  return 0
}

wait_for_slot() {
  while [[ "${#PIDS[@]}" -ge "$MAX_JOBS" ]]; do
    local pid="${PIDS[0]}"
    if ! wait "$pid"; then
      echo "[e59-outer] background job failed" >&2
      exit 1
    fi
    PIDS=("${PIDS[@]:1}")
  done
}

wait_all_jobs() {
  local status=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  PIDS=()
  return "$status"
}

check_source_complete() {
  local missing=0
  local outer seed prefix source_dir
  for outer in $OUTER_FOLDS; do
    for seed in $SEEDS; do
      prefix="v10ncv_${outer}_seed${seed}_outer"
      source_dir="$SOURCE_PRED_ROOT/$outer/refit/seed${seed}"
      for suffix in _with_bert.json _style_with_bert.json _conv_a_with_bert.json _conv_b_with_bert.json _selected_qwk.json _baseline_qwk.json; do
        if [[ ! -f "$source_dir/${prefix}${suffix}" ]]; then
          echo "Missing source artifact: $source_dir/${prefix}${suffix}" >&2
          missing=1
        fi
      done
    done
  done
  if [[ "$missing" != "0" ]]; then
    exit 1
  fi
}

train_trait() {
  local train_cache="$1"
  local dev_cache="$2"
  local out_dir="$3"
  local trait="$4"
  local preset="$5"
  local max_length="$6"
  local seed="$7"
  local gpu="$8"
  shift 8
  local trait_dir="$out_dir/$trait"

  require_path "$train_cache"
  require_path "$dev_cache"
  if model_ready "$trait_dir"; then
    echo "[e59-outer] reuse $trait model $out_dir"
    return
  fi
  mkdir -p "$out_dir"
  "$PY" "$ROOT/train_sidecar_bert.py" \
    --train-cache "$train_cache" \
    --dev-cache "$dev_cache" \
    --output-dir "$out_dir" \
    --gpu "$gpu" \
    --model-name distilbert-base-uncased \
    --max-length "$max_length" \
    --epochs 4 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --seed "$seed" \
    --traits "$trait" \
    --trait-text-presets "$trait=$preset" \
    "$@"
}

train_refit_models_for_outer() {
  local outer="$1"
  local gpu="$2"
  local train_cache="$DATA_ROOT/$outer/refit/sidecar_train.json"
  local dev_cache="$DATA_ROOT/$outer/refit/sidecar_dev.json"

  echo "[e59-outer] train/refit support models outer=$outer gpu=$gpu"
  train_trait "$train_cache" "$dev_cache" "$MODEL_ROOT/e35/$outer/refit/org" organization essay_plus_org_struct_v2_short 320 "$(refit_seed_of "$outer" 10)" "$gpu"
  train_trait "$train_cache" "$dev_cache" "$MODEL_ROOT/e38/$outer/refit/conv_struct" conventions essay_plus_conv_struct_v1 320 "$(refit_seed_of "$outer" 20)" "$gpu"
  train_trait "$train_cache" "$dev_cache" "$MODEL_ROOT/e48/$outer/refit/org_highscore" organization essay_plus_org_struct_v2_short 320 "$(refit_seed_of "$outer" 30)" "$gpu" \
    --loss-mode ce_ordinal_weighted \
    --ordinal-loss-weight 0.20 \
    --class-weights 1.00,1.00,1.00,1.35
  train_trait "$train_cache" "$dev_cache" "$MODEL_ROOT/e40/$outer/refit/org" organization essay_plus_org_struct_v3_compact 320 "$(refit_seed_of "$outer" 40)" "$gpu"
}

replay_e31_outer() {
  local outer="$1"
  local seed="$2"
  local prefix="v10ncv_${outer}_seed${seed}_outer"
  local source_dir="$SOURCE_PRED_ROOT/$outer/refit/seed${seed}"
  local out_dir="$E31_ROOT/$outer/refit/seed${seed}"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/${prefix}_selected_qwk.json" ]]; then
    return
  fi
  "$PY" "$ROOT/replay_mainline_from_artifacts.py" \
    --source-dir "$source_dir" \
    --source-prefix "$prefix" \
    --output-dir "$out_dir" \
    --output-prefix "$prefix" \
    --python-bin "$PY" \
    --no-selection-use-gold \
    --ideas-margin-threshold 0.19 \
    --ideas-replace-limit 1.0 \
    --ideas-max-replace-ratio 0.42 \
    --primary-gate gA \
    --fallback-gate gE \
    --min-overall-advantage 0.005 \
    --org-max-prob 0.60 \
    --org-margin 0.10 \
    --org-max-replace-ratio 0.24 \
    --style-max-prob 0.65 \
    --style-margin 0.10 \
    --style-max-replace-ratio 0.22 \
    --style-replace-limit 0.22 \
    --conv-max-prob 0.67 \
    --conv-margin 0.11 \
    --conv-max-replace-ratio 0.13 \
    --conv-boundary-mode 23_down \
    --conv-replace-limit 0.13
}

build_e35_outer() {
  local outer="$1"
  local seed="$2"
  local gpu="$3"
  local prefix="v10ncv_${outer}_seed${seed}_outer"
  local input_cache="$DATA_ROOT/$outer/sidecar_final_test.json"
  local source_dir="$SOURCE_PRED_ROOT/$outer/refit/seed${seed}"
  local model_dir="$MODEL_ROOT/e35/$outer/refit/org"
  local org_only_dir="$E35_ROOT/_org_only/$outer/refit/seed${seed}"
  local staged_dir="$E35_ROOT/_source_org_struct_v2_short/$outer/refit/seed${seed}"
  local out_dir="$E35_ROOT/$outer/refit/seed${seed}"
  mkdir -p "$org_only_dir" "$staged_dir" "$out_dir"
  if [[ ! -f "$org_only_dir/${prefix}_org_struct_with_bert.json" ]]; then
    "$PY" "$ROOT/infer_sidecar_bert.py" \
      --input-cache "$input_cache" \
      --model-dir "$model_dir" \
      --output "$org_only_dir/${prefix}_org_struct_with_bert.json" \
      --traits organization \
      --batch-size 32 \
      --max-length 320 \
      --gpu "$gpu"
  fi
  if [[ ! -f "$staged_dir/${prefix}_with_bert.json" ]]; then
    "$PY" "$ROOT/merge_org_only_into_with_bert.py" \
      --base-with-bert "$source_dir/${prefix}_with_bert.json" \
      --org-only-with-bert "$org_only_dir/${prefix}_org_struct_with_bert.json" \
      --output "$staged_dir/${prefix}_with_bert.json"
  fi
  cp -f "$source_dir/${prefix}_style_with_bert.json" "$staged_dir/${prefix}_style_with_bert.json"
  cp -f "$source_dir/${prefix}_conv_a_with_bert.json" "$staged_dir/${prefix}_conv_a_with_bert.json"
  cp -f "$source_dir/${prefix}_conv_b_with_bert.json" "$staged_dir/${prefix}_conv_b_with_bert.json"
  if [[ ! -f "$out_dir/${prefix}_selected_qwk.json" ]]; then
    "$PY" "$ROOT/replay_mainline_from_artifacts.py" \
      --source-dir "$staged_dir" \
      --source-prefix "$prefix" \
      --output-dir "$out_dir" \
      --output-prefix "$prefix" \
      --python-bin "$PY" \
      --no-selection-use-gold \
      --ideas-margin-threshold 0.19 \
      --ideas-replace-limit 1.0 \
      --ideas-max-replace-ratio 0.42 \
      --primary-gate gA \
      --fallback-gate gE \
      --min-overall-advantage 0.005 \
      --org-max-prob 0.60 \
      --org-margin 0.10 \
      --org-max-replace-ratio 0.24 \
      --style-max-prob 0.65 \
      --style-margin 0.10 \
      --style-max-replace-ratio 0.22 \
      --style-replace-limit 0.22 \
      --conv-max-prob 0.67 \
      --conv-margin 0.11 \
      --conv-max-replace-ratio 0.13 \
      --conv-boundary-mode 23_down \
      --conv-replace-limit 0.13
  fi
}

build_e36_outer() {
  local outer="$1"
  local seed="$2"
  local prefix="v10ncv_${outer}_seed${seed}_outer"
  local out_dir="$E36_ROOT/$outer/refit/seed${seed}"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/${prefix}_selected_qwk.json" ]]; then
    return
  fi
  "$PY" "$ROOT/overlay_trait_from_selected_preds.py" \
    --base-preds "$E31_ROOT/$outer/refit/seed${seed}/${prefix}_selected_preds.json" \
    --donor-preds "$E35_ROOT/$outer/refit/seed${seed}/${prefix}_selected_preds.json" \
    --output-preds "$out_dir/${prefix}_selected_preds.json" \
    --report-output "$out_dir/${prefix}_selected_report.json" \
    --trait organization \
    --name org_wle240_conf_p055_m030 \
    --word-le 240 \
    --donor-max-prob-ge 0.55 \
    --donor-margin-ge 0.30 \
    --direction changed
  "$PY" "$ROOT/eval_qwk.py" \
    --pred "$out_dir/${prefix}_selected_preds.json" \
    --output "$out_dir/${prefix}_selected_qwk.json" >/dev/null
}

build_e38_outer() {
  local outer="$1"
  local seed="$2"
  local gpu="$3"
  local prefix="v10ncv_${outer}_seed${seed}_outer"
  local input_cache="$DATA_ROOT/$outer/sidecar_final_test.json"
  local model_dir="$MODEL_ROOT/e38/$outer/refit/conv_struct"
  local source_dir="$SOURCE_PRED_ROOT/$outer/refit/seed${seed}"
  local stage_dir="$E38_ROOT/_conv_struct/$outer/refit/seed${seed}"
  local out_dir="$E38_ROOT/$outer/refit/seed${seed}"
  mkdir -p "$stage_dir" "$out_dir"
  if [[ ! -f "$stage_dir/${prefix}_conv_struct_with_bert.json" ]]; then
    "$PY" "$ROOT/infer_sidecar_bert.py" \
      --input-cache "$input_cache" \
      --model-dir "$model_dir" \
      --output "$stage_dir/${prefix}_conv_struct_with_bert.json" \
      --traits conventions \
      --batch-size 32 \
      --max-length 320 \
      --gpu "$gpu"
  fi
  if [[ ! -f "$out_dir/${prefix}_selected_qwk.json" ]]; then
    "$PY" "$ROOT/merge_conventions_consensus.py" \
      --base-input "$E36_ROOT/$outer/refit/seed${seed}/${prefix}_selected_preds.json" \
      --model-a-input "$stage_dir/${prefix}_conv_struct_with_bert.json" \
      --model-b-input "$source_dir/${prefix}_conv_b_with_bert.json" \
      --output "$out_dir/${prefix}_selected_preds.json" \
      --report-output "$out_dir/${prefix}_selected_report.json" \
      --max-prob 0.67 \
      --margin 0.11 \
      --max-replace-ratio 0.13 \
      --boundary-mode 23_bi \
      --no-distribution-guard \
      --allow-upward
    "$PY" "$ROOT/eval_qwk.py" \
      --pred "$out_dir/${prefix}_selected_preds.json" \
      --output "$out_dir/${prefix}_selected_qwk.json" >/dev/null
  fi
}

build_org_donor_outer() {
  local donor_name="$1"
  local model_dir="$2"
  local out_root="$3"
  local out_suffix="$4"
  local outer="$5"
  local seed="$6"
  local gpu="$7"
  local prefix="v10ncv_${outer}_seed${seed}_outer"
  local input_cache="$DATA_ROOT/$outer/sidecar_final_test.json"
  local out_dir="$out_root/$outer/refit/seed${seed}"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/${prefix}_${out_suffix}_with_bert.json" ]]; then
    return
  fi
  echo "[e59-outer] infer $donor_name outer=$outer seed=$seed gpu=$gpu"
  "$PY" "$ROOT/infer_sidecar_bert.py" \
    --input-cache "$input_cache" \
    --model-dir "$model_dir" \
    --output "$out_dir/${prefix}_${out_suffix}_with_bert.json" \
    --traits organization \
    --batch-size 32 \
    --max-length 320 \
    --gpu "$gpu"
}

run_outer_seed_cell() {
  local outer="$1"
  local seed="$2"
  local gpu="$3"
  echo "[e59-outer] build outer cell outer=$outer seed=$seed gpu=$gpu"
  replay_e31_outer "$outer" "$seed"
  build_e35_outer "$outer" "$seed" "$gpu"
  build_e36_outer "$outer" "$seed"
  build_e38_outer "$outer" "$seed" "$gpu"
  build_org_donor_outer e48 "$MODEL_ROOT/e48/$outer/refit/org_highscore" "$E48_DONOR_ROOT" org_highscore "$outer" "$seed" "$gpu"
  build_org_donor_outer e40 "$MODEL_ROOT/e40/$outer/refit/org" "$E40_DONOR_ROOT" orgv3 "$outer" "$seed" "$gpu"
}

echo "[e59-outer] sealed outer run; fixed e59 rule; no outer-based tuning"
echo "[e59-outer] source=$SOURCE_PRED_ROOT"
check_source_complete

echo "[e59-outer] phase 1/4 train fixed refit support models"
idx=0
for outer in $OUTER_FOLDS; do
  wait_for_slot
  train_refit_models_for_outer "$outer" "$(gpu_for_index "$idx")" >"$LOG_ROOT/train_${outer}.log" 2>&1 &
  PIDS+=("$!")
  idx=$((idx + 1))
done
wait_all_jobs

echo "[e59-outer] phase 2/4 build base and donor outer predictions"
idx=0
for outer in $OUTER_FOLDS; do
  for seed in $SEEDS; do
    wait_for_slot
    run_outer_seed_cell "$outer" "$seed" "$(gpu_for_index "$idx")" >"$LOG_ROOT/cell_${outer}_seed${seed}.log" 2>&1 &
    PIDS+=("$!")
    idx=$((idx + 1))
  done
done
wait_all_jobs

echo "[e59-outer] phase 3/4 apply fixed e59 consensus once"
"$PY" "$ROOT/apply_fixed_org_consensus_rule_outer.py" \
  --base-root "$E38_ROOT" \
  --donor-a-root "$E48_DONOR_ROOT" \
  --donor-b-root "$E40_DONOR_ROOT" \
  --reference-root "$E38_ROOT" \
  --output-root "$E59_ROOT" \
  --donor-a-name e48 \
  --donor-b-name e40_orgv3 \
  --donor-a-template "{stem}_org_highscore_with_bert.json" \
  --donor-b-template "{stem}_orgv3_with_bert.json" \
  --outer-folds "$(echo "$OUTER_FOLDS" | tr ' ' ',')" \
  --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
  --direction no_down \
  --prob-ge 0.50 \
  --prob-mode avg \
  --margin-ge 0.15 \
  --margin-mode avg \
  --word-filter all \
  --summary-output "$SUMMARY_OUT" \
  --scope-note "fixed e59 dev-selected organization consensus rule applied once on sealed outer_test; no outer-based tuning"

echo "[e59-outer] phase 4/4 write sealed manifest"
RUN_ID="$RUN_ID" DATA_ROOT="$DATA_ROOT" SOURCE_PRED_ROOT="$SOURCE_PRED_ROOT" SOURCE_MODEL_ROOT="$SOURCE_MODEL_ROOT" PRED_ROOT="$PRED_ROOT" MODEL_ROOT="$MODEL_ROOT" SUMMARY_OUT="$SUMMARY_OUT" MANIFEST_OUT="$MANIFEST_OUT" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "type": "sealed_outer_once",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": "Final fixed e59 outer_test evaluation. Do not use this result for tuning.",
    "outer_test_touched": True,
    "selection_on_outer": False,
    "run_id": os.environ["RUN_ID"],
    "data_root": os.environ["DATA_ROOT"],
    "source_pred_root": os.environ["SOURCE_PRED_ROOT"],
    "source_model_root": os.environ["SOURCE_MODEL_ROOT"],
    "pred_root": os.environ["PRED_ROOT"],
    "model_root": os.environ["MODEL_ROOT"],
    "summary": os.environ["SUMMARY_OUT"],
    "outer_folds": ["outer0", "outer1", "outer2", "outer3", "outer4"],
    "seeds": [20260325, 20260412],
    "fixed_rule": {
        "name": "e59",
        "base": "e38_conv_struct_outer",
        "donor_a": "e48_org_highscore",
        "donor_b": "e40_orgv3",
        "direction": "no_down",
        "prob_ge": 0.50,
        "prob_mode": "avg",
        "margin_ge": 0.15,
        "margin_mode": "avg",
        "word_filter": "all",
    },
}
Path(os.environ["MANIFEST_OUT"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"manifest_saved: {os.environ['MANIFEST_OUT']}")
PY

echo "[e59-outer] finished sealed outer; no tuning step follows"
echo "summary: $SUMMARY_OUT"
