#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-grid}"   # prepare | inner_cell | inner_grid | outer_cell | outer_grid | summary | grid
shift || true

TARGET_OUTER=""
TARGET_SPLIT=""
TARGET_SEED=""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/rcy/miniconda3/envs/llm/bin/python"
GPU="${GPU:-0}"
IDEAS_TEXT_PRESET="${IDEAS_TEXT_PRESET:-essay_only}"
IDEAS_MAX_LENGTH="${IDEAS_MAX_LENGTH:-256}"
IDEAS_EPOCHS="${IDEAS_EPOCHS:-4}"
IDEAS_LR="${IDEAS_LR:-2e-5}"
IDEAS_WEIGHT_DECAY="${IDEAS_WEIGHT_DECAY:-0.01}"
ORG_TEXT_PRESET="${ORG_TEXT_PRESET:-essay_plus_trait_reasoning}"
ORG_MAX_LENGTH="${ORG_MAX_LENGTH:-256}"
ORG_EPOCHS="${ORG_EPOCHS:-4}"
ORG_LR="${ORG_LR:-2e-5}"
ORG_WEIGHT_DECAY="${ORG_WEIGHT_DECAY:-0.01}"
ALLOW_OUTER_EVAL="${ALLOW_OUTER_EVAL:-0}"

if [[ "$MODE" == "inner_cell" && $# -ge 3 ]]; then
  TARGET_OUTER="$1"
  TARGET_SPLIT="$2"
  TARGET_SEED="$3"
  shift 3
elif [[ "$MODE" == "outer_cell" && $# -ge 2 ]]; then
  TARGET_OUTER="$1"
  TARGET_SEED="$2"
  shift 2
fi

TRAIN_SCRIPT="$ROOT/train_sidecar_bert.py"
RUNNER="$ROOT/run_sidecar_prod.py"
BUILD_OUTER_SCRIPT="$ROOT/build_nested_outer_folds.py"
BUILD_INNER_SCRIPT="$ROOT/build_nested_inner_kfold.py"
COLLECT_SCRIPT="$ROOT/collect_outer_metrics.py"

FULL_CACHE="$ROOT/prompt7_full_cache_mainline_final.json"
DATA_ROOT="$ROOT/sidecar_data/v10_nested_protocol_iA_orgbest_mainline"
PRED_ROOT="$ROOT/sidecar_preds/v10_nested_protocol_iA_orgbest_mainline"
MODEL_ROOT="$ROOT/sidecar_models/v10_nested_protocol_iA_orgbest_mainline"
SUMMARY_JSON="$PRED_ROOT/v10ncv_summary.json"

IDEAS_VARIANT="iA_devbest"
ORG_VARIANT="orgbest"
STRATIFY_BY="joint"
NUM_OUTERS=5
NUM_INNERS=3
OUTER_SEED=20260420
INNER_SEED=20260421
REFIT_SEED=20260422
INNER_DEV_RATIO=0.10
REFIT_TRAIN_RATIO=0.90
REFIT_DEV_RATIO=0.10

SEEDS=(20260325 20260412 20260413 20260414 20260415)
OUTERS=(outer0 outer1 outer2 outer3 outer4)
SPLITS=(s0 s1 s2)
INNER_BASE_SEEDS=(20260322 20260412 20260413)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --pred-root)
      PRED_ROOT="$2"
      shift 2
      ;;
    --model-root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --summary-json)
      SUMMARY_JSON="$2"
      shift 2
      ;;
    --ideas-variant)
      IDEAS_VARIANT="$2"
      shift 2
      ;;
    --org-variant)
      ORG_VARIANT="$2"
      shift 2
      ;;
    --stratify-by)
      STRATIFY_BY="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

require_path() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
}

require_outer_unlocked() {
  if [[ "$ALLOW_OUTER_EVAL" != "1" ]]; then
    cat >&2 <<'EOF'
[blocked] outer evaluation is locked.
This run only allows inner_* modes by default to avoid outer-based tuning.
If you really need a final outer check, re-run with: ALLOW_OUTER_EVAL=1
EOF
    exit 2
  fi
}

model_artifact_ready() {
  local trait_dir="$1"
  [[ -f "$trait_dir/config.json" ]] || return 1
  [[ -f "$trait_dir/pytorch_model.bin" ]] || return 1
  [[ -f "$trait_dir/tokenizer.json" ]] || return 1
  [[ -f "$trait_dir/tokenizer_config.json" ]] || return 1
  [[ -f "$trait_dir/vocab.txt" ]] || return 1
  return 0
}

outer_dir_of() {
  local outer="$1"
  echo "$DATA_ROOT/$outer"
}

inner_anchor_root_of() {
  local outer="$1"
  echo "$MODEL_ROOT/$outer/inner_anchors"
}

inner_run_root_of() {
  local outer="$1"
  echo "$MODEL_ROOT/$outer/inner_runs"
}

refit_anchor_root_of() {
  local outer="$1"
  echo "$MODEL_ROOT/$outer/refit_anchors"
}

refit_run_root_of() {
  local outer="$1"
  echo "$MODEL_ROOT/$outer/refit_runs"
}

split_seed_index_of() {
  local split="$1"
  case "$split" in
    s0) echo 0 ;;
    s1) echo 1 ;;
    s2) echo 2 ;;
    *) echo "Unknown split: $split" >&2; exit 1 ;;
  esac
}

outer_index_of() {
  local outer="$1"
  case "$outer" in
    outer0) echo 0 ;;
    outer1) echo 1 ;;
    outer2) echo 2 ;;
    outer3) echo 3 ;;
    outer4) echo 4 ;;
    *) echo "Unknown outer fold: $outer" >&2; exit 1 ;;
  esac
}

inner_anchor_seed_of() {
  local outer="$1"
  local split="$2"
  local outer_idx split_idx
  outer_idx="$(outer_index_of "$outer")"
  split_idx="$(split_seed_index_of "$split")"
  echo "$(( ${INNER_BASE_SEEDS[$split_idx]} + outer_idx * 100 ))"
}

refit_anchor_seed_of() {
  local outer="$1"
  local offset="${2:-0}"
  local outer_idx
  outer_idx="$(outer_index_of "$outer")"
  echo "$(( REFIT_SEED + outer_idx * 100 + offset ))"
}

ensure_nested_data() {
  if [[ -f "$DATA_ROOT/nested_inner_manifest.json" ]]; then
    echo "[nested] reuse existing nested data: $DATA_ROOT"
    return
  fi

  echo "[nested] building nested data under $DATA_ROOT"
  mkdir -p "$DATA_ROOT"
  "$PY" "$BUILD_OUTER_SCRIPT" \
    --input "$FULL_CACHE" \
    --output-dir "$DATA_ROOT" \
    --num-outer-folds "$NUM_OUTERS" \
    --outer-seed "$OUTER_SEED" \
    --stratify-by "$STRATIFY_BY" \
    --ideas-variant "$IDEAS_VARIANT" \
    --org-variant "$ORG_VARIANT"

  "$PY" "$BUILD_INNER_SCRIPT" \
    --outer-root "$DATA_ROOT" \
    --num-inner-folds "$NUM_INNERS" \
    --inner-seed "$INNER_SEED" \
    --refit-seed "$REFIT_SEED" \
    --stratify-by "$STRATIFY_BY" \
    --inner-dev-ratio "$INNER_DEV_RATIO" \
    --refit-train-ratio "$REFIT_TRAIN_RATIO" \
    --refit-dev-ratio "$REFIT_DEV_RATIO"
}

train_org_anchor() {
  local train_cache="$1"
  local dev_cache="$2"
  local out_dir="$3"
  local seed="$4"

  if model_artifact_ready "$out_dir/organization"; then
    return
  fi
  echo "[nested] rebuild org anchor -> $out_dir/organization"
  mkdir -p "$out_dir"
  "$PY" "$TRAIN_SCRIPT" \
    --train-cache "$train_cache" \
    --dev-cache "$dev_cache" \
    --output-dir "$out_dir" \
    --gpu "$GPU" \
    --model-name distilbert-base-uncased \
    --max-length "$ORG_MAX_LENGTH" \
    --epochs "$ORG_EPOCHS" \
    --batch-size 16 \
    --eval-batch-size 32 \
    --learning-rate "$ORG_LR" \
    --weight-decay "$ORG_WEIGHT_DECAY" \
    --seed "$seed" \
    --traits organization \
    --trait-text-presets "organization=$ORG_TEXT_PRESET"
}

train_style_anchor() {
  local train_cache="$1"
  local dev_cache="$2"
  local out_dir="$3"
  local seed="$4"

  if model_artifact_ready "$out_dir/style"; then
    return
  fi
  echo "[nested] rebuild style anchor -> $out_dir/style"
  mkdir -p "$out_dir"
  "$PY" "$TRAIN_SCRIPT" \
    --train-cache "$train_cache" \
    --dev-cache "$dev_cache" \
    --output-dir "$out_dir" \
    --gpu "$GPU" \
    --model-name distilbert-base-uncased \
    --max-length 256 \
    --epochs 4 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --seed "$seed" \
    --traits style \
    --trait-text-presets style=essay_only
}

train_conv_anchor() {
  local train_cache="$1"
  local dev_cache="$2"
  local out_dir="$3"
  local seed="$4"

  if model_artifact_ready "$out_dir/conventions"; then
    return
  fi
  echo "[nested] rebuild conventions anchor -> $out_dir/conventions"
  mkdir -p "$out_dir"
  "$PY" "$TRAIN_SCRIPT" \
    --train-cache "$train_cache" \
    --dev-cache "$dev_cache" \
    --output-dir "$out_dir" \
    --gpu "$GPU" \
    --model-name distilbert-base-uncased \
    --max-length 256 \
    --epochs 4 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --seed "$seed" \
    --traits conventions \
    --trait-text-presets conventions=essay_only
}

train_ideas_model() {
  local train_cache="$1"
  local dev_cache="$2"
  local out_dir="$3"
  local seed="$4"

  if model_artifact_ready "$out_dir/ideas"; then
    return
  fi
  echo "[nested] rebuild ideas model -> $out_dir/ideas"
  mkdir -p "$out_dir"
  "$PY" "$TRAIN_SCRIPT" \
    --train-cache "$train_cache" \
    --dev-cache "$dev_cache" \
    --output-dir "$out_dir" \
    --gpu "$GPU" \
    --model-name distilbert-base-uncased \
    --max-length "$IDEAS_MAX_LENGTH" \
    --epochs "$IDEAS_EPOCHS" \
    --batch-size 16 \
    --eval-batch-size 32 \
    --learning-rate "$IDEAS_LR" \
    --weight-decay "$IDEAS_WEIGHT_DECAY" \
    --seed "$seed" \
    --traits ideas \
    --trait-text-presets "ideas=$IDEAS_TEXT_PRESET" \
    --loss-mode ce_ordinal_weighted \
    --ordinal-loss-weight 0.20 \
    --class-weights 1.05,1.45,0.92,1.15
}

run_sidecar_eval() {
  local input_cache="$1"
  local model_dir="$2"
  local output_dir="$3"
  local prefix="$4"
  local style_dir="$5"
  local conv_a_dir="$6"
  local conv_b_dir="$7"
  local extra_flag="${8:-}"

  mkdir -p "$output_dir"

  local extra_args=()
  if [[ -n "$extra_flag" ]]; then
    extra_args+=("$extra_flag")
  fi

  "$PY" "$RUNNER" \
    --input-cache "$input_cache" \
    --model-dir "$model_dir" \
    --output-dir "$output_dir" \
    --prefix "$prefix" \
    --python-bin "$PY" \
    --gpu "$GPU" \
    "${extra_args[@]}" \
    --primary-gate gA \
    --fallback-gate gE \
    --org-max-prob 0.60 \
    --org-margin 0.10 \
    --org-max-replace-ratio 0.24 \
    --min-overall-advantage 0.0 \
    --enable-style-sidecar \
    --style-model-dir "$style_dir" \
    --style-max-prob 0.75 \
    --style-margin 0.20 \
    --style-min-delta 1 \
    --style-max-replace-ratio 0.10 \
    --style-replace-limit 0.12 \
    --style-min-overall-advantage 0.0 \
    --style-min-style-advantage 0.0 \
    --style-allow-upward \
    --enable-conv-consensus \
    --conv-model-a-dir "$conv_a_dir" \
    --conv-model-b-dir "$conv_b_dir" \
    --conv-max-prob 0.72 \
    --conv-margin 0.16 \
    --conv-min-delta 1 \
    --conv-max-replace-ratio 0.08 \
    --conv-boundary-mode 23_down \
    --conv-replace-limit 0.12 \
    --conv-min-overall-advantage 0.0 \
    --conv-min-conventions-advantage 0.0 \
    --no-conv-allow-upward
}

ensure_inner_anchors() {
  local outer="$1"
  local split="$2"
  local outer_dir train_cache dev_cache seed anchor_root
  outer_dir="$(outer_dir_of "$outer")"
  train_cache="$outer_dir/$split/sidecar_train.json"
  dev_cache="$outer_dir/$split/sidecar_dev.json"
  seed="$(inner_anchor_seed_of "$outer" "$split")"
  anchor_root="$(inner_anchor_root_of "$outer")/$split"

  require_path "$train_cache"
  require_path "$dev_cache"
  train_org_anchor "$train_cache" "$dev_cache" "$anchor_root/org" "$seed"
  train_style_anchor "$train_cache" "$dev_cache" "$anchor_root/style" "$seed"
  train_conv_anchor "$train_cache" "$dev_cache" "$anchor_root/conv_a" "$seed"
  train_conv_anchor "$train_cache" "$dev_cache" "$anchor_root/conv_b" "$((seed + 1))"
}

ensure_refit_anchors() {
  local outer="$1"
  local outer_dir train_cache dev_cache anchor_root
  outer_dir="$(outer_dir_of "$outer")"
  train_cache="$outer_dir/refit/sidecar_train.json"
  dev_cache="$outer_dir/refit/sidecar_dev.json"
  anchor_root="$(refit_anchor_root_of "$outer")"

  require_path "$train_cache"
  require_path "$dev_cache"
  train_org_anchor "$train_cache" "$dev_cache" "$anchor_root/org" "$(refit_anchor_seed_of "$outer" 0)"
  train_style_anchor "$train_cache" "$dev_cache" "$anchor_root/style" "$(refit_anchor_seed_of "$outer" 1)"
  train_conv_anchor "$train_cache" "$dev_cache" "$anchor_root/conv_a" "$(refit_anchor_seed_of "$outer" 2)"
  train_conv_anchor "$train_cache" "$dev_cache" "$anchor_root/conv_b" "$(refit_anchor_seed_of "$outer" 3)"
}

run_inner_cell() {
  local outer="$1"
  local split="$2"
  local seed="$3"

  echo "[nested] inner_cell outer=$outer split=$split seed=$seed"
  ensure_inner_anchors "$outer" "$split"

  local outer_dir model_dir out_dir anchor_root
  outer_dir="$(outer_dir_of "$outer")"
  anchor_root="$(inner_anchor_root_of "$outer")/$split"
  model_dir="$(inner_run_root_of "$outer")/$split/seed${seed}"
  out_dir="$PRED_ROOT/$outer/$split/seed${seed}"
  mkdir -p "$model_dir" "$out_dir"

  require_path "$outer_dir/$split/sidecar_train.json"
  require_path "$outer_dir/$split/sidecar_dev.json"
  require_path "$outer_dir/$split/sidecar_holdout.json"
  require_path "$anchor_root/org/organization"
  require_path "$anchor_root/style/style"
  require_path "$anchor_root/conv_a/conventions"
  require_path "$anchor_root/conv_b/conventions"

  if ! model_artifact_ready "$model_dir/organization"; then
    mkdir -p "$model_dir/organization"
    cp -af "$anchor_root/org/organization/." "$model_dir/organization/"
  fi

  train_ideas_model \
    "$outer_dir/$split/sidecar_train.json" \
    "$outer_dir/$split/sidecar_dev.json" \
    "$model_dir" \
    "$seed"

  local dev_prefix="v10ncv_${outer}_${split}_seed${seed}_dev"
  local holdout_prefix="v10ncv_${outer}_${split}_seed${seed}_holdout"

  if [[ ! -f "$out_dir/${dev_prefix}_selected_qwk.json" ]]; then
    run_sidecar_eval \
      "$outer_dir/$split/sidecar_dev.json" \
      "$model_dir" \
      "$out_dir" \
      "$dev_prefix" \
      "$anchor_root/style" \
      "$anchor_root/conv_a" \
      "$anchor_root/conv_b"
  fi

  if [[ ! -f "$out_dir/${holdout_prefix}_selected_qwk.json" ]]; then
    run_sidecar_eval \
      "$outer_dir/$split/sidecar_holdout.json" \
      "$model_dir" \
      "$out_dir" \
      "$holdout_prefix" \
      "$anchor_root/style" \
      "$anchor_root/conv_a" \
      "$anchor_root/conv_b"
  fi
}

run_outer_cell() {
  local outer="$1"
  local seed="$2"

  echo "[nested] outer_cell outer=$outer seed=$seed"
  ensure_refit_anchors "$outer"

  local outer_dir model_dir out_dir anchor_root
  outer_dir="$(outer_dir_of "$outer")"
  anchor_root="$(refit_anchor_root_of "$outer")"
  model_dir="$(refit_run_root_of "$outer")/seed${seed}"
  out_dir="$PRED_ROOT/$outer/refit/seed${seed}"
  mkdir -p "$model_dir" "$out_dir"

  require_path "$outer_dir/refit/sidecar_train.json"
  require_path "$outer_dir/refit/sidecar_dev.json"
  require_path "$outer_dir/sidecar_final_test.json"
  require_path "$anchor_root/org/organization"
  require_path "$anchor_root/style/style"
  require_path "$anchor_root/conv_a/conventions"
  require_path "$anchor_root/conv_b/conventions"

  if ! model_artifact_ready "$model_dir/organization"; then
    mkdir -p "$model_dir/organization"
    cp -af "$anchor_root/org/organization/." "$model_dir/organization/"
  fi

  train_ideas_model \
    "$outer_dir/refit/sidecar_train.json" \
    "$outer_dir/refit/sidecar_dev.json" \
    "$model_dir" \
    "$seed"

  local outer_prefix="v10ncv_${outer}_seed${seed}_outer"
  if [[ ! -f "$out_dir/${outer_prefix}_selected_qwk.json" ]]; then
    run_sidecar_eval \
      "$outer_dir/sidecar_final_test.json" \
      "$model_dir" \
      "$out_dir" \
      "$outer_prefix" \
      "$anchor_root/style" \
      "$anchor_root/conv_a" \
      "$anchor_root/conv_b" \
      "--no-selection-use-gold"
  fi
}

run_inner_grid() {
  ensure_nested_data
  local outer split seed
  for outer in "${OUTERS[@]}"; do
    require_path "$(outer_dir_of "$outer")/sidecar_protocol_manifest.json"
    for split in "${SPLITS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_inner_cell "$outer" "$split" "$seed"
      done
    done
  done
}

run_outer_grid() {
  ensure_nested_data
  local outer seed
  for outer in "${OUTERS[@]}"; do
    require_path "$(outer_dir_of "$outer")/refit/sidecar_train.json"
    for seed in "${SEEDS[@]}"; do
      run_outer_cell "$outer" "$seed"
    done
  done
}

run_summary() {
  echo "[nested] collect summary -> $SUMMARY_JSON"
  require_path "$PRED_ROOT"
  local allow_outer_args=()
  if [[ "$ALLOW_OUTER_EVAL" == "1" ]]; then
    allow_outer_args+=(--allow-outer)
  fi
  "$PY" "$COLLECT_SCRIPT" \
    --pred-root "$PRED_ROOT" \
    --outer-folds "$(IFS=,; echo "${OUTERS[*]}")" \
    --inner-splits "$(IFS=,; echo "${SPLITS[*]}")" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
    "${allow_outer_args[@]}" \
    --output "$SUMMARY_JSON"
}

echo "[nested] mode=$MODE root=$ROOT gpu=$GPU"

case "$MODE" in
  prepare)
    ensure_nested_data
    ;;
  inner_cell)
    if [[ -z "$TARGET_OUTER" || -z "$TARGET_SPLIT" || -z "$TARGET_SEED" ]]; then
      echo "Usage: $0 inner_cell <outer0..outer4> <s0|s1|s2> <seed>" >&2
      exit 1
    fi
    ensure_nested_data
    run_inner_cell "$TARGET_OUTER" "$TARGET_SPLIT" "$TARGET_SEED"
    ;;
  inner_grid)
    run_inner_grid
    ;;
  outer_cell)
    if [[ -z "$TARGET_OUTER" || -z "$TARGET_SEED" ]]; then
      echo "Usage: $0 outer_cell <outer0..outer4> <seed>" >&2
      exit 1
    fi
    require_outer_unlocked
    ensure_nested_data
    run_outer_cell "$TARGET_OUTER" "$TARGET_SEED"
    ;;
  outer_grid)
    require_outer_unlocked
    run_outer_grid
    ;;
  summary)
    run_summary
    ;;
  grid)
    require_outer_unlocked
    ensure_nested_data
    run_inner_grid
    run_outer_grid
    run_summary
    ;;
  *)
    echo "Usage: $0 [prepare|inner_cell|inner_grid|outer_cell|outer_grid|summary|grid]" >&2
    exit 1
    ;;
esac

echo "Done mode=$MODE"
echo "Data root: $DATA_ROOT"
echo "Pred root: $PRED_ROOT"
echo "Model root: $MODEL_ROOT"
echo "Summary: $SUMMARY_JSON"
echo "Ideas variant: $IDEAS_VARIANT"
echo "Org variant: $ORG_VARIANT"
echo "Ideas config: preset=$IDEAS_TEXT_PRESET max_length=$IDEAS_MAX_LENGTH epochs=$IDEAS_EPOCHS lr=$IDEAS_LR wd=$IDEAS_WEIGHT_DECAY"
echo "Org config: preset=$ORG_TEXT_PRESET max_length=$ORG_MAX_LENGTH epochs=$ORG_EPOCHS lr=$ORG_LR wd=$ORG_WEIGHT_DECAY"
echo "GPU: $GPU"
