# Running the Prompt 7 code

## Paths in the original scripts

Run from the root of a separate checkout containing your downloaded data.
Copied source files are unchanged. Before training, review these machine settings:

| File | Setting |
| --- | --- |
| `run_v10_nested_pipeline.sh` | `PY` is a literal path to the original Python environment; set it to your interpreter before using this script on another machine. `ROOT` follows the script location. |
| `run_e59_outer_sealed_once.sh` | Override `ROOT`, `PY`, `DATA_ROOT`, `SOURCE_PRED_ROOT` and `SOURCE_MODEL_ROOT` through environment variables. |
| `run_e90_outer_sealed_once.sh` | Override `ROOT`, `PY`, `DATA_ROOT` and `BASE_OUTER_ROOT` through environment variables. |
| `audit_e90_outer_cleanliness.py` | Its `ROOT` constant identifies the original project; review that path before using it. |

The `prepare` mode of `run_v10_nested_pipeline.sh` has older default partition
seeds. Use the explicit split commands below to match `configs/prompt7.json`.
The runner reuses those splits when `nested_inner_manifest.json` already exists.

## 1. Generate the initial cache

After obtaining the TSV and Qwen checkpoint as described in README:

```bash
CUDA_VISIBLE_DEVICES=0 python run_prompt7_evaluation.py \
  --no-auto-launch --debug-first-n 0 \
  --save-threshold-cache prompt7_full_cache_mainline_final.json
```

The cache contains essay text and reasoning and remains a local runtime artifact.
The four full prompt builders are in `trait_analysis_pipeline.py`, alongside the
JSON response checks. Their question lists are in `trait_questions_prompt7/`.

## 2. Prepare nested partitions

```bash
python build_nested_outer_folds.py \
  --input prompt7_full_cache_mainline_final.json \
  --output-dir sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424 \
  --num-outer-folds 5 --outer-seed 20260424 --stratify-by joint \
  --min-stratify-count 5 --ideas-variant iA_devbest --org-variant orgbest

python build_nested_inner_kfold.py \
  --outer-root sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424 \
  --num-inner-folds 3 --inner-seed 20260425 --refit-seed 20260426 \
  --stratify-by joint --min-stratify-count 5 \
  --inner-dev-ratio 0.10 --refit-train-ratio 0.90 --refit-dev-ratio 0.10
```

`configs/prompt7_split_ids.json` records the existing experimental partitions for
comparison. It contains essay IDs only, without texts, labels or predictions.
After configuration selection, the final models use a 90:10 train/development
division inside each outer training pool. Development data support checkpoint
selection and temperature calibration. Training seeds and partition seeds have
different roles.

## 3. Train source classifiers

After setting `PY` in `run_v10_nested_pipeline.sh` to your interpreter, run the
following commands. They process the ten seed-fold cells serially on GPU 0.
The named output folders belong to this checkout.

```bash
for outer in outer0 outer1 outer2 outer3 outer4; do
  for seed in 20260325 20260412; do
    ALLOW_OUTER_EVAL=1 GPU=0 \
    IDEAS_TEXT_PRESET=essay_sentblock IDEAS_MAX_LENGTH=256 \
    IDEAS_EPOCHS=5 IDEAS_LR=3e-5 \
    bash run_v10_nested_pipeline.sh outer_cell "$outer" "$seed" \
      --data-root "$PWD/sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424" \
      --pred-root "$PWD/sidecar_preds/v10ncv_clean_e30_outer_source_default" \
      --model-root "$PWD/sidecar_models/v10ncv_clean_e30_outer_source_default" \
      --summary-json "$PWD/sidecar_preds/v10ncv_clean_e30_outer_source_default/source_summary.json"
  done
done
```

The runner trains the initial trait classifiers and generates the predictions
used by the basic correction stage. It retains the original anchor training
seeds, separate from the partition seeds above.

## 4. Build the basic correction baseline

```bash
ROOT="$PWD" PY="$(command -v python)" GPUS=0 MAX_JOBS=1 \
DATA_ROOT="$PWD/sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424" \
SOURCE_PRED_ROOT="$PWD/sidecar_preds/v10ncv_clean_e30_outer_source_default" \
SOURCE_MODEL_ROOT="$PWD/sidecar_models/v10ncv_clean_e30_outer_source_default" \
bash run_e59_outer_sealed_once.sh
```

This step requires the source predictions from step 3. It trains structural
organization/conventions candidates and applies the baseline correction rules.

## 5. Train the final ideas candidates and evaluate

```bash
ROOT="$PWD" PY="$(command -v python)" GPUS=0 MAX_JOBS=1 \
DATA_ROOT="$PWD/sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424" \
BASE_OUTER_ROOT="$PWD/sidecar_preds/v10ncv_clean_outer_e59_sealed_20260427/e59_final_outer" \
bash run_e90_outer_sealed_once.sh
```

This step fine-tunes `microsoft/deberta-v3-base` and `bert-base-uncased`, applies
`apply_e90_e79_e87_outer.py`, and writes the final evaluation summaries. The fixed
configuration entry points do not repeat all previous model-selection searches.

## 6. Check the overall mapping

After the preceding runs, use:

```bash
python verify_overall_mapping.py \
  --data-root sidecar_data/v10_nested_protocol_iA_orgbest_mainline_clean_20260424 \
  --prediction-root sidecar_preds/v10ncv_clean_outer_e90_e79_e87_sealed_20260504/e90_primary_outer \
  --output reports/overall_mapping_verification.json
```

The script fits OLS within each outer training pool and compares integer overall
predictions with the retained fixed mapping. It writes a report without replacing
the original predictions.
