# Multi-Trait Automated Essay Scoring

Source code for **Multi-Trait Automated Essay Scoring and Its Gated Trait-Level
Correction Method under Low-Resource Conditions**.

Qwen generates trait analyses for ASAP Prompt 7. Fine-tuned classifiers provide
candidate scores for the basic correction stage. Two further candidates,
DeBERTa and BERT, re-examine ideas. Their suggestions are adopted when they
agree on a nonzero correction direction. A fixed function computes overall
from the four trait scores.

## Files

| Location | Contents |
| --- | --- |
| Root `.py` and `.sh` files | Original scoring, training, correction and evaluation code |
| `trait_questions_prompt7/` | Four question lists used by the Qwen pipeline |
| `configs/prompt7.json` | Model, training, split, gate and mapping settings |
| `configs/prompt7_split_ids.json` | Essay-ID assignments for outer, inner and refit partitions |
| `requirements.txt` | Dependencies recorded from the project's Python 3.10 environment |
| `docs/REPRODUCTION.md` | Execution sequence, prerequisites and path settings |
| `docs/FILES_ZH.md` | File guide in Chinese |
| `docs/source_manifest.json` | Source file checksums and configuration sources |

All copied code files retain their original contents and relative layout.
The configuration JSON is a reference record; the original scripts use their
command-line arguments and constants rather than loading this JSON automatically.

## Data and environment

Download `training_set_rel3.tsv` from the
[official ASAP AES competition](https://www.kaggle.com/c/asap-aes/data) and place it
in `data/training_set_rel3.tsv`. This release contains 1,569 Prompt 7 essays.
The code uses `rater1_trait1` through `rater1_trait4` as trait targets and
`domain1_score` as the overall target.

```bash
conda create -n mtaes python=3.10 -y
conda activate mtaes
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

These dependency versions were read from the existing project environment.
Training uses NVIDIA GPUs; CUDA and PyTorch installation should match the host.

Download the Qwen checkpoint separately:

```bash
huggingface-cli download Qwen/Qwen-7B-Chat --local-dir models/Qwen-7B-Chat
```

The original Qwen pipeline loads this model locally and uses
`trust_remote_code=True`. Dataset and model access terms apply separately.

## Execution sequence

Read [reproduction instructions](docs/REPRODUCTION.md) before running shell scripts;
the copied scripts retain the original project's absolute paths.

1. Generate a Qwen reasoning cache using `run_prompt7_evaluation.py`.
2. Create the nested splits with `build_nested_outer_folds.py` and
   `build_nested_inner_kfold.py`.
3. Train the source classifiers with `run_v10_nested_pipeline.sh`.
4. Build the basic correction baseline with `run_e59_outer_sealed_once.sh`.
5. Train the final candidates and evaluate direction consistency with
   `run_e90_outer_sealed_once.sh`.

Internal identifiers e59, e79, e87 and e90 refer to the basic correction baseline,
DeBERTa ideas candidate, BERT ideas candidate and final correction module.

This package contains code and configurations. Essays, reasoning caches, trained
weights, per-essay predictions, figures and manuscript files are not included.
Fresh cache generation and training produce a new run; exact replay of an earlier
run requires its retained artifacts. This packaging step does not retrain models.
