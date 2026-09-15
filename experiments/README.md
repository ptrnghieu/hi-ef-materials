# Source-folder-held-out experiments

This directory contains the frozen split protocol for the Hi-EF experiments.
It intentionally does **not** use the earlier component-safe split.

## Protocol

The prefix in each clip id is treated as an anonymized source-folder id. For
example, all MCIS samples containing clips under `01/...` stay in one split.
The released 53 source folders are allocated as 37 train, 8 validation, and 8
test folders. The assignment is optimized for MCIS count and clip-III/clip-IV
emotion balance.

The source-folder semantics are not stated in the public metadata. Evidence in
the preprocessing code suggests that the folders are episode-level sources,
but this remains a proxy until the dataset authors confirm it.

## Rebuild the frozen manifest

Run from the repository root:

```bash
python experiments/build_source_folder_split.py
```

Generated files:

- `experiments/manifests/source_folder_split_seed42.csv`
- `experiments/manifests/source_folder_split_seed42_audit.json`

The audit must report zero shared source folders, clips, and adjacent clip
pairs for every pair of splits.

## Run tests

```bash
PYTHONPATH=experiments python -m unittest experiments/test_source_folder_split.py
```

## Use on Kaggle

Open `experiments/kaggle_source_folder_split.ipynb` on Kaggle. Its bootstrap
cell clones the `experiments` branch, rebuilds the manifest deterministically,
and checks all leakage invariants. Kaggle Internet must be enabled for the
clone cell. Alternatively, upload the repository and change `REPO_DIR`.

Load the frozen assignment instead of independently recomputing a split inside
a training notebook:

```python
import pandas as pd

manifest = pd.read_csv(
    "/kaggle/working/hi-ef-materials/experiments/manifests/"
    "source_folder_split_seed42.csv"
)

train_rows = manifest[manifest["split"] == "train"]
val_rows = manifest[manifest["split"] == "val"]
test_rows = manifest[manifest["split"] == "test"]
```

Do not select seeds, checkpoints, or model variants using the test partition.
Select checkpoints using validation UAR and evaluate the frozen test partition
only after the experiment configuration is fixed.

## Audit frozen features on Kaggle

Before training, run the aggregate feature audit against clips I--III:

```bash
python experiments/feature_preflight.py \
  --manifest experiments/manifests/source_folder_split_seed42.csv \
  --features-dir /kaggle/input/datasets/ptrnghieu/hi-ef-features-v2 \
  --output /kaggle/working/feature_preflight.json
```

Missing/corrupt files, clip-id mismatches, unexpected shapes, non-finite values,
or malformed face masks fail the audit. Missing faces and audio are reported by
clip position and split as diagnostics because they may be valid properties of
the released data.

## Baseline smoke test

Run `experiments/kaggle_phase1_smoke.ipynb` with a Kaggle T4 GPU. It trains ten
batches per epoch for two epochs for both B1/context (clips I--II) and B2/full
(clips I--III). The smoke test validates training, checkpointing, and artifact
generation; its metrics are not research results. Test evaluation is disabled.

## Validation-only pilot

After the smoke test passes, run `experiments/kaggle_phase1_validation.ipynb`
with a Kaggle T4 GPU. It trains the context and full models on all training
samples for seed 42, selects checkpoints using validation only, and saves full
validation logits. The notebook deliberately leaves the test split unevaluated.

The seed-42 notebook above is a pipeline pilot, not the canonical multi-seed
result. The canonical run is `experiments/kaggle_phase1_multiseed_validation.ipynb`.
It invokes `experiments/run_validation_matrix.py` once to run the fixed seeds
`42, 123, 456, 789, 1024` for both context and full models. It writes a single
auditable JSON summary and CSV table, records code/data hashes, verifies every
run configuration, and rejects any run containing test metrics.

## Validation-only clip-III interventions

After saving the canonical multi-seed output, attach that notebook output and
the feature dataset to `experiments/kaggle_phase1_interventions.ipynb`. The
notebook uses the five frozen full-model checkpoints and evaluates true, zero,
global-shuffle, same-source/wrong-emotion, and same-emotion/different-source
clip-III conditions. Replacement controls use the 20 fixed seeds 1701--1720.
The runner reproduces each checkpoint's original validation UAR before running
controls, stores replacement manifests and logits, and never loads test rows.

Attach the saved `hief-interventions` output to
`experiments/kaggle_phase1_intervention_statistics.ipynb` for a CPU-only audit
of UAR, NLL, correct-class probability, prediction flips, and per-class recall.
The accompanying runner performs a fixed 5,000-replicate hierarchical bootstrap
over model seeds, source folders, and replacement manifests. These intervals
are explicitly diagnostic because validation selected the checkpoints and only
five model seeds and eight validation source folders are available.

## Party-A emotion-label oracle

Run `experiments/kaggle_phase1_label_oracle.ipynb` after attaching the feature
dataset and the saved `hief-multiseed-validation` output. This is the missing
construct-validity diagnostic: it compares context-only, context plus raw clip
III, and context plus the ground-truth emotion label of Party A under the same
source-folder split and five training seeds.

The notebook also evaluates the five frozen label-oracle checkpoints under 20
fixed global label shuffles and 20 fixed wrong-emotion permutations. Both
controls preserve the validation label distribution; the wrong-emotion control
guarantees that every Party-A label changes. The runner performs a forward and
backward preflight before training, stores all manifests and validation logits,
and never loads the test partition. Because it supplies a ground-truth input,
the label model is an oracle diagnostic rather than a deployable baseline.

Attach the saved baseline and label-oracle outputs to
`experiments/kaggle_phase1_label_oracle_statistics.ipynb`. This CPU-only job
computes paired UAR, NLL, correct-class probability, prediction flips, and
per-class recall for raw-versus-context, oracle-versus-context,
oracle-versus-raw, and both label controls. Its hierarchical bootstrap samples
model seeds and source folders for model comparisons, and additionally samples
replacement manifests for label controls. The resulting intervals remain
validation diagnostics rather than confirmatory test estimates.

The runner supports `--face-pooling masked` (corrected primary setting) and
`--face-pooling unmasked` (compatibility diagnostic). With masked pooling,
center-crop fallbacks are excluded according to `face_valid_mask`.
