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

The runner supports `--face-pooling masked` (corrected primary setting) and
`--face-pooling unmasked` (compatibility diagnostic). With masked pooling,
center-crop fallbacks are excluded according to `face_valid_mask`.
