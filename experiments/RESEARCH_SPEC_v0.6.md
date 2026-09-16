# Hi-EF follow-up study: post-audit diagnostic specification v0.6

Status: Phase-1 diagnostics frozen before execution. This stage is descriptive,
uses validation as development evidence, does not train or select a model, and
does not access test data.

## Questions

1. Does the residual improve macro/class-balanced behavior while degrading
   probabilistic fit through a small number of confident errors?
2. Are harmful corrections associated with larger residual norms?
3. Are corrections unstable for missing/weak face or audio observations?
4. Inside the same trained `both` checkpoint, what is contributed by the affect
   branch, interaction branch, and their combination?

## Frozen calibration diagnostics

For context and final logits of every variant and seed, report:

- UAR and WAR;
- micro-NLL and macro-NLL (equal mean over emotion-specific NLLs);
- multiclass Brier score;
- 15-bin expected calibration error;
- mean predictive entropy and confidence;
- mean confidence on correct and incorrect predictions;
- error rate with confidence at least 0.8;
- per-class recall, NLL, confidence, and sample count.

## Frozen residual diagnostics

For `affect`, `interaction`, and `both`, report per sample and aggregate:

- L2 and L-infinity norm of `Delta_A`;
- confidence, entropy, correct-class probability, and NLL changes;
- unchanged, beneficial, harmful, and wrong-to-wrong prediction flips;
- residual norms by flip category;
- rank correlation of residual norm with NLL and confidence changes;
- results by Party-A face-valid-frame count, audio availability, emotion, seed,
  and source folder.

## Frozen same-checkpoint interventions

For each of the five trained `both` checkpoints, evaluate the same validation
samples under:

1. `context`: bypass the residual;
2. `affect_only`: zero `Z_A_interaction`;
3. `interaction_only`: zero `Z_A_affect`;
4. `both`: use both representations.

No weights are changed. Report calibration metrics and paired comparisons:

- `affect_only - context`;
- `interaction_only - context`;
- `both - context`;
- `both - affect_only`;
- `both - interaction_only`.

For UAR and NLL deltas, run 5,000 diagnostic hierarchical bootstrap replicates
over model seed and validation source folder with fixed seeds 6901--6905.

## Interpretation constraints

This stage may identify a failure mode and motivate a predeclared revision. It
may not advance a model, reopen the failed v0.5 gate, or support confirmatory
claims. Validation has already selected checkpoints and informed this analysis.
The test partition remains sealed.
