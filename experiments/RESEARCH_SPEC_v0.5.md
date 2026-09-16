# Hi-EF follow-up study: canonical hierarchical audit specification v0.5

Status: audit comparisons, bootstrap procedure, and advancement gate frozen
after the descriptive v0.4 matrix and before running the hierarchical audit.
Test remains unopened. The architecture, objectives, and hypothesis remain
those in v0.4.

## Purpose

The validation matrix showed descriptive gains but cannot establish uncertainty
from five model seeds and eight source folders. This audit quantifies that
uncertainty. It does not retrain models, tune hyperparameters, or select the
largest validation mean.

## Frozen comparisons

Primary comparisons:

1. `both.final` versus the separately trained `context.final`, paired by model
   seed and validation sample.
2. `both.final` versus `both.context`, paired within the same checkpoint.

Secondary decomposition comparisons:

3. `affect.final` versus the separately trained `context.final`.
4. `interaction.final` versus the separately trained `context.final`.
5. `affect.final` versus `affect.context` within checkpoint.
6. `interaction.final` versus `interaction.context` within checkpoint.

Secondary complementarity comparisons:

7. `both.final` versus `affect.final`.
8. `both.final` versus `interaction.final`.
9. `interaction.final` versus `affect.final`.

Positive UAR delta means the first member is better. Positive NLL delta means
the first member has lower NLL.

## Frozen hierarchical bootstrap

- 5,000 bootstrap replicates per comparison.
- Fixed comparison-specific seeds `5901` through `5909` in the order above.
- Resampling units: model seed and validation source folder.
- Five model seeds are sampled with replacement.
- Eight source folders are sampled with replacement; a draw is rejected if it
  omits every example of any of the seven emotion classes.
- UAR and NLL are recomputed from aggregated folder-level sufficient statistics
  for every replicate.
- Report bootstrap mean, percentile 95% interval, and probability above zero.
- Also report observed per-seed UAR/NLL deltas, correct-class probability
  delta, prediction flips, harmful flips, beneficial flips, and per-class
  recall deltas.

These are validation uncertainty diagnostics, not confirmatory test estimates:
validation selected checkpoints, only five model seeds and eight folders are
available, and this audit specification was frozen after descriptive matrix
means were observed.

## Frozen advancement and claim rules

The canonical candidate remains `both`; the audit must not replace it merely
because another variant has a marginally higher validation statistic.

Advance `both` to exactly one frozen test evaluation only if the 95% bootstrap
lower bound for UAR delta is greater than zero in both primary comparisons:

```text
both.final > context.final
both.final > both.context
```

If either primary gate fails, do not evaluate test and revisit the method.

Claims about branch complementarity are separate from advancement:

- claim incremental affect beyond interaction only if the lower bound for
  `both.final - interaction.final` is greater than zero;
- claim incremental interaction beyond affect only if the lower bound for
  `both.final - affect.final` is greater than zero;
- otherwise report that the corresponding incremental contribution is not
  resolved on validation.

No test labels or test predictions may be loaded by the audit.
