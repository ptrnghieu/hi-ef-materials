# Hi-EF follow-up: train-only conditional-attribution audit v1.0

Status: Stage A is an exploratory, train-only audit. The released validation
and test partitions are not inputs to this procedure and must remain unopened.

## Purpose

The study asks a predictive (not causal) question:

> Under which observable interaction contexts does the actual Party-A clip
> provide more useful information for forecasting Party-B's future emotion than
> matched, non-corresponding Party-A clips?

This operationalizes the established research gap: existing integrated
multimodal forecasters do not make Party A's *incremental, context-dependent*
contribution measurable and accountable. Dataset split provenance and release
metadata are protocol limitations, not part of this gap.

## Data boundary

Only rows whose original manifest `split` equals `train` may be read. The audit
creates deterministic source-folder folds *inside that train partition*. A
fold's held-out rows are used only once, after fitting; they are not used for
early stopping, model selection, temperature fitting, or hyperparameter choice.

The output must record the SHA-256 of the source manifest, all source folders,
model/replacement seeds, fixed epoch count, and a declaration that validation
and test were not evaluated.

## Cross-fitted predictors

For every model seed and inner source-folder fold, fit two frozen-feature
baselines for a pre-specified fixed number of epochs:

1. context model: clips I--II;
2. direct-fusion model: clips I--III.

No learned residual, affect/inter-action decomposition, or gate is evaluated in
this stage. This is deliberately an attribution audit, not another architecture
search.

## Candidate Party-A inputs

Each held-out row is evaluated with its actual clip III, a zeroed clip III, and
the following deterministic replacement families drawn only from that inner
held-out fold:

1. global clip-III shuffle;
2. same-source, wrong-emotion replacement;
3. same-emotion, different-source replacement.

The replacement manifests must forbid all clips already present in the row.

## Outcomes and strata

For a candidate input `a`, its label-aware audit utility is

`U(a) = NLL(context) - NLL(a)`.

This quantity is an evaluation diagnostic, never an inference-time score.
The primary ranking outcome is the rate at which the actual Party-A clip has
higher utility than the mean utility of matched replacement clips. Ties receive
half credit. Secondary outcomes are UAR, NLL, prediction flips, beneficial
flips, and harmful flips relative to the same cross-fitted context prediction.

All results are reported overall and stratified by target emotion, Party-A
emotion, source folder, and an input-only context-confidence stratum. The last
stratum is fixed from cross-fitted context probabilities: low/medium/high
maximum class confidence using train-only 1/3 and 2/3 quantiles.

## Decision rule

This audit may identify a provisional H3 target only if a hierarchical bootstrap
resampling model seed and source folder has a strictly positive lower 95% bound
for the actual-A advantage over matched replacements in that pre-defined
stratum. It does not authorize a test evaluation or a new architecture by
itself. A null result means the next step is to report the limitation and
reformulate the method question; it is not evidence for causal irrelevance of
Party A.
