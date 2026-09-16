# Hi-EF follow-up study: Research Specification v0.2

Status: frozen for validation-only method development. The held-out test split
must remain unopened until the method and reporting protocol are frozen.

## 1. Scope and research gap

The task is future-emotion forecasting for Party B. Let `C` denote context
clips I--II, `A` denote Party A in clip III, `E_A` denote A's annotated emotion,
and `E_B` denote B's target emotion in clip IV.

The research gap is a construct-validity and signal-attribution gap: an
improvement obtained after adding clip III does not establish whether the model
has extracted affect-relevant information about A, or has instead used generic
scene, identity, semantic, or other nuisance information. Dataset-release and
split-documentation issues are methodological constraints, not the research gap.

The central question is:

> What incremental, correspondence-sensitive affective information does Party A
> provide for forecasting Party B beyond context, and can a model extract this
> information more effectively than symmetric raw-feature fusion?

This study concerns predictive contribution and representation. It does not
claim causal interpersonal influence.

## 2. Fixed data and evaluation protocol

- Dataset release: 2,830 MCIS rows in the released `sample.csv`.
- Split: frozen source-folder-held-out manifest, SHA-256
  `f333dde1b4cddea64c229dc00e94aa90df36324a27a6bc89fbe30b256bcbcd33`.
- Counts: 1,993 train, 428 validation, 409 test.
- Model seeds: `42, 123, 456, 789, 1024`.
- Replacement seeds: `1701` through `1720`.
- Primary task metric: UAR for `E_B`.
- Secondary metrics: NLL, WAR, correct-class probability, prediction flips, and
  per-class recall.
- Uncertainty diagnostic: 5,000-replicate hierarchical bootstrap over model
  seed and source folder; replacement manifest is an additional unit for
  intervention controls.
- Checkpoints are selected by validation UAR, breaking ties by validation NLL.
- No seed or checkpoint may be selected using test performance.

The source-folder prefix is an anonymized grouping proxy. Its exact semantics
remain unknown; this is reported as a limitation rather than a research claim.

## 3. Evidence frozen before method development

All values below are validation diagnostics.

| Comparison | Mean UAR delta | Hierarchical 95% interval | Status |
|---|---:|---:|---|
| Raw A minus context | +1.27 points | [-0.57, +3.33] | directional, uncertain |
| Oracle `E_A` minus context | +4.52 points | [+0.50, +8.09] | supported |
| Oracle `E_A` minus raw A | +3.25 points | [-1.33, +6.90] | directional by UAR |

Oracle `E_A` improves NLL over context with interval `[+0.094, +0.317]` and
over raw A with interval `[+0.018, +0.290]`. Therefore the representation gap
is supported distributionally, while its UAR magnitude remains uncertain.

Same-checkpoint interventions show that raw clip III is correspondence-sensitive:
zero, global shuffle, same-source/wrong-emotion, and same-emotion/different-source
controls all reduce validation UAR with positive bootstrap intervals. Oracle
label controls are stronger: true `E_A` exceeds global label shuffle by 12.98
points and wrong-emotion permutation by 16.36 points, with positive UAR and NLL
intervals.

## 4. Frozen hypotheses

- **H1 — Raw incremental contribution.** Raw A provides some predictive
  information beyond C. Current status: directional but uncertain under
  source-folder resampling.
- **H2 — Correspondence sensitivity.** Correctly corresponding A inputs
  outperform null or mismatched A inputs under a fixed checkpoint. Current
  status: supported on validation.
- **H3 — Affective oracle value.** `E_A` provides predictive information about
  `E_B` beyond C. Current status: supported on validation.
- **H4 — Representation gap.** Symmetric raw fusion does not fully extract or
  exploit the affect-relevant information represented by `E_A`. Current status:
  supported by NLL and directional by UAR.
- **H5 — Method hypothesis.** Explicit supervision and bottlenecking of A's
  affective state will improve future-emotion forecasting and correspondence
  sensitivity relative to symmetric raw fusion. Current status: untested.

## 5. Method-development matrix

All candidates reuse the frozen clip encoder, training schedule, split, and five
seeds. They receive clips I--III at inference and never receive oracle labels at
inference.

1. `full_aux`: symmetric raw fusion plus an auxiliary head predicting `E_A`.
2. `affective_bottleneck`: context plus a soft affect token obtained from the
   predicted distribution `q(E_A | X_A)`.
3. `hybrid_residual`: context plus the soft affect token and a context-residual
   A token. A learned context predictor estimates the context-predictable part
   of A; the remaining token carries information not reconstructed from C.

Fixed losses:

`L = CE(E_B) + 0.5 * balanced_CE(E_A)` for all candidates, with
`+ 0.1 * reconstruction + 0.05 * orthogonality` for `hybrid_residual`.

The auxiliary class weights are `N / (K * n_k)`, normalized to mean one and
computed from the training partition only. Future-emotion CE remains unweighted
to preserve comparability with the frozen baselines.

## 6. Candidate selection and stopping rule

The three candidates form one predeclared exploratory development matrix. Select
the candidate with the highest mean validation UAR across all five seeds. If
candidates differ by less than 0.5 UAR points, prefer lower mean validation NLL;
if still tied, prefer the model with fewer parameters.

A candidate advances only if either:

- it exceeds raw fusion by at least 1.0 mean UAR point and is positive in at
  least four of five paired seeds; or
- it has a positive hierarchical NLL interval versus raw fusion without losing
  more than 0.5 mean UAR points.

If no candidate advances, stop method development and report the attribution
findings without inventing a post-hoc architecture search.

## 7. Required checks before test evaluation

For the selected candidate, freeze and run same-checkpoint interventions on the
validation partition: zero A, global A shuffle, same-source/wrong-emotion, and
predicted-affect-token shuffle. Record full logits and fixed manifests.

Only after the selected method, losses, seeds, intervention algorithms, and
reporting tables are frozen may the five selected checkpoints be evaluated once
on the held-out test partition. Validation findings and final test findings must
be labeled separately.

## 8. Permitted claims

Permitted: predictive contribution, correspondence sensitivity, affective
oracle value, representation gap, and improved affect extraction if supported.

Not permitted: causal influence, psychological mechanism, conversation-level
generalization, identity invariance, or exact replication of the original Hi-EF
paper without the official metadata and split protocol.
