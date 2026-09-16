# Hi-EF follow-up study: canonical method specification v0.3

Status: architecture frozen; loss hyperparameters not yet frozen. Test remains unopened.

## Research question

What incremental, correspondence-sensitive affective information does Party A
provide for forecasting Party B beyond what is already predictable from the
interaction context?

Split construction and release metadata are protocol limitations, not the
research gap. The study makes predictive, not causal, claims.

## Canonical architecture

Let `C` be clips I--II, `X_A` be clip III, and `E_B` be clip IV's emotion.

```text
Z_A_affect, Z_A_interaction = f_A(X_A)
s_C = h_C(C)
Delta_A = g(C, Z_A_affect, Z_A_interaction)
P(E_B | C, X_A) = softmax(s_C + Delta_A)
```

`Delta_A` is a class-logit correction, not a feature-space difference and not
an independently predicted emotion. The residual module is context-conditioned.
For null, missing, redundant, or unreliable A, the desired behavior is
`Delta_A approximately 0`, returning the model to its context prior.

`Z_A_affect` represents emotion, valence/arousal, intensity, expression,
prosody, and affective gesture. `Z_A_interaction` represents speech, intent,
gaze, action, affect target/direction, and A's relation to B. This is functional
specialization, not a requirement of statistical independence.

## Objective

```text
L = L_final
  + lambda_C L_context
  + lambda_E L_emotion
  + lambda_con L_supcon
  + lambda_null L_null
  - lambda_N L_nuisance
```

The implementation uses gradient reversal for nuisance prediction, so its
reported scalar cross-entropy is added while its encoder gradient is reversed.
Conditional contrastive pairs use same-emotion/different-source positives and
different-emotion/same-source negatives. Source folder is only an anonymized
proxy; claims about scene or identity invariance are not permitted.

The discussion contains both KL directions for `L_null`. The runner therefore
requires the direction to be explicitly frozen as `context-to-null`,
`null-to-context`, or `symmetric`; it does not choose one silently.

## Required ablations

1. `context`: `s_C` only and `Delta_A = 0`.
2. `affect`: `g(C, Z_A_affect)`.
3. `interaction`: `g(C, Z_A_interaction)`.
4. `both`: `g(C, Z_A_affect, Z_A_interaction)`.

Every non-context checkpoint exposes both its context-prior prediction and its
updated prediction, enabling paired same-model attribution.

## Frozen evidence motivating the method

- Raw A versus context: mean validation UAR delta `+0.0127`, positive in 5/5
  seeds but uncertain under source-folder resampling.
- Oracle `E_A` versus context: mean validation UAR delta `+0.0452`.
- Oracle `E_A` versus raw A: mean validation UAR delta `+0.0325`.
- Same-checkpoint raw-A and label interventions establish correspondence
  sensitivity on validation.

These results motivate the architecture. They are not evidence that the new
architecture works.

The earlier `full_aux`, `affective_bottleneck`, and `hybrid_residual` matrix is
retained as exploratory evidence only. None is the canonical architecture in
this specification.

## Items that must be frozen before full training

- `lambda_C`, `lambda_E`, `lambda_con`, `lambda_null`, and `lambda_N`;
- contrastive temperature;
- null-divergence direction;
- whether the context branch is jointly trained throughout or pretrained and
  then frozen (an ablation discussed but not decided).

No full multi-seed run or test evaluation may begin until these choices are
recorded in this specification.
