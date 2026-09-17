# Hi-EF follow-up study: reliability mechanism diagnostic specification v0.7

Status: frozen before execution. This is a validation-only, post-hoc mechanism
diagnostic. It does not train an encoder, change a checkpoint, select a model,
or access the test partition.

## Purpose

The v0.6 audit found a small UAR tendency but worse NLL, Brier score, and
overconfident-error rate after applying the Party-A residual. This diagnostic
separates three possible explanations:

1. global probability miscalibration;
2. a residual whose global magnitude is too large;
3. a need for sample-conditioned reliability.

## Inputs

Use the five frozen `both` validation checkpoints and their saved aligned
`context_logits`, `delta_logits`, and `final_logits`. Enforce

```math
s_{final}=s_C+\Delta_A.
```

All fitting below is cross-fitted over the eight validation source folders:
fit on seven folders and evaluate only on the held-out folder.

## Frozen diagnostics

### Temperature scaling

Choose one positive temperature per model seed and held-out folder by minimizing
training-fold micro-NLL over 1,201 log-spaced values in `[0.05, 20]`:

```math
s_T=s_{final}/T.
```

Temperature scaling cannot change the predicted class.

### Global residual shrinkage

Choose one residual multiplier per model seed and held-out folder by minimizing
training-fold micro-NLL over 501 equally spaced values in `[0, 1]`:

```math
s_\alpha=s_C+\alpha\Delta_A.
```

### Oracle upper bounds

Two label-informed diagnostics quantify headroom and are not deployable models:

- NLL oracle: choose context or final logits per sample according to lower
  true-class NLL;
- accuracy oracle: choose the correct alternative when exactly one is correct,
  then break equal-correctness cases using lower true-class NLL.

## Outputs and uncertainty

Report UAR, WAR, micro/macro-NLL, Brier score, ECE-15, entropy, confidence,
overconfident errors, per-class metrics, cross-fitted parameters, and oracle
choice rates.

Run 5,000 fixed-prediction hierarchical bootstrap replicates over model seed and
source folder for:

- temperature versus final;
- shrinkage versus final;
- shrinkage versus context;
- NLL oracle versus final and shrinkage;
- accuracy oracle versus final and shrinkage.

Bootstrap seeds are fixed to 7901--7907. The bootstrap does not refit fold
parameters and remains a validation diagnostic.

## Decision interpretation

- temperature improvement isolates global calibration error;
- shrinkage improvement with UAR non-inferiority suggests excessive global
  residual magnitude;
- oracle headroom beyond shrinkage motivates a sample-conditioned reliability
  mechanism.

The UAR non-inferiority margin for shrinkage versus final is fixed at -0.005.
These indicators guide the next predeclared training experiment; they do not
advance a model or reopen the failed v0.5 test gate.
