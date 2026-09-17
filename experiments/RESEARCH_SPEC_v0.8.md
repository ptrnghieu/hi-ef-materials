# Hi-EF follow-up study: reliability-gated residual inner-development specification v0.8

Status: frozen before execution. This experiment develops the already agreed
context-conditioned affective-residual architecture. It does not redefine the
research gap, use the original validation partition, or access the test
partition.

## Research hypothesis retained

The target remains Party B's future emotion. The model separates the prediction
available from clips I--II from Party A's incremental, context-dependent
evidence in clip III:

```math
s_{final}=s_C+r_A\Delta_A^{raw},\qquad
r_A=\sigma(q(C,Z_A^{affect},Z_A^{interaction})).
```

The scalar gate is a learned estimate of whether Party A's evidence is useful
for this sample. It is not an independent emotion predictor. A real Party-A
clip is not automatically assigned gate target one: real evidence can be
redundant or harmful.

## Motivation fixed from v0.7

Cross-fitted temperature scaling improved probability calibration but could
not change UAR. A single global residual multiplier frequently collapsed to
zero and lost UAR, while label-informed per-sample oracles retained substantial
headroom. Therefore this experiment tests sample-conditioned reliability, not
another global scaling parameter.

## Data boundary and inner split

Only rows assigned to `train` in
`manifests/source_folder_split_seed42.csv` are eligible. They are repartitioned
by complete source folder into `inner_train` and `inner_development` with fixed
seed 8042, 29 and 8 source folders respectively. The committed inner manifest
contains no original validation or test row. Source-folder, clip, and adjacent
clip-pair disjointness are mandatory.

The three fixed development seeds are 42, 123, and 456. Checkpoints are selected
only by inner-development UAR, with inner-development NLL as the tie breaker.
The original validation and test partitions remain sealed.

## Frozen 2 x 2 ablation

All variants use the canonical `both` architecture and the same context,
affect, interaction, and raw-residual modules.

| Variant | Reliability gate | Invalid-A counterfactual training |
| --- | --- | --- |
| `ungated` | no (`r_A=1`) | no |
| `gate` | yes | no |
| `counterfactual` | no (`r_A=1`) | yes |
| `gate_counterfactual` | yes | yes |

The gate is a scalar MLP over the concatenated context, affect, and interaction
representations. Its last-layer weight and bias are initialized to zero
(`r_A=0.5`), so it does not encode a prior decision to accept or reject Party A.

## Counterfactual construction and losses

The canonical zero-A null consistency loss is retained for every variant. For
counterfactual variants, a cyclic one-position roll pairs each context with the
Party-A representations of another sample in the same shuffled training batch.
This preserves the internal coherence of the replacement Party-A clip while
breaking its relationship with the context. A singleton final batch uses zero
Party-A representations.

The invalid-A prediction is trained to return to a detached context prior:

```math
L_{cf}=D_{KL}(\operatorname{stopgrad}(p_C)\;||\;p_{invalid}).
```

For `gate_counterfactual` only, invalid samples also receive a gate-zero loss:

```math
L_{gate0}=-\log(1-r_{invalid}).
```

No positive gate target is imposed on real Party-A samples. Fixed additional
weights are `lambda_cf=1.0` and `lambda_gate0=0.1`.

## Canonical settings retained

The v0.4 settings remain fixed: 50 epochs, batch size 32, AdamW learning rate
`1e-4`, weight decay `1e-5`, patience 8, `d_model=512`, two temporal layers,
two context layers, dropout 0.1, masked face pooling, and joint end-to-end
training. Loss weights remain context 1.0, Party-A emotion 0.5, conditional
contrastive 0.1, zero-A null consistency 1.0, and source nuisance 0.05.

## Outputs and interpretation

Save aligned context logits, raw residuals, effective residuals, final logits,
reliability gates, labels, sample ids, and source folders. Report UAR, WAR,
micro-NLL, raw/effective residual norms, gate distribution, prediction flips,
and beneficial/harmful flips relative to each model's own context prior.

This inner-development matrix is an architecture-development result. It may
freeze one candidate for a later five-seed evaluation on the still-unopened
original validation partition, but it cannot support a test claim. Temperature
scaling remains a calibration procedure and is not counted as an architectural
contribution.

## Advancement rule

`gate_counterfactual` is the sole predeclared candidate. It advances to a
separately frozen original-validation audit only when all of the following hold
on inner development:

1. mean NLL is lower than `ungated`;
2. mean UAR difference from `ungated` is at least -0.005;
3. final UAR exceeds its own context UAR on at least two of three seeds;
4. mean harmful-flip rate is no greater than `ungated`.

Failure stops this candidate. The other cells diagnose the gate and
counterfactual components; they are not post-hoc replacement candidates.
