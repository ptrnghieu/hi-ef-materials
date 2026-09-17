# Hi-EF follow-up study: relative-reliability inner-development specification v0.9

Status: frozen before execution. This is the final planned reliability-method
development round. It retains the agreed context-conditioned affective-residual
architecture and changes only the supervision of its scalar reliability gate.
It does not redefine the research gap, use the original validation partition,
or access the test partition.

## Retained research hypothesis

The target remains Party B's future emotion. Clips I--II provide a context
prior, while Party A in clip III provides an incremental, context-dependent
correction:

```math
s_{final}=s_C+r_A\Delta_A^{raw},\qquad
r_A=\sigma(q(C,Z_A^{affect},Z_A^{interaction})).
```

The gate is not an independent emotion predictor. It estimates the relative
reliability of Party A's evidence for the current context.

## Frozen motivation from v0.8

The v0.8 `gate_counterfactual` candidate collapsed to a mean gate of 0.0067,
zero prediction flips, and exactly zero UAR improvement over its own context.
The ungated residual nevertheless improved over its own context in all three
inner-development seeds. Therefore v0.9 retains the residual hypothesis but
replaces absolute invalid-gate-to-zero supervision with a relative ordering
constraint. No v0.8 cell is promoted post hoc.

## Data boundary

Use exactly `manifests/inner_development_seed8042.csv`. Only original-training
rows are eligible: 29 complete source folders in `inner_train` and eight in
`inner_development`. The fixed development seeds are 42, 123, and 456.
Checkpoints are selected by inner-development UAR with NLL as tie breaker.
Original validation and test remain sealed.

## Frozen three-cell matrix

All cells use the canonical `both` residual architecture.

| Variant | Learned scalar gate | Relative ranking | Counterfactual output consistency |
| --- | --- | --- | --- |
| `ungated` | no (`r_A=1`) | no | no |
| `relative_gate` | yes | yes | no |
| `relative_gate_cf` | yes | yes | yes |

`relative_gate_cf` is the sole predeclared advancement candidate.
`relative_gate` diagnoses the ranking objective and may not be substituted after
results are observed.

## Invalid Party A and frozen losses

A cyclic one-position roll pairs each context with the complete Party-A
representation of another shuffled training example. A singleton batch uses
zero Party-A representations. For gated cells, let `r_real` and `r_cf` be the
gates for the aligned and counterfactual pairs. The relative reliability loss is

```math
L_{rank}=\max(0,\;m-r_{real}+r_{cf}),\qquad m=0.20.
```

It requires aligned evidence to rank above counterfactual evidence without
assigning an absolute target of one or zero. The fixed weight is
`lambda_rank=0.1`.

For `relative_gate_cf` only, the counterfactual prediction is trained toward a
detached context prior:

```math
L_{cf}=D_{KL}(\operatorname{stopgrad}(p_C)\;||\;p_{cf}),
```

with `lambda_cf=1.0`. No absolute gate target is used. The canonical zero-A
null-consistency loss remains active for every cell.

## Canonical settings retained

Use 50 epochs, batch size 32, AdamW learning rate `1e-4`, weight decay `1e-5`,
patience 8, `d_model=512`, two temporal layers, two context layers, dropout
0.1, masked face pooling, and joint end-to-end training. Existing loss weights
remain context 1.0, Party-A emotion 0.5, conditional contrastive 0.1, zero-A
null consistency 1.0, and source nuisance 0.05.

## Outputs and collapse diagnostics

Save aligned context logits, raw and effective residuals, final logits, real
gates, counterfactual gates, labels, sample IDs, and source folders. Report UAR,
WAR, NLL, gate quantiles, real-minus-counterfactual gate separation, prediction
flips, and beneficial/harmful flips relative to each model's own context.

## Frozen advancement rule

`relative_gate_cf` advances to a separately frozen original-validation audit
only if every criterion below holds on inner development:

1. its mean gate is in `[0.10, 0.90]`;
2. its mean real-minus-counterfactual gate separation is greater than zero;
3. its mean prediction-flip rate is greater than zero;
4. its mean beneficial-flip rate exceeds its mean harmful-flip rate;
5. final UAR exceeds its own context UAR in at least two of three seeds;
6. mean UAR is no worse than `ungated - 0.005`;
7. in a paired hierarchical bootstrap over model seed and source folder, the
   95% lower bound for `relative_gate_cf - ungated` UAR is greater than `-0.005`.

The bootstrap uses 5,000 replicates and seed 6901. Failure stops learned-gate
development; there is no v0.10 architecture search under this plan. The
original validation and test partitions may not be opened by this matrix.
