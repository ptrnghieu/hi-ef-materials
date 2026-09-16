# Hi-EF follow-up study: canonical method specification v0.4

Status: architecture, objectives, training configuration, variants, and model
seeds frozen before the full canonical matrix. Test remains unopened. This
document supersedes v0.3 only for training configuration; the research question
and architecture are unchanged.

## Research question and hypothesis

What incremental, correspondence-sensitive affective information does Party A
provide for forecasting Party B beyond what is already predictable from the
interaction context?

The hypothesis is that Party A contributes a context-dependent correction to
the forecast of Party B, and that explicitly separating affective information
from interaction information makes that correction more attributable and more
robust than an unconstrained raw representation. This is a predictive, not
causal, hypothesis. Split construction and release metadata are protocol
limitations, not the research gap.

## Frozen canonical architecture

Let `C` be clips I--II, `X_A` be clip III, and `E_B` be clip IV's emotion.

```text
Z_A_affect, Z_A_interaction = f_A(X_A)
s_C = h_C(C)
Delta_A = g(C, Z_A_affect, Z_A_interaction)
P(E_B | C, X_A) = softmax(s_C + Delta_A)
```

`Delta_A` is a context-conditioned class-logit correction. `Z_A_affect` and
`Z_A_interaction` do not independently predict `E_B`. The four frozen variants
are `context`, `affect`, `interaction`, and `both`. Each non-context checkpoint
exposes its own context-prior and final predictions for paired attribution.

## Frozen objective

```text
L = L_final
  + 1.00 L_context
  + 0.50 L_emotion
  + 0.10 L_supcon
  + 1.00 L_null
  + 0.05 L_nuisance_GRL
```

- `L_emotion` uses mean-one inverse-frequency class weights computed from the
  training partition only.
- Conditional SupCon temperature is `0.1`. Positives are same-emotion,
  different-source pairs; negatives are different-emotion, same-source pairs.
  Valid-anchor coverage is logged because some minibatches cannot form both.
- The nuisance head uses gradient reversal scale `1.0`; its CE is added to the
  scalar objective while its encoder gradient is reversed.
- The null objective is
  `KL(stopgrad(p_C) || p_C,A_null)`, with the context contribution detached so
  it trains only the residual path toward a zero null correction.
- All branches are jointly optimized end-to-end. There is no context pretraining
  or freezing in the primary matrix.

## Frozen optimization and evaluation

| Setting | Value |
|---|---:|
| model seeds | 42, 123, 456, 789, 1024 |
| batch size | 32 |
| maximum epochs | 50 |
| early-stopping patience | 8 |
| learning rate | 0.0001 |
| weight decay | 0.00001 |
| model width | 512 |
| temporal layers | 2 |
| context layers | 2 |
| dropout | 0.1 |
| face pooling | masked |
| checkpoint criterion | validation UAR, then validation NLL |

The matrix contains exactly 20 runs: four variants by five model seeds. It uses
train and validation only. The primary diagnostic is validation UAR. For every
non-context run, report both (a) final minus its own context-prior UAR and (b)
final minus the separately trained context variant at the same seed. Also report
Party-A affect UAR, individual training losses, conditional-SupCon valid-anchor
coverage, and mean residual norm.

The matrix summary is descriptive and cannot select a final method by itself.
A separately frozen hierarchical validation audit must quantify uncertainty
over model seeds and validation source folders before one variant is advanced.
The test partition is evaluated once only after that audit freezes the method.

## Prior evidence and scope

The v0.3 motivating evidence remains unchanged: raw A gave a small validation
gain over context; the Party-A label oracle gave a larger gain; and intervention
diagnostics showed correspondence sensitivity. These results motivate but do
not validate the new method. The earlier `full_aux`, `affective_bottleneck`, and
`hybrid_residual` matrix remains exploratory and is not part of this canonical
architecture.
