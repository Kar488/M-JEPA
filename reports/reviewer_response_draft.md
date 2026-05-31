# Draft: response to reviewer follow-up on comparative IG (D26)

> NOTE: numbers in [BRACKETS] are placeholders to be filled from
> `outputs/ig_grid/grid_paired_stats.json` once the 4-assay x 3-seed x 3-arm
> grid completes. This is a reduced-budget controlled ablation (CPU); final
> manuscript figures should be regenerated under the production GPU protocol.

## What the reviewer is right about

The reviewer is correct that comparing the contrastive baseline does **not**
require stepping outside the leakage-resistant pipeline. The scaffold split,
per-assay single-task training, and validation-only calibration are all
objective-agnostic: one swaps the pretrained encoder and runs the *same* Phase-3
hybrid schedule on the *same* scaffold-split train/test folds. We have therefore
withdrawn the "outside the leakage-resistant pipeline" justification and run the
comparison.

## What we ran (in-pipeline)

For each of four assays (NR-AR, NR-AhR, SR-MMP, NR-ER) and three seeds, we
pretrained three encoders at matched compute on the same molecule set and
fine-tuned each through the identical hybrid staged-unfreeze schedule on the same
Bemis-Murcko scaffold split, then computed Integrated Gradients (atom-, bond-,
and motif-level) on the same scaffold-split test molecules. The three arms are:

- **jepa** - predictive JEPA with connected-subgraph masking (contiguity = 1);
- **jepa_dispersed** - predictive JEPA with non-contiguous masking (contiguity = 0);
- **contrastive** - the compute-matched InfoNCE baseline.

Attribution coherence is summarised per molecule by four scale-free proxies:
atom-IG Gini, top-20% atom-attribution mass, largest connected component among
the top-20% salient atoms, and top-1 motif attribution share. Because the capped
test order is seed-deterministic, the same molecules appear in every arm, so we
report **paired** per-molecule differences with Wilcoxon signed-rank tests and
bootstrap 95% CIs (matching the paired-statistics style used elsewhere in the
paper).

Two contrasts isolate two distinct factors:

- **jepa vs contrastive** isolates the **objective family** (predictive vs
  instance-discrimination);
- **jepa vs jepa_dispersed** isolates **connected-subgraph masking specifically**
  - which is the precise causal target of the reviewer's comment, and which a
  JEPA-vs-contrastive comparison alone cannot isolate.

## Result (12/12 cells, 2964 paired molecules; reduced-budget CPU ablation)

**Objective family (jepa - contrastive):** metric-dependent.
- Motif top-1 share (the manuscript's motif-level unit of analysis):
  Δ = +0.055, JEPA wins 82% of molecules, Wilcoxon p ~ 1e-255; per-assay
  Δ +0.013 (NR-AR) to +0.115 (SR-MMP), JEPA-favouring on all four assays.
- Atom top-20% mass: Δ = -0.030 (contrastive somewhat more concentrated).
- Salient-atom connectivity: Δ = -0.013 (mixed across assays).
- Atom Gini: Δ = +0.009 (tie).

**Masking specifically (jepa - jepa_dispersed):** no coherence advantage for
connected masking on any metric (motif Δ ~ 0 to slightly negative; connectivity
negative on all four assays).

Interpretation: motif-level attribution coherence is *higher* for the predictive
JEPA objective than for the InfoNCE baseline, but does *not* depend on the
connected-subgraph masking contiguity. The reviewer's narrow causal point holds
(coherence is not masking-specific); the broader concern that coherence is
objective-agnostic does not (JEPA is more motif-coherent than contrastive).

## Manuscript changes (these go INTO the paper, not just this letter)

1. **Replace** the "Comparative attribution against the contrastive baseline"
   paragraph (currently declining the analysis) with a subsection reporting the
   in-pipeline comparison and its paired statistics; full per-assay/per-seed
   numbers in a new SI table.

2. **Add** the connected-vs-dispersed masking comparison *on attribution
   coherence* (not just ESOL RMSE) to the masking-ablation section / SI - this is
   the control that actually speaks to "masking specifically".

3. **Reframe Contribution (1):** attribute motif-level attribution coherence to
   the **predictive JEPA objective** (now supported: JEPA > contrastive on motif
   top-1 share across all four assays), and demote "connected-subgraph masking"
   to a chemically-interpretable *target construction* (Algorithm 1) that is NOT
   the source of the coherence (jepa vs jepa_dispersed shows no masking effect).

4. **Adjust the line-678 rationale:** state connected masking's motivation as
   target interpretability, and note explicitly that attribution coherence is
   objective-driven (predictive vs contrastive) rather than masking-contiguity-
   driven, per the new comparison - consistent with the existing "diagnostic, not
   mechanistic" framing.
