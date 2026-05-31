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

## Result (to fill)

Objective family (jepa - contrastive): [mean_diff / p / CI per metric]
Masking specifically (jepa - jepa_dispersed): [mean_diff / p / CI per metric]

Provisional NR-AR pilot (single seed, reduced budget): attribution concentration
was **not** higher for connected-subgraph JEPA; the InfoNCE baseline was equal or
more concentrated (Gini +0.054 in favour of contrastive on ~99% of molecules,
Wilcoxon p ~ 1e-119), with salient-atom connectivity statistically
indistinguishable. The full grid tests whether this holds across assays/seeds and
whether the masking-specific contrast moves at all.

## Manuscript changes (these go INTO the paper, not just this letter)

1. **Replace** the "Comparative attribution against the contrastive baseline"
   paragraph (currently declining the analysis) with a subsection reporting the
   in-pipeline comparison and its paired statistics; full per-assay/per-seed
   numbers in a new SI table.

2. **Add** the connected-vs-dispersed masking comparison *on attribution
   coherence* (not just ESOL RMSE) to the masking-ablation section / SI - this is
   the control that actually speaks to "masking specifically".

3. **Reframe Contribution (1)** from "chemically coherent masking" (which implies
   an attribution advantage we did not measure) to "connected-subgraph masking as
   a chemically coherent *target construction* (an algorithmic property; Algorithm
   1)", with attribution coherence reported as a representation **diagnostic** that
   [is / is not] objective-specific.

4. **Adjust the line-678 rationale** so the motivation for connected masking is
   stated as target interpretability, explicitly noting that attribution
   coherence is [not] unique to this objective/masking choice per the new
   comparison - consistent with the paper's existing "diagnostic, not mechanistic"
   framing.
