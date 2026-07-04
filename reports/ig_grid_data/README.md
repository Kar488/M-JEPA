# IG coherence grid — data snapshot

Raw evidence behind the contrastive-vs-JEPA Integrated-Gradients coherence
ablation (reviewer comment D26 follow-up). Committed so the data survives
container reclamation.

## Contents

- `ig_grid_snapshot.tgz` — per-molecule IG artifacts for every completed cell:
  atom/bond IG CSVs, motif summaries, per-cell `result_summary.json` and
  `run.log`. **Excludes** per-molecule PNG heatmaps (~3.3 GB of visualisations)
  and `pretrained_encoder.pt` files (regenerable). Unpack with
  `tar -xzf ig_grid_snapshot.tgz`.
- `grid_paired_stats.json` — aggregated paired statistics (the headline numbers).

## Design

- Grid: assays {NR-AR, NR-AhR, SR-MMP, NR-ER} × seeds {42, 1, 2} × arms
  {jepa (connected masking), jepa_dispersed (contiguity=0), contrastive (InfoNCE)}.
- Mode: hybrid (pretrain encoder → staged-unfreeze fine-tune → IG), matching the
  paper's reported IG configuration. Reduced-budget CPU ablation (20/20 epochs,
  IG capped to 250 test molecules per assay). **Directional evidence, not
  production numbers** — regenerate under the GPU protocol for final figures.
- Pairing: capped test order is seed-deterministic, so the same molecules appear
  in every arm of a cell; per-molecule differences are exact.

## Two contrasts

- `jepa_minus_contrastive` — isolates objective family (predictive vs InfoNCE).
- `jepa_minus_jepa_dispersed` — isolates connected-subgraph masking specifically
  (the reviewer's literal causal target).

## Headline finding (pooled, 12/12 cells, 2964 paired molecules)

Metric-dependent, so state it precisely:

- **Motif-level concentration (motif top-1 share) — the manuscript's unit of
  analysis: JEPA > contrastive**, robustly and across all four assays
  (pooled Δ +0.055, JEPA wins 82% of molecules, Wilcoxon p ~ 1e-255; per-assay
  Δ +0.013 to +0.115). The paper's motif-level coherence claim is *supported*
  by the comparison.
- **Atom-level metrics are mixed:** contrastive is somewhat more concentrated on
  top-20% atom mass (Δ -0.030) and marginally more connected; atom Gini is a tie.
- **Connected-subgraph masking specifically (jepa vs jepa_dispersed): no
  coherence advantage** on any metric (motif Δ ≈ 0 to slightly negative;
  connectivity negative on all four assays). The masking *contiguity* does not
  drive coherence.

Refined conclusion: motif-level attribution coherence traces to the **predictive
JEPA objective**, not to connected-subgraph masking. So the reviewer's narrow
point holds (coherence is not masking-specific), but the broader worry that
"contrastive would be equally coherent" does not — JEPA is more motif-coherent
than contrastive. RQ1 (predictive transfer) is untouched.

NOTE ON EARLIER PARTIAL SNAPSHOTS: 2/4/8-cell intermediate aggregates were
NR-AR-heavy (the noisy 4.2%-positive assay) and over-indexed on atom-level
metrics; they suggested "contrastive wins decisively." The balanced 12-cell
grid does not support that — see motif-level result above.

Regenerate stats: `PYTHONPATH=. python experiments/aggregate_ig_grid.py --root <unpacked>/ig_grid`
