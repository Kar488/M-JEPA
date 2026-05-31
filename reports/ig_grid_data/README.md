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

## Headline finding (pooled, ~2000 paired molecules over 8–12 cells)

- Contrastive attributions are **equal or more** concentrated/connected than
  JEPA's (contrastive wins 62–79% of molecules; Wilcoxon p down to ~1e-190).
  JEPA leads only on motif top-1 share.
- Connected-subgraph masking specifically shows **no coherence advantage** over
  non-contiguous masking (all four metrics ≈0 or slightly negative).
- Conclusion: attribution coherence is **not** a JEPA/masking-specific property.
  RQ1 (JEPA wins predictive *transfer*) is untouched — this concerns attribution
  maps (RQ4), not transfer.

Regenerate stats: `PYTHONPATH=. python experiments/aggregate_ig_grid.py --root <unpacked>/ig_grid`
