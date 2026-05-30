# Contrastive-vs-JEPA Integrated-Gradients ablation

## Motivation

A reviewer observed that the chemical-coherence-of-masking claim rests on
within-M-JEPA evidence only. Without a contrastive comparison, the observed
attribution quality cannot be attributed to **connected-subgraph masking
specifically** rather than to the **predictive objective broadly**.

The original manuscript argued such a comparison "would require fine-tuning the
contrastive baseline outside the leakage-resistant pipeline." That framing
conflates two independent things:

- **Leakage resistance** comes from the Bemis-Murcko scaffold split
  (`data/scaffold_split.py`). It is a property of the data partition and is
  identical regardless of pretraining objective.
- **The frozen-lineage policy** (`docs/frozen_lineage_policy.rst`) is a CI
  bookkeeping convention that carries only the Phase-1 *winner* forward and
  stamps it immutable. It is why no contrastive checkpoint was graded — not
  leakage.

A like-for-like comparison is therefore fully achievable inside the
leakage-resistant design, as a separate task run outside the locked grading
pipeline.

## Method

`experiments/contrastive_ig_ablation.py` runs the existing Tox21 case study
(`experiments.case_study.run_tox21_case_study`) twice with **identical settings**,
varying only the pretraining objective:

| Held constant across both arms | Varied |
| --- | --- |
| Tox21 molecules (`data/tox21/data_rdkit_clean.csv`) | objective: JEPA masking vs InfoNCE |
| Bemis-Murcko scaffold split, seed 42 (train 5806 / val 726 / test 726) | |
| Architecture (mpnn, hidden 256, 3 layers) | |
| Pretrain 25 ep + fine-tune 25 ep, `end_to_end` full fine-tune | |
| Fixed Tox21 recipe: encoder_lr 1e-5, head_lr 3e-4, layerwise 0.8, dynamic pos-weight, temperature calibration | |
| Integrated Gradients + IG-motif on the same 722-molecule test set | |

The run is deliberately driven through the engine directly, **bypassing the CI
frozen-lineage gating**. It writes no `encoder_frozen.ok` marker and does not
touch any graded lineage.

`experiments/analyze_ig_coherence.py` then quantifies attribution coherence on
the matched test molecules:

- **Concentration:** Gini coefficient and top-20% mass of `|IG|` over atoms.
- **Spatial connectivity:** fraction of the top-20% salient atoms that lie in
  their single largest connected component (using molecular bonds). Higher =>
  salient atoms form a connected subgraph.
- **Motif concentration:** share of total motif importance held by the single
  most important motif.

## Results (reduced-budget, NR-AR, seed 42)

| Metric | JEPA (masking) | Contrastive (InfoNCE) |
| --- | --- | --- |
| Atom IG Gini (mean) | 0.539 | **0.561** |
| Top-20% atom mass (mean) | 0.502 | **0.537** |
| Motif top-1 share (mean) | **0.544** | 0.533 |
| Salient-atom largest CC fraction (mean) | 0.687 | **0.759** |
| Downstream ROC-AUC | 0.636 | 0.707 |

**Finding.** At this matched budget, connected-subgraph masking does **not**
produce more coherent attributions than the contrastive objective. Contrastive
is marginally more concentrated and notably more spatially connected; the two
are essentially tied on motif concentration. This is evidence consistent with
the reviewer's concern: the attribution coherence is **not** uniquely
attributable to connected-subgraph masking, and a predictive contrastive
objective yields comparable or better coherence on these proxies.

## Caveats

This is a **reduced-budget, single-task, single-seed, CPU** sub-study, not a
headline result:

- Self-pretrained on ~5,806 Tox21 molecules, **not** the 10M ZINC corpus used
  for the frozen headline encoder.
- Single assay (NR-AR), single seed (42), `end_to_end` full fine-tune (not the
  locked `hybrid` frozen-encoder evaluation mode).
- 25 pretrain + 25 fine-tune epochs.

Absolute numbers are therefore **not comparable** to the headline frozen-encoder
Tox21 metrics and must not be reported as such. The comparison between objectives
is the interpretable quantity. A full-budget, multi-seed, multi-task replication
on GPU would be required to make the comparison definitive; the present evidence
points away from the masking-specificity claim rather than toward it.

## Reproduce

```bash
PYTHONPATH=. python experiments/contrastive_ig_ablation.py \
  --csv data/tox21/data_rdkit_clean.csv --task NR-AR \
  --pretrain-epochs 25 --finetune-epochs 25 \
  --explain-mode ig,ig_motif --explain-steps 32 \
  --arms jepa,contrastive --out outputs/ig_ablation

PYTHONPATH=. python experiments/analyze_ig_coherence.py \
  --root outputs/ig_ablation --task NR-AR
```
