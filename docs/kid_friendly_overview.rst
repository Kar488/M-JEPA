M-JEPA: A Kid-Friendly Guide
============================

Welcome to the M-JEPA project! Think of this repository as a giant science
lab where computers learn to understand molecules. Each folder is a
different room with its own tools. This tour explains the whole package in
simple, kid-friendly language while pointing you to the exact files that
drive the latest logic.

Mission Control
---------------

``main.py``
  The big switchboard of the lab. It can run:

  - ``demo`` – tiny smoke tests that pit JEPA against a contrastive baseline
    and even run a miniature Tox21 case study.
  - ``full`` – full self-supervised pretraining followed by fine-tuning and
    evaluation on labelled benchmarks.
  - ``grid`` – YAML/JSON-driven sweeps powered by the tools in
    ``experiments/``.

``scripts/train_jepa.py``
  A command-line friendlier control panel. It exposes subcommands such as
  ``pretrain``, ``finetune``, ``evaluate``, ``benchmark`` and ``tox21``. Each
  stage reuses the shared helpers in ``training/`` and ``experiments/`` so the
  whole pipeline can run from one script.

``scripts/commands/``
  Modular commands that mirror the subcommands above. They are reused by the
  sweep launcher so that automation and local runs stay in sync.

Data Lab: ``data/``
-------------------

This room builds datasets and augmentation recipes.

``mdataset.py``
  Stores molecule graphs as ``GraphData`` objects and bundles them into
  ``GraphDataset`` collections. It can ingest SMILES strings, numpy arrays,
  or cached pickles and keeps track of node, edge and positional features.

``augment.py``
  Generates alternative molecular views. Recent updates added structured
  configs for rotation, angle masking, dihedral noise, atom/bond masking and
  subgraph removal. Helpers like ``generate_views`` and ``apply_graph_augmentations``
  are imported directly by the training loops.

``parquet_loader.py`` & ``moleculenet_dc.py``
  Efficient readers that batch large Parquet/CSV corpora. They respect cache
  directories so repeated sweeps can reuse featurised graphs.

``scaffold_split.py``
  Prepares deterministic train/validation/test splits based on Bemis-Murcko
  scaffolds so downstream evaluation remains fair.

``BASF_AIPubChem_v4/`` and ``ZINC-canonicalized/``
  Example shards stored in ``data/`` for offline experiments. The download
  helpers in ``scripts/download_unlabeled.py`` know how to refresh them.

Model Workshop: ``models/``
---------------------------

Every file here defines a different brain for molecules.

``factory.py``
  The main entry point. Given a configuration, it instantiates the requested
  encoder, predictor and optional exponential moving average (``models/ema.py``).

``gnn_variants.py``
  Implements the supported backbones: MPNN, GIN, GraphSAGE, multi-head GAT,
  DMPNN and SchNet3D. The shared ``models/base.py`` interface keeps them
  interchangeable.

``predictor.py``
  Builds the JEPA prediction head that compares context and target views.

``edge_encoder.py``
  Extends the base encoders so bond features can be encoded alongside atom
  embeddings.

Training Tracks: ``training/``
------------------------------

This is where models learn and where much of the rewritten logic lives.

``unsupervised.py``
  Runs JEPA and contrastive training end-to-end. It wires together
  augmentations, masking strategies, the EMA teacher, multi-device support,
  gradient scaling and W&B logging. Environment variables such as
  ``SWEEP_CACHE_DIR`` and ``WANDB_*`` are respected so sweeps and local
  experiments behave the same way.

``supervised.py`` and ``supervised_with_val.py``
  Train lightweight heads on frozen encoders. The validation-aware variant
  performs early stopping and metric aggregation across seeds.

``supervised_multi.py`` and ``multitask.py``
  Handle multi-output heads and multi-task learning where a single encoder
  feeds several prediction heads.

``pretrain.py`` and ``baselines.py``
  Thin wrappers that prepare encoders and call into the unsupervised trainer.
  ``baselines.py`` also integrates third-party methods so comparisons stay
  apples-to-apples.

``train_on_embeddings.py``
  Consumes saved embeddings and trains quick probes (logistic regression,
  random forests, etc.) without rerunning the GNN backbone.

Helpful Tools: ``utils/``
-------------------------

Shared utilities that every room borrows.

``dataset.py`` & ``dataloader.py``
  Glue code that turns ``GraphDataset`` objects into PyTorch ``DataLoader``
  instances. They include smart defaults for worker counts, file descriptor
  limits and pinned memory.

``logging.py``
  Provides ``maybe_init_wandb`` so scripts can safely log to Weights & Biases
  or fall back to a no-op logger during tests.

``checkpoint.py``
  Saves/restores encoder, predictor, EMA and optimizer state. Used by both
  the training scripts and by the sweep agents to resume work.

``schedule.py`` & ``early_stopping.py``
  Implement cosine-with-warmup schedules, plateau detectors and patience-based
  stopping criteria.

``graph_ops.py`` & ``pooling.py``
  Encode raw graphs into tensors and reduce node embeddings to graph-level
  vectors.

``metrics.py`` & ``scatter.py``
  Evaluate regression/classification scores and supply efficient scatter
  operations for graph batching.

Experiment Control Tower: ``experiments/``
------------------------------------------

Where sweeps, reports and exploratory studies live.

``grid_search.py``
  Reads YAML/JSON specs and spawns training runs with shared logging. This is
  what the ``grid`` mode in ``main.py`` calls.

``case_study.py`` & ``probing.py``
  Run qualitative evaluations like the Tox21 toxicity case study or probing
  classifiers over frozen embeddings.

``ablation.py``
  Systematically disables model components to measure their impact.

``baseline_integration.py``
  Wraps external repositories (MolCLR, HiMol, GeomGCL, etc.) so their encoders
  can be pre-trained and evaluated using the same scripts.

``reporting.py``
  Builds CSV summaries, 95% confidence intervals, bar charts and heatmaps for
  sweep results.

Automation & Agents: ``scripts/``
---------------------------------

Automation scripts keep the lab humming, especially inside CI.

``scripts/ci/run-grid-or-phase1.sh``
  Launches paired JEPA and contrastive sweeps. It rewrites sweep specs so both
  methods share the same backbone/seed grid (controlled by ``PHASE1_BACKBONES``
  and ``PHASE1_SEEDS``) and then spawns Weights & Biases agents.

``scripts/ci/paired_effect_from_wandb.py``
  Downloads sweep history, groups runs by shared architecture knobs and writes
  ``paired_effect.json`` containing the per-method deltas.

``scripts/ci/phase1_decision.py``
  A new resolver that interprets ``paired_effect.json`` safely. It reports
  whether JEPA, contrastive or a tie won and whether a tie-breaker was needed.

``scripts/ci/export_best_from_wandb.py`` & ``scripts/ci/recheck_topk_from_wandb.py``
  Materialise the top configurations from phase 1 and refresh the shortlist
  if new runs arrive late.

``scripts/ci/run-grid-phase2.sh``
  Starts the Bayesian phase 2 sweep for the winning method. It plugs the top
  configs exported above into a new spec stored under ``grid/`` so tracked
  templates are never overwritten.

``scripts/ci/run-pretrain.sh``, ``scripts/ci/run-finetune.sh``, ``scripts/ci/run-bench.sh`` and ``scripts/ci/run-tox21.sh``
  Stage-specific launchers used by GitHub Actions. They all rely on the shared
  setup helpers in ``scripts/ci/common.sh`` and ``scripts/ci/stage.sh``.

``commands/sweep_run.py``
  The entry point executed by each sweep agent. It logs the backbone, hidden
  dimension, layer count and masking strategy that later identify matched run
  pairs.

Adapters & Third-Party Friends
------------------------------

``adapters/``
  Provides bridges into external baselines. ``cli_runner.py`` shells out to
  other repos, while ``native_adapter.py`` imports them as Python packages. A
  central ``config.yaml`` records paths and CLI templates.

``third_party/``
  Holds the vendor code (MolCLR, HiMol, GeomGCL, etc.) used during baseline
  comparisons.

Reports, Samples & Sweeps
-------------------------

``sweeps/``
  YAML specs for phase-1 and phase-2 Weights & Biases sweeps. They expose all
  tunable knobs and are edited on-the-fly by the CI scripts described above.

``reports/``
  Where automated summaries land. Phase-1 runs generate decision artifacts,
  phase-2 produces ranked CSVs and plots (thanks to ``experiments/reporting.py``).

``samples/``
  Tiny CSV/Parquet files that let tests and demos run without downloading
  millions of molecules.

Quality Checks
--------------

``tests/``
  Pytest suites that cover dataset loading, augmentation transforms, logging,
  sweep wiring and training utilities. They make heavy use of the synthetic
  samples and of the logging fallbacks in ``utils/logging.py``.

``training/README.md``
  Quick reference for the training sub-packages and the expected entry points.

What's New?
-----------

Recent rewrites introduced:

* A resilient phase-1 decision helper (``scripts/ci/phase1_decision.py``) that
  guarantees automation can detect ties.
* Expanded augmentations and masking strategies in ``data/augment.py`` and
  ``training/unsupervised.py`` so JEPA and contrastive baselines share the same
  view-generation code.
* Confidence-interval reporting utilities in ``experiments/reporting.py`` that
  feed the plots saved under ``reports/``.
* Multi-task and multi-head training helpers under ``training/multitask.py`` and
  ``training/supervised_multi.py``.
* Improved sweep launchers that respect ``PHASE1_BACKBONES`` and ``PHASE1_SEEDS``
  so larger paired comparisons can be run without editing the tracked YAML.

FAQ
---

Why did a 30-run phase-1 sweep report only nine pairs?
  The phase-1 driver launches one sweep for JEPA and one for the contrastive
  baseline. Both sweeps reuse the same grid of shared hyper-parameters: the
  backbone ``gnn_type`` values ``gine``, ``dmpnn`` and ``schnet3d`` and the
  random seeds ``{1, 2, 3}``. The paired-effect script only forms a pair when it
  finds matching runs from *both* methods that share the same backbone and seed,
  so there can be at most ``3 backbones × 3 seeds = 9`` such combinations. Extra
  trials that a sweep might launch explore method-specific knobs (such as JEPA's
  ``mask_ratio`` or EMA decay) and still feed phase-2 selection, but they do not
  increase the matched-pair count because they lack a partner run from the other
  method with identical shared settings.

What happens when there is no clear winner on the primary metric?
  ``scripts/ci/phase1_decision.py`` inspects ``paired_effect.json``. If the mean delta is
  within the configured tolerance (``--tie-tol``), or if the JSON reports that a
  tie-breaker was used, the script returns ``tie``. Otherwise it selects the
  method that improves the requested metric (``val_rmse`` by default for
  regression, ``val_auc`` for classification). Automation can then choose to run
  both methods in phase 2 or stick with the winner.

What configuration keys form a "pair" today?
  ``scripts/commands/sweep_run.py`` records four architecture knobs for every
  run: ``gnn_type``, ``hidden_dim``, ``num_layers`` and ``contiguity``. The
  paired-effect analysis hashes those into a ``pair_id`` so JEPA and contrastive
  runs with matching settings can be compared fairly. Method-specific hyper-
  parameters (like JEPA's ``mask_ratio``) are deliberately excluded so each
  method can explore its own search space.

How do we increase the number of matched pairs beyond ``3 × 3 = 9``?
  Set ``PHASE1_BACKBONES`` and/or ``PHASE1_SEEDS`` before running
  ``scripts/ci/run-grid-or-phase1.sh``. The script rewrites temporary YAML files
  so both sweeps cover the requested combinations without altering the tracked
  templates in ``sweeps/``. For example,
  ``PHASE1_SEEDS=1,2,3,4,5 GRID_MODE=wandb bash scripts/ci/run-grid-or-phase1.sh``
  yields ``3 backbones × 5 seeds = 15`` potential pairs.

How many phase-1 runs do we need before ``pair_id`` overlap becomes likely?
  Phase-1 sweeps sample backbones uniformly. With ``n`` runs per method, the
  chance that at least one backbone overlaps – and therefore that a ``pair_id``
  match exists – increases quickly with ``n``. Historical plots and tables can be
  regenerated with the helpers in ``reports/`` if you want to explore past
  budgets before scheduling new sweeps.

Have fun exploring!
     :widths: 15 25 30

     * - Runs per method (``n``)
       - Probability of a shared ``pair_id``
       - Probability both sweeps cover all three backbones
     * - 3
       - 94.2%
       - 4.9%
     * - 4
       - 98.6%
       - 19.8%
     * - 5
       - 99.7%
       - 38.1%
     * - 6
       - 99.9%
       - 54.9%

  Even with four runs per method there is already a 98.6% chance that the two
  sweeps share at least one ``pair_id``.  Increasing the run count mainly
  improves the odds that both sweeps exercise *every* backbone, which is useful
  when you want to make a decision per backbone rather than relying on the
  global fallback in the paired-effect report.
