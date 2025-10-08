M‑JEPA Agents Handbook
=======================

> **Summary**
> - Frozen encoder lineages are immutable.
> - Dependent runs must set ``PRETRAIN_EXP_ID`` to reuse them.
> - To rebuild, remove or override the freeze marker explicitly.

Welcome! This guide gives automation agents and contributors a concise but
complete map of the repository. Use it alongside ``docs/kid_friendly_overview.rst``
and ``docs/frozen_lineage_policy.md`` when you need a narrative tour; the sections
below highlight the latest logic, how components interact, and where to plug in
when extending the pipelines.

Project Snapshot
----------------

*Purpose.* M‑JEPA develops Joint Embedding Predictive Architectures for
molecular graphs and benchmarks them against contrastive pretraining baselines.
Self-supervised encoders are trained on large unlabeled corpora, evaluated on
MoleculeNet-style downstream tasks, and orchestrated via Weights & Biases (W&B)
sweeps.

*Key Questions Answered.*

- Which backbone (GINE, DMPNN, SchNet3D, etc.) and masking strategy works best
  for JEPA-style objectives?
- How does JEPA compare to contrastive baselines under matched hyper-parameters?
- Which pretraining checkpoint should feed downstream evaluation pipelines?

Core Entry Points
-----------------

``main.py``
  Three execution modes:

  * ``demo`` – smoke tests covering JEPA vs. contrastive, linear probes, and a
    miniature Tox21 case study.
  * ``full`` – sequential pretraining, fine-tuning, evaluation.
  * ``grid`` – YAML/JSON-driven sweeps (delegates to ``experiments/grid_search.py``).

``scripts/train_jepa.py``
  Primary CLI used in automation. Subcommands map to reusable helpers:

  * ``pretrain`` / ``finetune`` / ``evaluate`` – orchestrate the stages in
    ``training/``.
  * ``benchmark`` – compare JEPA and contrastive checkpoints on the same dataset.
  * ``tox21`` – run the full Tox21 case study.

``scripts/commands/``
  Subcommand implementations imported by both ``scripts/train_jepa.py`` and the
  sweep driver. Each module encapsulates argument parsing, dataset loading, and
  logging for its stage.

Data & Feature Engineering
--------------------------

``data/mdataset.py``
  Defines ``GraphData`` and ``GraphDataset`` with utilities to materialise graphs
  from SMILES, cached pickles, or synthetic samples. Graphs store node, edge and
  positional features.

``data/augment.py``
  Generates augmented molecular views. Recent rewrites introduce configurable
  rotation, angle masking, dihedral noise, subgraph removal, bond deletion, and
  atom masking. ``training/unsupervised.py`` pulls these helpers directly.

``utils/dataset.py`` & ``utils/dataloader.py``
  Shared loaders with caching support. They expose ``--cache-dir`` /
  ``SWEEP_CACHE_DIR`` knobs so repeated sweeps reuse featurised tensors.

``data/parquet_loader.py`` & ``data/moleculenet_dc.py``
  Efficient readers for large unlabeled corpora and MoleculeNet benchmarks. They
  understand partitioned Parquet layouts and streaming over many shards.

Model & Training Stack
----------------------

``models/factory.py``
  Builds encoders (MPNN, GIN, GraphSAGE, GAT, DMPNN, SchNet3D), prediction heads,
  and EMA teachers from configuration dictionaries. The rest of the code always
  calls through the factory.

``training/unsupervised.py``
  Heart of JEPA/contrastive pretraining. Notable features:

  * Shared masking & augmentation pipeline so methods compare fairly.
  * Automatic AMP/GradScaler selection, BF16 support, and DDP helpers via
    ``utils/ddp.py``.
  * Checkpoint resume, timeout-aware logging, and resilient W&B logging via
    ``utils/logging.maybe_init_wandb``.

``training/supervised.py`` & ``training/supervised_with_val.py``
  Linear probes over frozen encoders. The validation-aware version aggregates
  metrics across seeds and supports early stopping.

``training/supervised_multi.py`` & ``training/multitask.py``
  Multi-head evaluation utilities for datasets with multiple targets.

``training/train_on_embeddings.py``
  Light-weight probes (logistic regression, random forests, etc.) that consume
  stored graph embeddings without re-running the encoder.

Automation & Orchestration
--------------------------

The CI/CD pipeline pairs GitHub workflows with Vast jobs to orchestrate every
stage. CI launches ``scripts/ci/run-pretrain.sh`` to seed new lineages, then
delegates to ``run-grid-or-phase1.sh`` and ``run-grid-phase2.sh`` sweeps. Vast
workers pick up sweep agents, emit structured logs under ``logs/phase*`` and
stream metrics to Weights & Biases. ``scripts/ci/run-tox21.sh`` grades frozen
encoders and stamps ``bench/encoder_frozen.ok`` when the lineage is immutable.
See ``docs/pipeline_overview.md`` for the complete flow.

Stage Agents & Responsibilities
-------------------------------

Each automation agent follows a strict lineage policy. All stage names align
with ``docs/frozen_lineage_policy.md`` and ``docs/pipeline_overview.md``.

| Agent | Role | Reads From | Writes To | Notes |
| --- | --- | --- | --- | --- |
| ``pretrain-agent`` | Launches initial pretrain stage and produces encoder artifacts. | — | ``$EXPERIMENTS_ROOT/$EXP_ID/artifacts`` | Creates new lineage. Skipped if frozen. |
| ``phase1-agent`` | Runs Phase-1 sweep on encoder. | ``$EXPERIMENTS_ROOT/$PRETRAIN_EXP_ID/artifacts`` | ``$EXPERIMENTS_ROOT/$GRID_EXP_ID/grid/phase1`` | No-op when frozen. |
| ``phase2-agent`` | Runs Phase-2 sweep/recheck/export. | ``$GRID_DIR`` (phase1 results) | ``$EXPERIMENTS_ROOT/$EXP_ID/grid/phase2_*`` | Reads frozen grid, writes current logs. |
| ``finetune-agent`` | Finetunes on labeled data using frozen encoder. | ``$PRETRAIN_EXP_ID/artifacts`` | ``$EXP_ID/finetune`` | Always runs under new ``EXP_ID``. |
| ``tox21-agent`` | Grades encoder on Tox21 tasks. | ``$PRETRAIN_EXP_ID/artifacts`` | ``$EXP_ID/tox21`` | On success, writes freeze marker. |
| ``report-agent`` | Generates evaluation reports. | All above outputs | ``$EXP_ID/report`` | Read-only if lineage frozen. |

Frozen vs. New Lineage Behaviour
--------------------------------

The lineage flow is ``pretrain → phase1/phase2 → tox21 → benchmark/report``.
When ``tox21-agent`` stamps ``bench/encoder_frozen.ok``, subsequent runs must
declare ``PRETRAIN_EXP_ID`` and operate in read-only mode against the frozen
artifacts. Only initiator stages (``pretrain-agent`` and new grid sweeps) produce
new ``EXP_ID`` roots unless explicit overrides are set. Downstream stages write to
fresh ``EXP_ID`` directories, preserving prior results. For override semantics see
``docs/frozen_lineage_policy.md``.

Lifecycle Example
-----------------

1. ``pretrain-agent`` creates ``$EXPERIMENTS_ROOT/1759825317`` and writes
   ``artifacts/``.
2. ``phase1-agent`` and ``phase2-agent`` reuse that encoder, producing
   ``grid/phase1`` and ``grid/phase2_export`` under ``$GRID_EXP_ID``.
3. ``tox21-agent`` grades the encoder, emits ``bench/encoder_frozen.ok`` and
   freezes the lineage.
4. Weeks later, ``finetune-agent`` runs with ``PRETRAIN_EXP_ID=1759825317``.
   CI assigns ``EXP_ID=18327125156`` for fresh ``finetune/`` and ``report/``
   outputs without mutating the frozen tree.

Override Flags
--------------

Automation honours the explicit lineage overrides described in
``docs/frozen_lineage_policy.md``. Agents may set:

- ``FORCE_UNFREEZE_GRID=1`` to rebuild a lineage and re-materialise sweeps.
- ``ALLOW_CODE_DRIFT_WHEN_FROZEN=1`` to bypass commit-hash checks on dependents.
- ``STRICT_FROZEN=1`` to enforce commit parity before reading a frozen lineage.
- ``FORCE_RERUN=stage1,stage2`` (comma-separated) to re-run selected stages even
  if cached artifacts exist.
- ``PRETRAIN_EXP_ID`` / ``GRID_EXP_ID`` to bind a run to a specific lineage.

Automation & Sweeps
-------------------

``sweeps/``
  Tracked W&B sweep templates. Phase‑1 specs compare JEPA vs. contrastive; the
  derived phase‑2 specs live under ``grid/`` so tracked templates remain clean.

``scripts/ci/run-grid-or-phase1.sh``
  Creates paired sweeps and launches W&B agents. Respects
  ``PHASE1_BACKBONES``/``PHASE1_SEEDS`` to expand the shared backbone/seed grid.
  Logs run IDs under ``logs/phase1_*``.

``scripts/ci/paired_effect_from_wandb.py``
  Downloads completed runs, groups them by the pairing key emitted by
  ``scripts/commands/sweep_run.py`` (``gnn_type``, ``hidden_dim``, ``num_layers``,
  ``contiguity``), and writes ``paired_effect.json``.

``scripts/ci/phase1_decision.py``
  Resolver that interprets ``paired_effect.json`` safely. Returns the winner
  (``jepa``, ``contrastive`` or ``tie``), whether a tie-breaker fired, and the
  inferred task type. Automation uses this to decide phase‑2 policies.

``scripts/ci/export_best_from_wandb.py`` & ``scripts/ci/recheck_topk_from_wandb.py``
  Export top configurations (CSV + YAML) and refresh them if new runs finish.

``scripts/ci/run-grid-phase2.sh``
  Launches a Bayesian optimisation sweep for the winning method using the
  exported top‑K as seeds. Writes derived specs to ``grid``.

``scripts/ci/run-pretrain.sh`` / ``run-finetune.sh`` / ``run-tox21.sh``
  Stage-specific wrappers used in CI. They share environment setup and failure
  handling via ``scripts/ci/common.sh`` and ``scripts/ci/stage.sh``.

Reporting & Artifacts
---------------------

``experiments/reporting.py``
  Builds ranked CSVs, confidence intervals, bar plots and heatmaps from sweep
  results. Automation drops these into ``reports/``.

``reports/``
  Stores exported configs, decision JSON artifacts and rendered plots. Inspect
  this directory to audit the latest sweeps.

``adapters/`` & ``third_party/``
  Bridges into baseline repositories (MolCLR, HiMol, GeomGCL). ``adapters``
  encapsulates CLI/native integration; ``third_party`` holds the vendor code.

Running Sweeps Manually
-----------------------

1. Authenticate with W&B (``wandb login`` or ``export WANDB_API_KEY=...``).
2. Set ``GRID_MODE=wandb`` and optional overrides like ``PHASE1_BACKBONES`` or
   ``SWEEP_CACHE_DIR``.
3. Launch phase 1:

   .. code-block:: bash

      GRID_MODE=wandb bash scripts/ci/run-grid-or-phase1.sh

4. Once ``paired_effect.json`` is produced, inspect the winner:

   .. code-block:: bash

      python scripts/ci/phase1_decision.py grid/paired_effect.json

5. For the winning method (or both, on ties), launch phase 2:

   .. code-block:: bash

      bash scripts/ci/run-grid-phase2.sh

6. To re-materialise top configs or plots later, call
   ``python scripts/ci/recheck_topk_from_wandb.py`` followed by
   ``python experiments/reporting.py``-style helpers from within ``reports/``.

Logging & Observability
-----------------------

- All major scripts call ``utils.logging.maybe_init_wandb``. If W&B credentials
  are unavailable, the helper returns a no-op logger so tests still run.
- ``scripts/ci/wandb_utils.sh`` wraps ``wandb sweep`` / ``wandb agent`` and
  normalises sweep IDs, making local and CI execution consistent.
- Sweep agents emit structured JSON under ``logs/`` to simplify debugging.

Testing & Quality Gates
-----------------------

- ``pytest`` is configured via ``pytest.ini``. Tests rely on tiny samples stored
  under ``samples/`` and mock W&B handles.
- Linting and formatting hooks are configured through ``pyproject.toml`` and
  ``pre-commit`` (see ``README.md`` for setup commands).
- Key invariants:

  * Keep shared sweep knobs in sync between JEPA and contrastive specs.
  * Ensure dataset loaders honour ``--cache-dir`` to avoid duplicate featurisation.
  * Update documentation (this file + ``docs/kid_friendly_overview.rst``) when
    adding or renaming stages, scripts or sweep knobs.

Contribution Checklist
----------------------

1. Update or add unit tests under ``tests/`` for any new behaviour.
2. Document new scripts or workflow changes in this handbook and in the
   kid-friendly overview.
3. Keep automation resilient: prefer extending the existing factories and CLI
   subcommands rather than creating one-off entry points.
4. Run relevant CI wrappers locally (e.g., ``bash scripts/ci/run-pretrain.sh``)
   before submitting substantial changes.

Happy hacking!
