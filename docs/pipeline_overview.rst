Pipeline Overview
=================

.. admonition:: Summary

   - Frozen encoder lineages are immutable.
   - Dependent runs must set ``PRETRAIN_EXP_ID`` to reuse them.
   - To rebuild, remove or override the freeze marker explicitly.

Lineage Hierarchy
-----------------

.. code-block:: text

   /data/mjepa/experiments/
   ├── $PRETRAIN_EXP_ID/   ← frozen encoder lineage (e.g., 1759825317)
   │   ├── artifacts/
   │   └── bench/encoder_frozen.ok
   ├── $EXP_ID/            ← new run consuming the frozen lineage (e.g., 18327125156)
   │   ├── finetune/
   │   ├── grid/phase2_recheck/
   │   ├── tox21/
   │   └── report/

Only initiator stages (``pretrain-agent`` and the Phase‑1 grid sweep it
launches) mint new encoder lineages. Phase‑1 now assigns a fresh ``EXP_ID`` and
sets ``GRID_EXP_ID`` to that value so its outputs live under a dedicated
directory. Downstream jobs bind to existing ``PRETRAIN_EXP_ID`` values and emit
fresh ``EXP_ID`` directories so prior artifacts remain untouched.

Identifier Hierarchy
--------------------

- ``PRETRAIN_EXP_ID`` – canonical ID for the encoder lineage and frozen assets.
- ``GRID_EXP_ID`` – companion ID used by grid sweeps (phase 1 and phase 2).
- ``EXP_ID`` – unique run directory for the current workflow invocation.

Before the encoder is frozen, ``GRID_EXP_ID`` tracks the active Phase‑1 sweep
(``GRID_EXP_ID=$EXP_ID``). Once the Tox21 gate stamps
``bench/encoder_frozen.ok``, automation keeps ``PRETRAIN_EXP_ID`` fixed to the
frozen lineage and reuses its ``GRID_EXP_ID`` in read-only mode. Override this
behaviour (for example, to rebuild a sweep) by setting ``FORCE_UNFREEZE_GRID=1``
or by passing explicit ``GRID_EXP_ID`` bindings.

Stage Orchestration
-------------------

#. **Pretrain.** ``pretrain-agent`` writes encoder checkpoints and seeds the
   lineage.
#. **Phase 1 sweep.** ``phase1-agent`` runs the coarse sweep, writing into
   ``$EXPERIMENTS_ROOT/$EXP_ID/grid/phase1`` while the sweep is active.
#. **Phase 2 sweep.** ``phase2-agent`` rechecks and exports winning configs,
   writing ``grid/phase2_*`` artifacts.
#. **Tox21 grading (benchmark stage).** ``tox21-agent`` evaluates the encoder on
   Tox21 tasks and writes ``bench/encoder_frozen.ok`` on success. Hybrid mode
   (``TOX21_EVALUATION_MODE=hybrid``) runs a three-phase fine-tuning schedule
   with warmup+cosine LR decay driven by CI defaults.
#. **Finetune / Benchmark / Report.** ``finetune-agent`` and ``report-agent``
   consume the frozen lineage and store outputs under the new ``EXP_ID``.

   Fine-tune logs report per-rank loader shard sizes when running under DDP and
   include an estimated global batch count so readers can distinguish local
   shards from the global dataset footprint.

The CI/CD pipeline automatically resumes from the latest completed stage, using
GitHub workflows to schedule jobs and Vast.ai workers to execute sweeps and
benchmarks. Freeze markers prevent accidental overwrites; set
``FORCE_UNFREEZE_GRID=1`` or remove the marker only when intentionally
rebuilding an encoder lineage. ``FORCE_RERUN=stage1,stage2`` remains available
to selectively invalidate cached stages when experimenting.

For policy details and override semantics, read :doc:`frozen_lineage_policy`.


Configuration defaults and precedence
-------------------------------------

- **Local runs.** ``scripts/train_jepa.py`` loads defaults from
  ``scripts/default.yaml`` (for example, ``pretrain.epochs=100``,
  ``finetune.epochs=50``) and applies any explicit CLI flags on top.
- **CI/Vast runs.** ``scripts/ci/stage.sh`` builds each stage's argument vector
  from ``scripts/ci/train_jepa_ci.yml``. That YAML maps workflow environment
  variables (such as ``PRETRAIN_EPOCHS``/``FINETUNE_EPOCHS``) to the appropriate
  CLI flags, so values exported by ``.github/workflows/ci-vast.yml`` take
  precedence over ``scripts/default.yaml`` during automation.
- **Per-stage wrappers.** ``scripts/ci/run-*.sh`` invoke ``stage.sh`` and fill in
  stage-specific defaults before the CLI is built. For example,
  ``run-tox21.sh`` supplies its own ``FINETUNE_EPOCHS`` fallback when the
  workflow leaves it blank, and the wrappers clear ``BESTCFG_NO_EPOCHS`` at the
  start so stage-local defaults are applied deterministically.
- **Stage cleanup.** ``scripts/ci/stage.sh`` runs a defensive cleanup pass at
  both stage start and stage exit to terminate orphaned ``train_jepa.py`` /
  ``torchrun`` jobs tied to the active ``EXP_ID`` or experiment paths, so later
  stages do not hang on stale DDP workers. Torchrun launches are terminated by
  process group to ensure all ranks exit together. Set
  ``MJEPACI_DISABLE_CLEANUP=1`` to opt out or ``MJEPACI_CLEANUP_DRYRUN=1`` to log
  matches without killing. The same runner also writes per-stage lock files
  under ``$DATA_ROOT/locks`` to prevent overlapping runs; set
  ``MJEPACI_DISABLE_LOCKS=1`` to opt out.
- **Benchmark DDP policy.** The benchmark stage only honors multi-GPU settings
  when launched under ``torchrun``. If ``--devices`` is greater than 1 without a
  distributed launcher, the stage logs a warning and forces single-device
  execution to avoid misleading results.

Precedence summary: ``ci-vast.yml`` env vars → ``train_jepa_ci.yml`` argument
templates → CLI flags → ``scripts/default.yaml``. If a value is omitted at a
higher layer, the next layer supplies the default.

Sweep sizing recommendations
---------------------------

- **Phase 1 sweep size.** Use ``WANDB_COUNT`` (or
  ``PHASE1_JEPA_COUNT``/``PHASE1_CONTRAST_COUNT``) of **30** per method by
  default to ensure deep coverage of the backbone/seed grid. If you need a lean
  coarse sweep, you can trim to the 6–10 range while still keeping high odds of
  overlapping ``pair_id`` coverage across backbones.

- **Phase 2 sweep size.** Aim for **80** agents for the Bayesian refinement; this
  usually suffices to explore the neighbourhood around the Phase 1 winner while
  keeping costs reasonable. Drop to ~40 runs for quick-turn experiments or keep
  the default (100) when you have capacity and want higher-resolution surfaces
  on large GPU pools.

- **Wall-clock budgets.** From Phase 1 onward, allocate at least 1,500 minutes
  (25 hours) via ``HARD_WALL_MINS`` so agents can finish even on slower nodes or
  during heavy I/O. The extra cushion keeps Phase 2 fine-tuning and downstream
  grading from timing out when the 10 M-cache is already built; trim the wall
  only for smoke tests and keep the cache warmer in place to avoid stalls during
  early epochs.
