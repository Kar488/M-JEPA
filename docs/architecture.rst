Architecture & Orchestration
=============================

.. note::

   **Summary**
   - Frozen encoder lineages are immutable.
   - Dependent runs must set ``PRETRAIN_EXP_ID`` to reuse them.
   - To rebuild, remove or override the freeze marker explicitly.

Sequence Diagram
----------------

::

   PretrainAgent -> Phase1Agent : seed GRID_EXP_ID
   Phase1Agent  -> Phase2Agent : reuse grid/phase1
   Phase2Agent  -> Tox21Agent  : export grid/phase2_*
   Tox21Agent   -> Experiments : stamp bench/encoder_frozen.ok
   Tox21Agent   -> VastCI      : signal frozen lineage
   VastCI       -> NextRun     : allocate EXP_ID
   NextRun      -> FinetuneAgent : read $PRETRAIN_EXP_ID/artifacts
   NextRun      -> ReportAgent   : assemble reports (read-only)

Once the freeze marker is present, subsequent runs declare ``PRETRAIN_EXP_ID``
and reuse the grid snapshots in read-only mode.

CI, Vast & Resume Logic
-----------------------

- **GitHub Actions** schedules ``scripts/ci/run-pretrain.sh`` to create new
  lineages when no freeze marker exists.
- **Vast.ai workers** execute ``run-grid-or-phase1.sh`` and ``run-grid-phase2.sh``
  sweeps, writing logs to ``logs/phase*`` and artifacts under
  ``$EXPERIMENTS_ROOT/$GRID_EXP_ID``.
- After ``tox21-agent`` writes ``bench/encoder_frozen.ok``, GitHub Actions and
  Vast respect the frozen lineage by binding ``PRETRAIN_EXP_ID`` for all dependent
  stages. ``STRICT_FROZEN=1`` enforces commit parity; ``ALLOW_CODE_DRIFT_WHEN_FROZEN``
  relaxes the check for exploratory reruns.
- Resume logic reads freeze markers and timestamps to skip completed stages while
  allowing ``FORCE_RERUN`` overrides when manual intervention is needed.

Cross-reference :doc:`pipeline_overview` for directory structures and
:doc:`frozen_lineage_policy` for flag semantics.
