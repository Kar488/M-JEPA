Frozen Encoder Lineage Policy
=============================

.. admonition:: Summary

   - Frozen encoder lineages are immutable.
   - Dependent runs must set ``PRETRAIN_EXP_ID`` to reuse them.
   - To rebuild, remove or override the freeze marker explicitly.

Definition
----------

A *frozen encoder lineage* is any pretraining experiment whose encoder artifacts
are locked for reuse by downstream stages. Once the Tox21 grading stage succeeds
and writes the freeze marker, all future runs must treat the lineage as
read-only. New experiments reference the frozen encoder via ``PRETRAIN_EXP_ID``
(and, when applicable, ``GRID_EXP_ID``) while emitting their own ``EXP_ID``
folders for writable outputs.

Freeze Marker
-------------

- **Path:** ``$EXPERIMENTS_ROOT/$PRETRAIN_EXP_ID/bench/encoder_frozen.ok``
- **Owner:** ``tox21-agent`` after successful grading
- **Meaning:** encoder artifacts, grid specs, and configs are immutable

Removing the marker or forcing overrides is the only way to make the lineage
writable again.

Behaviour Matrix
----------------

.. list-table::
   :header-rows: 1

   * - Aspect
     - Unfrozen Lineage
     - Frozen Lineage
   * - Encoder artifacts
     - Writable during pretrain/phase sweeps
     - Read-only; reused via ``PRETRAIN_EXP_ID``
   * - Grid directories
     - Writable under ``$GRID_EXP_ID``
     - Read-only snapshots, reused in future runs
   * - New runs
     - Extend existing ``EXP_ID`` tree
     - Always allocate a fresh ``EXP_ID``
   * - Stage execution
     - ``pretrain → phase1 → phase2 → tox21 → report`` produces artifacts
     - Later runs skip pretrain/phase stages unless forced
   * - CI enforcement
     - CI seeds new lineage on demand
     - CI binds ``PRETRAIN_EXP_ID`` and aborts on drift unless overridden

Override Flags
--------------

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``FORCE_UNFREEZE_GRID=1``
     - Rebuild the lineage grid and regenerate sweeps even if frozen.
   * - ``ALLOW_CODE_DRIFT_WHEN_FROZEN=1``
     - Skip commit hash checks for dependent stages reading the frozen lineage.
   * - ``STRICT_FROZEN=1``
     - Enforce commit parity before consuming frozen artifacts.
   * - ``FORCE_RERUN=stage1,stage2``
     - Re-run selected stages (comma-separated) despite cached outputs.
   * - ``PRETRAIN_EXP_ID=<id>`` / ``GRID_EXP_ID=<id>``
     - Bind orchestration to a specific lineage and grid snapshot.

Run Relationships
-----------------

- **Frozen (read-only):** ``phase1-agent``, ``phase2-agent``, ``tox21-agent``,
  and ``report-agent`` read artifacts from the frozen lineage and write
  stage-specific logs under a new ``EXP_ID``.
- **Writable (new lineage):** ``pretrain-agent`` creates encoder artifacts and
  seeds the lineage. When ``FORCE_UNFREEZE_GRID=1`` is set, ``phase1-agent`` and
  ``phase2-agent`` can rebuild the grid for the lineage owner.
- **Finetune runs:** ``finetune-agent`` always consumes frozen artifacts via
  ``PRETRAIN_EXP_ID`` and produces outputs under ``$EXP_ID/finetune``.

Lifecycle
---------

#. ``pretrain-agent`` produces encoder checkpoints under a new ``EXP_ID`` and
   ``GRID_EXP_ID``.
#. ``phase1-agent`` and ``phase2-agent`` populate ``grid/phase1`` and
   ``grid/phase2_*`` for the lineage.
#. ``tox21-agent`` completes Tox21 grading (benchmark stage) and writes
   ``bench/encoder_frozen.ok``.
#. Subsequent ``finetune-agent`` or ``report-agent`` runs set ``PRETRAIN_EXP_ID``
   (and ``GRID_EXP_ID`` when reusing sweeps) while the CI pipeline allocates a new
   ``EXP_ID`` for writable outputs.

For a visual walkthrough of directory relationships, see
:doc:`pipeline_overview`.
