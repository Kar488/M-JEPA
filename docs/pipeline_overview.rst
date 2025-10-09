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

Only initiator stages (``pretrain-agent`` and the grid sweeps it launches) mint
new encoder lineages. Downstream jobs bind to existing ``PRETRAIN_EXP_ID`` values
and emit fresh ``EXP_ID`` directories so prior artifacts remain untouched.

Identifier Hierarchy
--------------------

- ``PRETRAIN_EXP_ID`` – canonical ID for the encoder lineage and frozen assets.
- ``GRID_EXP_ID`` – companion ID used by grid sweeps (phase 1 and phase 2).
- ``EXP_ID`` – unique run directory for the current workflow invocation.

When the lineage is frozen, ``GRID_EXP_ID`` usually equals ``PRETRAIN_EXP_ID``.
Automation may override this (e.g. for cross-lineage sweeps) via
``FORCE_UNFREEZE_GRID=1`` or explicit ``GRID_EXP_ID`` bindings.

Stage Orchestration
-------------------

#. **Pretrain.** ``pretrain-agent`` writes encoder checkpoints and seeds the
   lineage.
#. **Phase 1 sweep.** ``phase1-agent`` runs the coarse sweep, writing into
   ``$EXPERIMENTS_ROOT/$GRID_EXP_ID/grid/phase1``.
#. **Phase 2 sweep.** ``phase2-agent`` rechecks and exports winning configs,
   writing ``grid/phase2_*`` artifacts.
#. **Tox21 grading (benchmark stage).** ``tox21-agent`` evaluates the encoder on
   Tox21 tasks and writes ``bench/encoder_frozen.ok`` on success.
#. **Finetune / Benchmark / Report.** ``finetune-agent`` and ``report-agent``
   consume the frozen lineage and store outputs under the new ``EXP_ID``.

The CI/CD pipeline automatically resumes from the latest completed stage, using
GitHub workflows to schedule jobs and Vast.ai workers to execute sweeps and
benchmarks. Freeze markers prevent accidental overwrites; set
``FORCE_UNFREEZE_GRID=1`` or remove the marker only when intentionally rebuilding
an encoder lineage.

For policy details and override semantics, read :doc:`frozen_lineage_policy`.
