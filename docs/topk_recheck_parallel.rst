Top-K Recheck Parallelisation Plan
=================================

Motivation
----------

The Phase 2 re-check currently replays the top configurations sequentially: it
loops over each pending seed, spawns ``run_once`` and waits for that process to
finish before moving to the next one. The launcher therefore keeps a single GPU
busy even when more devices are available, unlike the sweep agents that split
work across multiple GPUs. ``run_once`` itself defers to ``micromamba`` to launch
``scripts/train_jepa.py`` with the selected configuration and W&B metadata, but
it does not currently receive any GPU affinity hints or execute in parallel with
other seeds.

Design Goals
------------

* Keep the lineage policy intact: every re-check run must continue to reuse the
  frozen encoder lineage discovered during grading, respect
  ``RECHECK_EXP_ID``/``RECHECK_SEED`` bookkeeping and land in the same W&B naming
  scheme.
* Offer a drop-in optimisation: existing CI wrappers (``run_phase2_recheck_stage``)
  and manual invocations should still work, but can opt into multi-GPU parallelism
  via environment variables that mirror the sweep agent knobs.
* Remain deterministic: parallel launches must not alter hyper-parameters,
  random seeds or output destinations, and retries/timeouts should behave exactly
  as today.

High-Level Proposal
-------------------

1. **Discover GPU slots.**
   Inspect ``CUDA_VISIBLE_DEVICES`` (or fall back to ``torch.cuda.device_count``)
   at the start of ``main`` to build an ordered list of GPU identifiers. Allow
   ``PHASE2_AGENT_COUNT`` or a dedicated ``PHASE2_RECHECK_AGENT_COUNT`` override
   to cap concurrency so users can match the sweep agent footprint.

2. **Build a launch queue.**
   The existing planning phase already assembles ``(config_idx, cfg, seeds, ...)``
   tuples. Extend it to expand each ``(config_idx, seed)`` pair into an explicit
   job description that includes the GPU slot once one is assigned. Preserve the
   current resume logic that skips completed seeds.

3. **Introduce a lightweight scheduler.**
   Replace the inner ``for seed in pending_seeds`` loop with a worker pool that
   draws from the job queue. A simple implementation can use
   ``concurrent.futures.ThreadPoolExecutor`` (threads suffice because each worker
   just invokes ``subprocess.Popen``) or a manual dispatcher built on ``queue.Queue``.
   Each worker calls a refactored ``run_once`` that accepts an optional
   ``device_env`` argument (e.g. ``{"CUDA_VISIBLE_DEVICES": "2"}``) before
   delegating to ``micromamba``.

4. **Propagate GPU affinity.**
   Extend ``run_once`` with an optional ``device_id`` parameter. When provided, it
   should clone the existing ``env`` dictionary, set ``CUDA_VISIBLE_DEVICES`` to
   the designated GPU, and forward ``--device``/``--devices`` flags only if they
   are already present in the configuration. This keeps the frozen encoder logic
   intact while ensuring each seed binds to a distinct GPU.

5. **Preserve logging and monitoring.**
   Logs already write to per-seed files under ``recheck_{method}_seed{seed}.log``;
   these file names remain unique under parallel execution. The heartbeat and
   sentinel files that ``run_phase2_recheck_stage`` expects do not depend on the
   launch order, so no changes are needed beyond ensuring the parent process waits
   for all futures before computing summary statistics.

6. **Make concurrency opt-in.**
   Default to sequential execution when no GPU mask is discoverable or when the
   agent count is ``1``. Document the new environment variables in the CI stage
   script so that automation can set ``PHASE2_RECHECK_AGENT_COUNT=${PHASE2_AGENT_COUNT}``
   when multi-GPU hosts are available.

Trade-Offs & Considerations
---------------------------

* Parallel seeds will reserve more GPU memory simultaneously; the scheduler should
  surface clear error messages when a worker OOMs so that the retry logic can kick
  in per seed rather than failing the entire batch.
* Launch-time contention on shared input pipelines (dataset caching, disk I/O)
  may increase, so default concurrency should remain conservative (e.g. clamp to
  the number of visible GPUs).
* Extending ``run_once`` introduces a small refactor risk; adding a thin wrapper
  (e.g. ``launch_seed(cfg, seed, gpu_id)``) can keep the public surface narrow and
  make unit tests easier.
* Future extensions could swap the process-level parallelism for distributed
  data-parallel training by forwarding ``--devices N`` into ``train_jepa.py``, but
  that would require broader changes to the training stack (DDP initialisation,
  checkpoint naming). The proposed scheduler keeps that option open without
  entangling it with the immediate speedup.

Next Steps
----------

* Implement the scheduler and ``run_once`` refactor in ``scripts/ci/recheck_topk_from_wandb.py``.
* Update ``scripts/ci/stage.sh`` to document the new environment variables and
  forward defaults derived from ``PHASE2_AGENT_COUNT``.
* Add regression coverage that exercises two synthetic seeds in parallel using a
  fake GPU mask to confirm log separation, environment propagation and summary
  aggregation.

Operational Updates
-------------------

* ``recheck_topk_from_wandb.py`` now falls back to ``nvidia-smi`` when
  ``torch`` is unavailable, so GitHub runners without a populated
  ``CUDA_VISIBLE_DEVICES`` mask still expose their GPU indices to the scheduler.
* Each configuration write refreshes ``grid/best_grid_config.json`` and the
  consolidated summary. If the process terminates early, it stamps
  ``grid/phase2_recheck/recheck_incomplete.ok`` so the next invocation resumes
  pending seeds instead of restarting from scratch.
* A successful run clears the incomplete marker and rewrites
  ``recheck_done.ok`` to keep downstream stages (export, pretrain) aligned with
  the latest winner.
