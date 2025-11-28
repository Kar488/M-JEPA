# Fine-tune seeds and epoch knobs

This note summarises where fine-tune seed/epoch defaults live and how CI/sweeps
override them when searching for configurations that keep quality while
constraining runtime.

## Default CLI/YAML values
- `scripts/default.yaml` is the baseline loaded by `scripts/train_jepa.py`.
  - Fine-tune defaults: `epochs: 10` and `seeds: [0]`, with early stopping via
    `patience: 10`. These are minimal values that still give the linear head a
    chance to converge on typical MoleculeNet splits.
- `scripts/commands/finetune.py` resolves seeds by preferring CLI `--seeds` then
  falling back to the YAML `finetune.seeds` list, so per-run overrides are easy
  without editing configs.

## CI and shell wrappers
- `scripts/ci/run-finetune.sh` sets `BESTCFG_NO_EPOCHS=1` so Phase-2 best-config
  imports skip overriding fine-tune epochs; CI relies on the requested epochs
  from the YAML/CLI rather than sweep suggestions, avoiding accidental
  regression to very short runs.
- `scripts/ci/train_jepa_ci.yml` wires the CI fine-tune stage to
  `finetune_epochs: ${FINETUNE_EPOCHS:-10}` and exposes up to three seeds via
  `FINETUNE_SEED_{0,1,2}` (defaulting to `[0]`). This keeps CI aligned with the
  10-epoch, 1–3 seed regime unless explicitly tightened for debugging.

## Sweep templates
- Phase-1 sweeps (`sweeps/sweep_phase1_*.yaml`) and the Phase-2 grid sweep
  (`sweeps/grid_sweep_phase2.yaml`) favour short probes to stay within budget:
  - Phase-1: `finetune_epochs: {values: [1]}` with seeds injected by CI
    (`[1,2,3]`), intended only for ranking backbones rather than final quality.
  - Phase-2 grid: explores `finetune_epochs` in `[1, 3]` with seeds `[0,1]` to
    validate leaderboard winners quickly.
- These sweeps are tuned for speed; use the CI/default settings above for
  quality-oriented runs.

## Practical recommendations
- For quality-sensitive fine-tuning, stick with **10 epochs and 1–3 seeds**
  (default), optionally extending epochs if validation metrics plateau early but
  remain noisy. Single-seed runs are acceptable for quick checks; multi-seed
  averaging (`--seeds 0 1 2`) stabilises metrics without excessive cost.
- Avoid the 1–3 epoch sweep presets unless performing comparative sweeps; they
  trade quality for throughput and are not meant for final head training.
