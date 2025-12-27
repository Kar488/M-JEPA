# Fine-tune seeds and epoch knobs

This note summarises where fine-tune seed/epoch defaults live and how CI/sweeps
override them when searching for configurations that keep quality while
constraining runtime.

## Default CLI/YAML values
- `scripts/default.yaml` is the baseline loaded by `scripts/train_jepa.py`.
  - Fine-tune defaults: `epochs: 50`, `patience: 5`, and `seeds: [0]`. These are
    the repository-wide defaults for local runs unless a CLI flag overrides
    them.
- `scripts/commands/finetune.py` resolves seeds by preferring CLI `--seeds` then
  falling back to the YAML `finetune.seeds` list, so per-run overrides are easy
  without editing configs.

## CI and shell wrappers
- `scripts/ci/train_jepa_ci.yml` is the template the CI stage wrappers read. It
  maps `FINETUNE_EPOCHS` to the `--epochs` flag for the `finetune` subcommand
  (`finetune_epochs: ${FINETUNE_EPOCHS:-50}` for the main finetune stage, and
  `${FINETUNE_EPOCHS:-15}` for the Tox21 case-study stage). Environment values
  set by the workflow therefore override `scripts/default.yaml` for CI runs.
- `scripts/ci/run-finetune.sh` sets `BESTCFG_NO_EPOCHS=1` so Phase-2 best-config
  imports skip overriding fine-tune epochs; CI relies on the requested epochs
  from the YAML/CLI rather than sweep suggestions, avoiding accidental
  regression to very short runs.
- `scripts/ci/run-tox21.sh` also obeys `FINETUNE_EPOCHS`; when left unset it
  defaults to 10 epochs in the `end_to_end` path and 20 when evaluating a
  fine-tuned encoder, ensuring the tox21 baseline does not silently fall back to
  the shorter sweep-style probes.

## Sweep templates
- Phase-1 sweeps (`sweeps/sweep_phase1_*.yaml`) run with `finetune_epochs: 10`
  to keep cost bounded while still training the probe enough to rank backbones.
- The Phase-2 grid sweep (`sweeps/grid_sweep_phase2.yaml`) uses
  `finetune_epochs: 25` by default because Phase‑2 already filters to promising
  configs and can afford a longer probe.
- These sweeps are tuned for speed and ranking; use the CI/default settings
  above for quality-oriented runs.

## Practical recommendations
- For quality-sensitive fine-tuning, stick with the **50-epoch default and 1–3
  seeds**, optionally trimming to 20–30 epochs for faster iteration once you
  have verified convergence behaviour. Single-seed runs are acceptable for quick
  checks; multi-seed averaging (`--seeds 0 1 2`) stabilises metrics without
  excessive cost.
- Avoid the sweep-style settings (10–25 epochs) unless you are explicitly
  ranking backbones or probing hyper-parameters under tight budget constraints.
