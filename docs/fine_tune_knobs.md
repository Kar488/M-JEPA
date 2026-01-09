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
- Hybrid tox21 evaluations (`TOX21_EVALUATION_MODE=hybrid`) always use
  `TOX21_EPOCHS` for the fine-tune horizon and ignore sweep best-config
  overrides. The hybrid schedule splits those epochs into freeze/partial/full
  phases and applies warmup+cosine decay using the CI knobs.
- `TOX21_CHECKPOINT_METRIC` selects which validation metric drives best-checkpoint
  selection and early stopping for the tox21 stage (defaults to `pr_auc`; set
  `TOX21_CHECKPOINT_METRIC=roc_auc` in `ci-vast.yml` to revert to ROC-AUC).
  `threshold_metric` stays reserved for post-hoc decision-threshold tuning.

## Metric roles (checkpoint vs threshold vs early stop)
- **`checkpoint_metric`**: validation metric used to pick the best checkpoint
  and (when set) to drive early stopping in the Tox21 case study. Defaults to
  `pr_auc` in `scripts/default.yaml` and can be overridden with
  `--checkpoint-metric` / `TOX21_CHECKPOINT_METRIC`.
- **`early_stop_metric`**: the internal training loop metric. Tox21 sets this to
  `checkpoint_metric` when provided; otherwise it falls back to `val_auc` for
  full-finetune runs or `val_loss` for frozen/probe runs.
- **`threshold_metric`**: validation metric used only for post-hoc decision
  threshold tuning and reporting. It does **not** affect early stopping or
  checkpoint selection.

## Per-task Tox21 overrides
- Per-task YAML overrides are loaded by `scripts/commands/tox21.py` and applied
  per assay before calling `experiments/case_study.run_tox21_case_study`.
- Location: `scripts/ci/per_task_hparams/tox21_hparams.yaml` (or your own file
  passed with `--per-task-hparams`).
- Supported keys include:
  - `head_lr`, `encoder_lr`, `layerwise_decay`
  - `pos_weight`, `class_weights`, `dynamic_pos_weight`, `oversample_minority`
  - `use_focal_loss`, `focal_gamma`
  - `threshold_metric`, `checkpoint_metric`
  - `calibrate_probabilities` (apply post-hoc calibration on the validation split)
  - `calibration_method` (per-task override → CLI → default). Supported methods:
    `temperature`, `isotonic`, and `platt` (Platt scaling).
  - `__hybrid__` section for shared hybrid defaults (e.g., `head_lr`,
    `encoder_lr`, `layerwise_decay`).

Calibration method resolution is: per-task override → `--calibration-method`
→ default `temperature`. Unsupported values raise a clear error at runtime.

## Hybrid early stopping guard
- Hybrid training is split into freeze/partial/full phases; early stopping can
  terminate training before the full schedule completes.
- To guard against that, set `--hybrid-early-stop-min-epochs` (or
  `case_study.hybrid_early_stop_min_epochs` in `scripts/default.yaml`) to require
  a minimum number of epochs before early stopping can trigger. When unset, the
  previous behaviour is preserved.

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
