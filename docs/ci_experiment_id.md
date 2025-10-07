# CI Experiment Identifiers

This note captures the conventions adopted for Vast/CI runners after auditing
experiment identifiers and artifact hand-offs.

## Canonical experiment ID (`EXP_ID`)

* `EXP_ID` is the numeric identifier for an experiment directory under
  `$EXPERIMENTS_ROOT/<EXP_ID>/` (defaults to `/data/mjepa/experiments`).
* `run-pretrain.sh` generates an `EXP_ID` when one is not provided and writes the
  state to both the canonical
  `$EXPERIMENTS_ROOT/<EXP_ID>/pretrain_state.json` **and** the legacy
  `$EXPERIMENTS_ROOT/pretrain_state.json` for backward compatibility.
* Every stage sources `scripts/ci/common.sh`, which resolves `EXP_ID` from (in
  order): a per-experiment state file, the `EXP_ID` environment variable, or,
  for pretrain, the generated fallback.

## Directory layout

* Canonical artifacts (manifest, checkpoints, tox21 env) live inside
  `$EXPERIMENTS_ROOT/<EXP_ID>/`.
* Downstream scripts never embed W&B run identifiers in filesystem paths; the
  W&B ID is now only used for logging metadata.
* `PRETRAIN_STATE_FILE_CANONICAL` always points at the per-experiment state
  file. The legacy state file is kept in sync via the pretrain stage and logged
  as deprecated.

## Artifact collection

* The GitHub workflow ensures the remote `artifacts/` directory exists before
  running `rsync`, eliminating the previous `change_dir` warnings.
* Transfers warn (with `::warning::`) when optional files are absent but fail
  the job if `encoder.pt`, `encoder_manifest.json`, `pretrain.json`, or
  `pretrain_state.json` are missing after sync.
* `Fetch pretrain state snapshot` captures the resolved paths and exports them
  as job outputs so that later jobs use the same experiment root.

## EXP_ID flow into tox21

* `run-tox21.sh` reports the resolved `EXP_ID`, manifest path, state path and
  tox21 environment file at start-up.
* The tox21 GitHub job exports the canonical paths from the pretrain job output
  and forwards them to Vast via the appleboy `envs` list. The stage now invokes
  Python through the project environment (`resolve_ci_python`) instead of a bare
  `python` binary.

## Acceptance checks

* `tests/ci/test_ci_flow.py` performs a shimmed dry-run to verify the pretrain
  and tox21 scripts write/read the canonical files, inspects the workflow for
  the artifact warnings, and confirms tox21 uses the resolved Python command.
