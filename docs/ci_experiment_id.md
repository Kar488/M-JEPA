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
  for pretrain, the generated fallback. Phase‑1 sweeps now force
  `GRID_EXP_ID=$EXP_ID` so the sweep emits outputs under its own lineage before
  the encoder is frozen.

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

## Git runner vs. appleboy

* Vast jobs still execute on a dedicated GPU tenant rather than the
  self-hosted GitHub runner. The runner ("git runner") exists purely to launch
  workflows and sync artifacts; it has no GPUs and no access to the Vast mount
  points used by training scripts.
* Because of that split, `.github/workflows/ci-vast.yml` continues to call
  `appleboy/ssh-action@v1` for every stage that needs GPUs. The action connects
  to the Vast host, forwards the environment (e.g., `APP_DIR`,
  `TOX21_EXPLAIN_MODE`), and executes the same shell scripts the runner would
  have invoked locally.
* Removing appleboy would require co-locating the GitHub runner with the Vast
  tenant (or exposing the GPU filesystem over the network), neither of which is
  feasible under the current security policy. Keeping the SSH hop isolates the
  control plane from long-running training jobs while still letting CI stream
  logs and collect artifacts.

## Acceptance checks

* `tests/ci/test_ci_flow.py` performs a shimmed dry-run to verify the pretrain
  and tox21 scripts write/read the canonical files, inspects the workflow for
  the artifact warnings, and confirms tox21 uses the resolved Python command.
