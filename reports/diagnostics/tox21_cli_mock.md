# Tox21 Baseline Diagnostics

This note documents two checks that mirror the CI `tox21` stage while using mocked
paths so they can run inside the repository workspace.

## 1. Direct CLI Mock

This reproduces the invocation emitted by the CI wrapper once argument
construction completes.

```bash
python -m scripts.commands.tox21 \
  --cache-dir /data/mjepa/cache/graphs_250k \
  --csv /srv/mjepa/data/tox21/data.csv \
  --tasks NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53 \
  --dataset tox21 \
  --tox21-dir /data/mjepa/experiments/19022127244/tox21 \
  --pretrain-epochs 70 \
  --finetune-epochs 18 \
  --pretrain-time-budget-mins 90 \
  --finetune-time-budget-mins 60 \
  --encoder-checkpoint /data/mjepa/experiments/19022127244/pretrain/encoder.pt \
  --encoder-manifest /data/mjepa/experiments/19022127244/artifacts/encoder_manifest.json \
  --encoder-source pretrain_frozen \
  --evaluation-mode pretrain_frozen \
  --epochs 50 \
  --patience 12 \
  --head-lr 0.001 \
  --encoder-lr 0.0001 \
  --weight-decay 0.0001 \
  --class-weights auto \
  --verify-match-threshold 0.98 \
  --use-wandb \
  --num-workers 8 \
  --prefetch-factor 2 \
  --persistent-workers 1 \
  --pin-memory 0 \
  --bf16 1 \
  --devices 2 \
  --device cuda \
  --lr 0.001 \
  --gnn-type gine \
  --num-layers 3
```

**Outcome.** The command exits successfully (status 0). Because `pandas` is
absent, the loader logs a fallback to its simplified path yet still emits the
aggregated metrics CSV under the mocked `$TOX21_DIR`.

## 2. Stage Wrapper Simulation

To match the CI harness more closely, the stage runner was executed with
`MJEPACI_STAGE_SHIM=/bin/true` so the wrapper performs all dependency and
micromamba checks without launching a real training job. The supporting
artifacts (`encoder_manifest.json`, `encoder.pt`, and
`grid/best_grid_config.json`) were stubbed under `tmp/experiments/19022127244/`
to satisfy the preflight assertions.

```bash
MJEPACI_STAGE_SHIM=/bin/true \
  EXPERIMENTS_ROOT=$PWD/tmp/experiments \
  EXP_ID=19022127244 PRETRAIN_EXP_ID=19022127244 GRID_EXP_ID=19022127244 \
  PRETRAIN_MANIFEST=$PWD/tmp/experiments/19022127244/artifacts/encoder_manifest.json \
  GRID_DIR=$PWD/tmp/experiments/19022127244/grid \
  bash scripts/ci/run-tox21.sh
```

**Outcome.** The wrapper resolves `/root/micromamba/envs/mjepa` as its Python
interpreter, imports `yaml` successfully, and proceeds to the shimmed stage
execution. With PyYAML present inside that micromamba environment, the run
completes without triggering the retry/repair loop that would otherwise explain
an early exit.
