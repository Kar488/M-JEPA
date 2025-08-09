
This drop includes:
- adapters/config.yaml : fill with your repo commands and paths.
- adapters/cli_runner.py : small wrapper to call baseline repos (train + embed).
- experiments/baseline_integration.py : glue for baseline pretrain/export.
- training/train_on_embeddings.py : sklearn heads for embeddings (with val/test support).
- experiments/grid_search.py : updated to support baselines, mean/std/CI across seeds.
- experiments/reporting.py : simple plots (top-N bar, heatmap) and CSV ranking.
- sweeps/zinc_small.yaml : example sweep spec; set baseline_* files to your paths.
- main.py : previously generated; patched to support baseline full-mode.

Example grid:
  python main.py --mode grid --sweep sweeps/zinc_small.yaml

Example full baseline run:
  python main.py --mode full --device cuda --method molclr     --baseline_unlabeled_file data/zinc_pubchem_train.csv     --label_train_dir data/esol_scaffold/train     --label_val_dir data/esol_scaffold/val     --label_test_dir data/esol_scaffold/test     --label_col ESOL

## Analytics

Weights & Biases can be used for experiment tracking. Authenticate with one of
the following methods before launching a run:

```
export WANDB_API_KEY=YOUR_API_KEY
## set it in repo secret
```

```python
import wandb
wandb.login(key="YOUR_API_KEY")
```

Enable logging by passing `--use_wandb` to the training script. For example:

```
python main.py --mode grid --sweep sweeps/zinc_small.yaml --use_wandb
```

## Reports

The `analysis/plot_results.py` utility reads evaluation CSV files and generates
ROC curves, loss curves, and bar charts. By default figures are written to the
`reports/` directory.

Example usage:

```
python analysis/plot_results.py path/to/eval.csv --out_dir reports
```

Sample outputs:

![ROC Curve](reports/roc_curve.png)
![Loss Curve](reports/loss_curve.png)
![Bar Chart](reports/bar_chart.png)

## Baselines

External baselines are included as git submodules under `third_party/`.
After cloning this repository run:

```
git submodule update --init --recursive
```

Each baseline exposes a training routine that can be invoked through the
`training.baselines.run_baseline` helper. Example:

```
from training.baselines import run_baseline
run_baseline(name="molclr", config_path="third_party/MolCLR/config.yaml")
```

You can also execute it from the command line:

```
python - <<'PY'
from training.baselines import run_baseline
run_baseline("molclr")
PY
```

## Unlabeled data downloads

The `scripts/download_unlabeled.py` helper streams SMILES strings from the
public ZINC and PubChem APIs and converts them into `GraphDataset` shards.
Each run produces `train/`, `val/` and `test/` directories under
`data/unlabeled/` containing parquet files with graph features.

Example:

```
python scripts/download_unlabeled.py --total 10000 --out-root data/unlabeled
```

**Rate limiting.** Requests to both APIs are throttled via a configurable
sleep interval (`--sleep`, default 0.5s) to stay within public quotas.

**Resume logic.** Download progress (current ZINC page and PubChem CID) is
stored in `progress.json`. Re‑run the script with `--resume` to continue from
the last checkpoint.

**Disk footprint.** With the default shard size each block of 1k molecules
requires roughly 5–8 MB on disk, so 1 M molecules will occupy on the order of
5–8 GB.
