
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