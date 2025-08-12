"""
Test plotting on a tiny results DataFrame.
- If outputs/small_grid_results.csv exists, use it.
- Otherwise, synthesize a small DataFrame.
It will save PNGs to outputs/.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def test_plot_small(wb):
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / "small_grid_results.csv"

    try:
        from utils.plotting import plot_hyperparameter_results
        has_utils = True
    except Exception:
        has_utils = False

    if csv.exists(): 
        df = pd.read_csv(csv)
    else:
        # Synthesize a minimal dataframe with expected columns
        df = pd.DataFrame(
            {
                "add_3d": [False, True, False, True],
                "mask_ratio": [0.1, 0.1, 0.2, 0.2],
                "contiguous": [False, True, False, True],
                "hidden_dim": [64, 64, 64, 64],
                "num_layers": [2, 2, 2, 2],
                "gnn_type": ["mpnn", "mpnn", "gcn", "gcn"],
                "ema_decay": [0.95, 0.95, 0.99, 0.99],
                "roc_auc": [0.62, 0.70, 0.66, 0.74],
                "pr_auc": [0.58, 0.65, 0.60, 0.68],
            }
        )

    if has_utils and "roc_auc" in df.columns:
        df_plot = df.copy()
        df_plot.index = df.apply(
            lambda r: f"3D={r['add_3d']} MR={r['mask_ratio']} C={r['contiguous']} "
            f"H={r['hidden_dim']} L={r['num_layers']} {r['gnn_type']} EMA={r['ema_decay']}",
            axis=1,
        )
        plot_hyperparameter_results(
            df_plot,
            metric="roc_auc",
            title="ROC-AUC Across Hyper-parameters (Tiny Test)",
            top_n=min(15, len(df_plot)),
            wb=wb,
        )
    else:
        # Fallback: simple matplotlib bar
        import matplotlib.pyplot as plt

        metric = "roc_auc" if "roc_auc" in df.columns else df.columns[-1]
        vals = pd.to_numeric(df[metric], errors="coerce").astype("float64").to_numpy()
        x = np.arange(len(vals), dtype=float)
        labels = [
            f"{r['gnn_type']}/{r['hidden_dim']}/{r['mask_ratio']}/{int(r['add_3d'])}"
            for _, r in df.iterrows()
        ]
        plt.bar(x, vals)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
        plt.title(f"{metric} Across Configs (Tiny Test)")
        plt.tight_layout()
        fig = plt.gcf()
        path = outdir / "tiny_grid_bar.png"
        plt.savefig(path)
        wb.log({"saved_plot": str(path)})
        wb.log({"tiny_hparam_plot": wb.Image(fig)})

    wb.log({"plot_test": "done"})
