from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def summarize_with_ci(df: pd.DataFrame, metrics: List[str], seeds_col: str = "seeds") -> pd.DataFrame:
    out = df.copy()
    n = out[seeds_col].iloc[0] if seeds_col in out.columns else 3
    if isinstance(n, list): n = len(n)
    for m in metrics:
        if m in out.columns and f"{m}_std" in out.columns:
            out[f"{m}_ci95"] = 1.96 * out[f"{m}_std"] / max(1, np.sqrt(n))
    return out

def plot_topn_bar(df: pd.DataFrame, metric: str, top_n: int, out_path: str, title: Optional[str] = None) -> None:
    df2 = df.sort_values(metric, ascending=False).head(top_n)
    labels = (df2["method"] + " | hid=" + df2["hidden_dim"].astype(str) + " L=" +
              df2["num_layers"].astype(str) + " m=" + df2["mask_ratio"].astype(str)).tolist() \
             if "method" in df2.columns else df2.index.astype(str).tolist()
    vals = df2[metric].to_numpy()
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    if title: plt.title(title)
    plt.tight_layout(); _ensure_dir(os.path.dirname(out_path)); plt.savefig(out_path, dpi=200); plt.close()

def pivot_heatmap(df: pd.DataFrame, metric: str, x: str, y: str, out_path: str, title: Optional[str] = None) -> None:
    table = df.pivot_table(values=metric, index=y, columns=x, aggfunc="mean")
    plt.figure()
    plt.imshow(table.values, aspect="auto")
    plt.xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    plt.yticks(range(table.shape[0]), table.index)
    if title: plt.title(title)
    plt.colorbar(); plt.tight_layout(); _ensure_dir(os.path.dirname(out_path)); plt.savefig(out_path, dpi=200); plt.close()

def save_ranked_csv(df: pd.DataFrame, metric: str, out_csv: str, top_n: int = 50) -> None:
    df.sort_values(metric, ascending=False).head(top_n).to_csv(out_csv, index=False)

def build_full_report(df: pd.DataFrame, metric: str, out_dir: str, top_n: int = 20) -> None:
    _ensure_dir(out_dir)
    save_ranked_csv(df, metric=metric, out_csv=os.path.join(out_dir, f"ranked_top{top_n}.csv"), top_n=top_n)
    plot_topn_bar(df, metric=metric, top_n=top_n, out_path=os.path.join(out_dir, f"top{top_n}_bar.png"),
                  title=f"Top-{top_n} by {metric}")
    if {"mask_ratio", "gnn_type"} <= set(df.columns):
        pivot_heatmap(df[df["method"]=="jepa"], metric=metric, x="mask_ratio", y="gnn_type",
                      out_path=os.path.join(out_dir, "heatmap_jepa_mask_gnn.png"),
                      title=f"JEPA {metric}: mask_ratio × gnn_type")
    if {"ema_decay", "hidden_dim"} <= set(df.columns):
        pivot_heatmap(df[df["method"]=="jepa"], metric=metric, x="ema_decay", y="hidden_dim",
                      out_path=os.path.join(out_dir, "heatmap_jepa_ema_hidden.png"),
                      title=f"JEPA {metric}: ema_decay × hidden_dim")
