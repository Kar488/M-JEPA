"""Plotting utilities for training and experiment results.

Functions in this module produce Matplotlib figures for visualising
training loss curves and hyper‑parameter search results. These plots
help identify which configurations perform best. Requires
``matplotlib``, ``pandas``, and ``seaborn``.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import logging

logger = logging.getLogger(__name__)



def plot_training_curves(
    curves: Dict[str, List[float]],
    title: str = "Training Loss",
    normalize: bool = False,
) -> None:
    """Plot nicely styled training loss curves for multiple models.

    This function uses seaborn’s whitegrid style and distinct colour palette
    to produce aesthetically pleasing line plots. Markers are added to
    emphasise individual epochs. Optionally, each loss curve can be
    normalised by its initial value to highlight relative improvement.

    Args:
        curves: Mapping from model names to lists of loss values.
        title: Title of the plot.
        normalize: If True, divide each loss curve by its first value.
    """
    # Set seaborn style for a clean look
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    palette = sns.color_palette("husl", n_colors=len(curves))
    for i, (name, losses) in enumerate(curves.items()):
        epochs = range(1, len(losses) + 1)
        if normalize and len(losses) > 0 and losses[0] != 0:
            norm_losses = [l / losses[0] for l in losses]
            y_vals = norm_losses
        else:
            y_vals = losses
        plt.plot(
            epochs,
            y_vals,
            label=name,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=5,
            color=palette[i],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Relative Loss" if normalize else "Loss")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_results(
    df: pd.DataFrame,
    metric: str,
    title: str = "Hyperparameter Search Results",
    top_n: int = 10,
) -> None:
    """Visualise hyper‑parameter search results using a bar plot.

    This function sorts the configurations by the given metric and displays
    only the top N entries to avoid overly cluttered plots. Bars are
    colour‑mapped by their metric values. For classification metrics
    (ROC‑AUC and PR‑AUC) higher values are better; for regression
    metrics (RMSE and MAE) lower values are better.

    Args:
        df: DataFrame with hyper‑parameters as index and a metric column.
        metric: The metric to visualise (column name).
        title: Title of the plot.
        top_n: Number of top configurations to display (default: 10). If
            top_n <= 0, all configurations will be shown.
    """
    if df.empty or metric not in df.columns:
        logger.warning("No data to plot.")
        return
    # Determine sorting order: descending for ROC/PR metrics, ascending otherwise
    ascending = metric not in {"roc_auc", "pr_auc"}
    df_sorted = df.sort_values(by=metric, ascending=ascending).copy()
    # Select top N rows if requested
    if top_n > 0 and len(df_sorted) > top_n:
        df_sorted = df_sorted.iloc[:top_n]
    # Reset index to generate positional labels
    df_sorted = df_sorted.reset_index(drop=True)
    # Normalise metric for colour mapping
    values = df_sorted[metric]
    norm_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    cmap = sns.color_palette("viridis", as_cmap=True)
    colors = [cmap(v) for v in norm_values]
    plt.figure(figsize=(8, max(4, len(df_sorted) * 0.5)))
    bars = plt.barh(y=range(len(df_sorted)), width=values, color=colors)
    # Build labels from configuration columns (except the metric columns)
    label_cols = [
        col
        for col in df_sorted.columns
        if col not in {metric, "roc_auc", "pr_auc", "rmse", "mae"}
    ]
    labels = [
        ", ".join(f"{col}={df_sorted.iloc[i][col]}" for col in label_cols)
        for i in range(len(df_sorted))
    ]
    plt.yticks(range(len(df_sorted)), labels, fontsize=7)
    plt.xlabel(metric)
    plt.title(title)
    # Annotate bars with metric values
    for bar, val in zip(bars, values):
        width = bar.get_width()
        offset = 0.01 if ascending else -0.01
        ha = "left" if ascending else "right"
        plt.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha=ha,
            fontsize=7,
        )
    plt.tight_layout()
    plt.show()
