"""Plot experiment results and diagnostics.

Requires :mod:`matplotlib`, :mod:`pandas`, and :mod:`scikit-learn` for
generating plots and computing evaluation metrics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    """Create directory ``p`` if it does not already exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out_path: str) -> None:
    """Save ``fig`` to ``out_path`` and close it."""
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(df: pd.DataFrame, out_path: str) -> None:
    """Plot ROC curve if ``y_true`` and ``y_score`` columns exist."""
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
    auc = roc_auc_score(df["y_true"], df["y_score"])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    _save(fig, out_path)


def plot_loss(df: pd.DataFrame, out_path: str) -> None:
    """Plot loss curve if ``epoch`` and ``loss`` columns exist."""
    fig, ax = plt.subplots()
    ax.plot(df["epoch"], df["loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    _save(fig, out_path)


def plot_bar(df: pd.DataFrame, out_path: str, label: str, value: str) -> None:
    """Plot bar chart for ``label`` vs ``value`` columns."""
    fig, ax = plt.subplots()
    ax.bar(df[label], df[value])
    ax.set_xlabel(label)
    ax.set_ylabel(value)
    ax.set_title("Bar Chart")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_all(csv_path: str, out_dir: str) -> Iterable[str]:
    """Generate available plots from ``csv_path`` into ``out_dir``.

    Returns an iterable of output file paths.
    """
    df = pd.read_csv(csv_path)
    out_paths = []

    if {"y_true", "y_score"} <= set(df.columns):
        out = os.path.join(out_dir, "roc_curve.png")
        plot_roc(df, out)
        out_paths.append(out)

    if {"epoch", "loss"} <= set(df.columns):
        out = os.path.join(out_dir, "loss_curve.png")
        plot_loss(df, out)
        out_paths.append(out)

    if {"category", "value"} <= set(df.columns):
        out = os.path.join(out_dir, "bar_chart.png")
        plot_bar(df, out, "category", "value")
        out_paths.append(out)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("csv", help="Path to evaluation CSV file")
    parser.add_argument(
        "--out_dir", default="reports", help="Directory in which to store figures"
    )
    args = parser.parse_args()

    plot_all(args.csv, args.out_dir)


if __name__ == "__main__":
    main()