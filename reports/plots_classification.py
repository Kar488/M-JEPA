"""Plots for classification experiments."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc


COLORS = plt.cm.tab10.colors


def plot_roc_pr_curves(
    curves: Mapping[str, Mapping[str, Sequence[Tuple[float, float]]]],
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    roc_ax, pr_ax = axes
    for idx, (label, payload) in enumerate(curves.items()):
        roc_points = payload.get("roc", [])
        pr_points = payload.get("pr", [])
        if roc_points:
            fpr, tpr = zip(*roc_points)
            roc_ax.plot(fpr, tpr, label=label, color=COLORS[idx % len(COLORS)])
            roc_ax.set_xlabel("FPR")
            roc_ax.set_ylabel("TPR")
            roc_ax.set_title("ROC")
        if pr_points:
            recall, precision = zip(*pr_points)
            pr_ax.plot(recall, precision, label=label, color=COLORS[idx % len(COLORS)])
            pr_ax.set_xlabel("Recall")
            pr_ax.set_ylabel("Precision")
            pr_ax.set_title("Precision-Recall")
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    roc_ax.legend()
    pr_ax.legend()
    return fig


def reliability_diagram(
    probs: Sequence[float], labels: Sequence[int], *, bins: int = 10
):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs, bin_edges, right=True) - 1
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    expected = []
    observed = []
    counts = []
    for idx in range(bins):
        mask = bin_ids == idx
        if not np.any(mask):
            expected.append(bin_centres[idx])
            observed.append(np.nan)
            counts.append(0)
            continue
        expected.append(np.mean(probs[mask]))
        observed.append(np.mean(labels[mask]))
        counts.append(np.sum(mask))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.plot(expected, observed, marker="o")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    return fig, pd.DataFrame(
        {"expected": expected, "observed": observed, "count": counts}
    )


def confusion_matrix_plot(matrix: np.ndarray, labels: Sequence[str]):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix")
    return fig
