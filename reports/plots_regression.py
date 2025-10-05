"""Regression-focused plotting utilities."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def parity_plot(
    y_true: Sequence[float], y_pred: Sequence[float], *, title: str = "Parity"
):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig


def residual_plots(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    atoms: Sequence[int] | None = None,
):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2 if atoms is not None else 1, figsize=(10, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual vs predicted")
    if atoms is not None:
        axes[1].scatter(atoms, residuals, alpha=0.6)
        axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel("# atoms")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residual vs atoms")
    fig.tight_layout()
    return fig


def learning_curve_plot(
    fractions: Sequence[float],
    metrics: Mapping[str, Sequence[float]],
    *,
    ylabel: str = "metric",
):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in metrics.items():
        ax.plot(fractions, values, marker="o", label=label)
    ax.set_xlabel("Train fraction")
    ax.set_ylabel(ylabel)
    ax.set_title("Learning curve")
    ax.legend()
    return fig


def seed_variance_plot(metrics: Mapping[str, Sequence[float]]):
    data = []
    for name, values in metrics.items():
        for value in values:
            data.append({"metric": name, "value": value})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="metric", y="value", ax=ax, inner="quartile")
    ax.set_title("Seed variance")
    return fig
