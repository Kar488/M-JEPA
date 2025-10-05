"""Plotting helpers for pretraining diagnostics."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_axis(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return fig, ax
    return ax.figure, ax


def plot_metric_curves(
    histories: Sequence[pd.DataFrame], metric_key: str, *, label: str = "metric"
):
    fig, ax = _prepare_axis()
    for history in histories:
        if metric_key not in history:
            continue
        ax.plot(
            history.get("_step", np.arange(len(history))),
            history[metric_key],
            alpha=0.4,
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{label} vs steps")
    return fig


def plot_embedding_variance(variances: Mapping[str, Sequence[float]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in variances.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Variance")
    ax.set_title("Embedding variance per dimension")
    ax.legend()
    return fig


def plot_cosine_similarity(similarities: Mapping[str, Sequence[float]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in similarities.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Context vs target cosine")
    ax.legend()
    return fig


def plot_ema_drift(steps: Sequence[int], distances: Mapping[str, Sequence[float]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in distances.items():
        ax.plot(steps[: len(values)], values, label=name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Distance")
    ax.set_title("EMA drift")
    ax.legend()
    return fig
