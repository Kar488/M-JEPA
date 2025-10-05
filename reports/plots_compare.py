"""Method comparison plotting helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def comparison_bar(
    datasets: Sequence[str], metrics: Mapping[str, Sequence[float]], *, ylabel: str
):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(metrics), 1)
    for idx, (label, values) in enumerate(metrics.items()):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Method comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def radar_plot(metrics: Mapping[str, float]):
    labels = list(metrics.keys())
    values = list(metrics.values())
    if not labels:
        raise ValueError("No metrics provided")
    values.append(values[0])
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Performance radar")
    return fig
