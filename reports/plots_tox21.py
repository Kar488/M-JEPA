"""Tox21 specific plotting helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def enrichment_curve(ranks: Sequence[float], labels: Sequence[int]):
    order = np.argsort(ranks)
    labels = np.asarray(labels)[order]
    cum_actives = np.cumsum(labels)
    total_actives = np.sum(labels)
    fractions = np.linspace(0, 1, len(labels), endpoint=True)
    enrichment = cum_actives / max(total_actives, 1)
    random = fractions * total_actives / max(total_actives, 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fractions, enrichment, label="model")
    ax.plot(fractions, random, label="random", linestyle="--")
    ax.set_xlabel("Top-k fraction")
    ax.set_ylabel("Actives found")
    ax.set_title("Enrichment curve")
    ax.legend()
    return fig


def retention_vs_workload(
    retained: Mapping[str, Sequence[float]], workload: Sequence[float]
):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in retained.items():
        ax.plot(workload, values, marker="o", label=label)
    ax.set_xlabel("% removed")
    ax.set_ylabel("% actives retained")
    ax.set_title("Retention vs workload")
    ax.legend()
    return fig


def avoided_assays_bar(data: Mapping[str, float]):
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(data.keys())
    values = list(data.values())
    ax.bar(names, values)
    ax.set_ylabel("Assays avoided")
    ax.set_title("Assay cost proxy")
    return fig
