"""Representation analysis utilities."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import umap
except Exception:  # pragma: no cover - fallback if umap missing
    umap = None

from sklearn.manifold import TSNE


def compute_embedding_2d(
    embeddings: np.ndarray, *, random_state: int = 0
) -> np.ndarray:
    if umap is not None:
        reducer = umap.UMAP(random_state=random_state)
        return reducer.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=random_state, init="pca")
    return tsne.fit_transform(embeddings)


def plot_embedding(coords: np.ndarray, labels: Sequence, *, title: str, cmap="tab10"):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="label")
    ax.add_artist(legend)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    return fig


def build_embedding_table(
    coords: np.ndarray, metadata: Mapping[str, Sequence]
) -> pd.DataFrame:
    data = {"x": coords[:, 0], "y": coords[:, 1]}
    for key, values in metadata.items():
        data[key] = values
    return pd.DataFrame(data)
