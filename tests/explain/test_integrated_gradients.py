import numpy as np
import pytest

from data.mdataset import GraphData
from explain.integrated_gradients import compute_integrated_gradients, build_zero_baseline_graph

torch = pytest.importorskip("torch")


def _linear_model():
    def model(graph):
        return graph.x.sum()

    return model


def test_integrated_gradients_completeness():
    graph = GraphData(
        x=np.array([[1.0], [2.0]], dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    baseline = build_zero_baseline_graph(graph)
    model = _linear_model()

    node_scores, edge_scores = compute_integrated_gradients(
        model,
        graph,
        baseline_graph=baseline,
        m_steps=10,
        device=torch.device("cpu"),
    )

    assert edge_scores.shape == (0,)
    assert pytest.approx(node_scores.sum(), rel=1e-5) == 3.0
    assert np.allclose(node_scores, np.array([1.0, 2.0], dtype=np.float32), atol=1e-4)


def test_integrated_gradients_zero_when_matching_baseline():
    graph = GraphData(
        x=np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    baseline = GraphData(
        x=np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    model = _linear_model()
    node_scores, edge_scores = compute_integrated_gradients(
        model,
        graph,
        baseline_graph=baseline,
        m_steps=5,
        device=torch.device("cpu"),
    )

    assert np.allclose(node_scores, np.zeros_like(node_scores))
    assert edge_scores.size == 0
