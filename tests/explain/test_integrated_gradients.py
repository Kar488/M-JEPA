import numpy as np
import pytest

from data.mdataset import GraphData
from explain.integrated_gradients import (
    build_zero_baseline_graph,
    compute_integrated_gradients,
    describe_bond_types,
    render_molecule_heatmap,
)

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


def test_describe_bond_types_handles_out_of_range(monkeypatch):
    calls = []

    class _FakeBond:
        def GetBondType(self):
            return "SINGLE"

    class _FakeAtom:
        def __init__(self, symbol):
            self._symbol = symbol

        def GetSymbol(self):
            return self._symbol

    class _FakeMol:
        def GetNumAtoms(self):
            return 2

        def GetBondBetweenAtoms(self, i, j):
            if i >= self.GetNumAtoms() or j >= self.GetNumAtoms():
                raise AssertionError("out of range bond lookup")
            calls.append((i, j))
            return _FakeBond()

        def GetAtomWithIdx(self, idx):
            return _FakeAtom(["C", "O"][idx])

    monkeypatch.setattr(
        "explain.integrated_gradients.Chem", type("ChemNS", (), {"MolFromSmiles": lambda smiles: _FakeMol()})
    )

    pairs = [(0, 1), (2, 0)]
    descriptions = describe_bond_types("CO", pairs)

    assert descriptions[(0, 1)] == "C-O:SINGLE"
    assert descriptions[(2, 0)] == "bond_2_0"
    assert calls == [(0, 1)]


def test_render_molecule_heatmap_skips_invalid_bonds(monkeypatch, tmp_path):
    calls = []

    class _FakeBond:
        def __init__(self, idx):
            self._idx = idx

        def GetIdx(self):
            return self._idx

    class _FakeMol:
        def GetNumAtoms(self):
            return 2

        def GetBondBetweenAtoms(self, i, j):
            if i >= self.GetNumAtoms() or j >= self.GetNumAtoms():
                raise AssertionError("out of range bond lookup")
            calls.append((i, j))
            return _FakeBond(idx=len(calls))

    class _FakeDrawer:
        def __init__(self, *_):
            self._bytes = b"image"

        def FinishDrawing(self):
            return None

        def GetDrawingText(self):
            return self._bytes

    def _prepare(drawer, mol, **_kwargs):
        drawer._bytes += b"_prepared"

    monkeypatch.setattr(
        "explain.integrated_gradients.Chem", type("ChemNS", (), {"MolFromSmiles": lambda smiles: _FakeMol()})
    )
    monkeypatch.setattr(
        "explain.integrated_gradients.rdMolDraw2D",
        type(
            "DrawNS",
            (),
            {
                "MolDraw2DCairo": _FakeDrawer,
                "PrepareAndDrawMolecule": staticmethod(_prepare),
            },
        ),
    )
    monkeypatch.setattr("explain.integrated_gradients.Draw", object())

    output = tmp_path / "heatmap.png"
    render_molecule_heatmap(
        "CO",
        [0.1, -0.2],
        {(0, 1): 0.5, (3, 0): -0.3},
        str(output),
    )

    assert output.is_file()
    assert calls == [(0, 1)]
