"""Integrated Gradients utilities for graph explanations."""
from __future__ import annotations

import base64
import logging
import os
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from data.mdataset import GraphData

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except Exception as e:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore[assignment]
    print("[RDKit Chem import failed]", repr(e))

try:  # pragma: no cover - optional dependency
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception as e:  # pragma: no cover - optional dependency
    Draw = None  # type: ignore[assignment]
    rdMolDraw2D = None  # type: ignore[assignment]
    print("[RDKit Draw import failed]", repr(e))


_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8AARgMByFW9ggAAAABJRU5ErkJggg=="
)


def _as_numpy(value) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot convert None to numpy array")
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _zeros_like(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = _as_numpy(value)
    return np.zeros_like(arr)


def build_zero_baseline_graph(graph: GraphData) -> GraphData:
    """Return a baseline graph with the same structure and zeroed features."""

    baseline = GraphData(
        x=_zeros_like(graph.x),
        edge_index=_as_numpy(getattr(graph, "edge_index", np.zeros((2, 0), dtype=np.int64))).astype(
            np.int64, copy=False
        ),
        edge_attr=_zeros_like(getattr(graph, "edge_attr", None)),
        pos=getattr(graph, "pos", None),
    )
    return baseline


def compute_integrated_gradients(
    model: Callable[[SimpleNamespace], torch.Tensor],
    graph: GraphData,
    baseline_graph: Optional[GraphData] = None,
    *,
    m_steps: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Integrated Gradients for ``graph`` using ``model``."""

    if m_steps <= 0:
        raise ValueError("m_steps must be positive")

    device_t = torch.device(device or "cpu")
    baseline = baseline_graph or build_zero_baseline_graph(graph)

    x_input = torch.as_tensor(graph.x, dtype=torch.float32, device=device_t)
    x_base = torch.as_tensor(baseline.x, dtype=torch.float32, device=device_t)
    if x_input.shape != x_base.shape:
        raise ValueError("Baseline node features must match input shape")

    pos_tensor = None
    if getattr(graph, "pos", None) is not None:
        pos_tensor = torch.as_tensor(graph.pos, dtype=torch.float32, device=device_t)

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device_t)

    edge_attr = getattr(graph, "edge_attr", None)
    base_edge_attr = getattr(baseline, "edge_attr", None)
    edge_tensor = None
    base_edge_tensor = None
    diff_edge = None
    if edge_attr is not None or base_edge_attr is not None:
        input_attr = edge_attr
        base_attr = base_edge_attr
        if input_attr is None and base_attr is not None:
            input_attr = np.zeros_like(base_attr)
        if base_attr is None and input_attr is not None:
            base_attr = np.zeros_like(input_attr)
        if input_attr is None or base_attr is None:
            width = edge_index_tensor.shape[1]
            edge_tensor = torch.zeros((width, 0), dtype=torch.float32, device=device_t)
            base_edge_tensor = edge_tensor.clone()
        else:
            edge_tensor = torch.as_tensor(input_attr, dtype=torch.float32, device=device_t)
            base_edge_tensor = torch.as_tensor(base_attr, dtype=torch.float32, device=device_t)
        if edge_tensor.shape != base_edge_tensor.shape:
            raise ValueError("Baseline edge features must match input shape")
        diff_edge = edge_tensor - base_edge_tensor

    diff_x = x_input - x_base
    ptr = torch.tensor([0, x_input.size(0)], dtype=torch.long, device=device_t)

    grad_acc_x = torch.zeros_like(x_input)
    grad_acc_edge = torch.zeros_like(diff_edge) if diff_edge is not None else None

    for step in range(1, m_steps + 1):
        alpha = float(step) / float(m_steps)
        x_step = x_base + alpha * diff_x
        x_step.requires_grad_(True)
        if diff_edge is not None and base_edge_tensor is not None:
            edge_step = base_edge_tensor + alpha * diff_edge
            edge_step.requires_grad_(True)
        else:
            edge_step = None

        graph_ns = SimpleNamespace(
            x=x_step,
            edge_index=edge_index_tensor,
            graph_ptr=ptr,
        )
        if edge_step is not None:
            graph_ns.edge_attr = edge_step
        if pos_tensor is not None:
            graph_ns.pos = pos_tensor

        with torch.enable_grad():
            output = model(graph_ns)
            if not torch.is_tensor(output):
                output = torch.as_tensor(output, dtype=torch.float32, device=device_t)
            scalar = output.reshape(-1)[0]
            grads = torch.autograd.grad(
                scalar,
                [tensor for tensor in (x_step, edge_step) if tensor is not None],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grad_acc_x = grad_acc_x + grads[0]
        if edge_step is not None and grad_acc_edge is not None:
            edge_grad = grads[1] if len(grads) > 1 else None
            if edge_grad is not None:
                grad_acc_edge = grad_acc_edge + edge_grad

    scale = 1.0 / float(m_steps)
    node_attr = diff_x * grad_acc_x * scale
    node_scores = node_attr.sum(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)

    if diff_edge is None or grad_acc_edge is None:
        edge_scores = np.zeros(edge_index_tensor.shape[1], dtype=np.float32)
    else:
        edge_attr_contrib = diff_edge * grad_acc_edge * scale
        edge_scores = edge_attr_contrib.sum(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)

    return node_scores, edge_scores


def normalise_attributions(values: Sequence[float], mode: str = "signed") -> np.ndarray:
    """Normalise attribution scores for visualisation."""

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if mode == "none":
        return arr

    abs_vals = np.abs(arr)
    max_val = float(abs_vals.max())
    if max_val <= 0:
        return np.zeros_like(arr)

    scaled = arr / max_val
    if mode == "absolute":
        return np.abs(scaled)
    return scaled


def aggregate_undirected_edge_scores(edge_index, edge_scores: Sequence[float]) -> Dict[Tuple[int, int], float]:
    """Collapse directed edge scores into undirected bond scores."""

    if edge_index is None:
        return {}
    idx = np.asarray(edge_index)
    if idx.ndim != 2 or idx.shape[0] != 2:
        return {}
    scores = {}
    for col, score in zip(idx.T, edge_scores):
        src, dst = int(col[0]), int(col[1])
        key = tuple(sorted((src, dst)))
        scores[key] = scores.get(key, 0.0) + float(score)
    return scores


def describe_atom_types(smiles: Optional[str], num_atoms: int) -> Sequence[str]:
    """Return atom labels derived from SMILES when available."""

    if Chem is None or not smiles:
        return [f"atom_{idx}" for idx in range(num_atoms)]
    try:  # pragma: no cover - depends on rdkit
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None or mol.GetNumAtoms() != num_atoms:
        return [f"atom_{idx}" for idx in range(num_atoms)]
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def describe_bond_types(smiles: Optional[str], pairs: Iterable[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    """Return textual descriptions for bond pairs when RDKit is available."""

    descriptions: Dict[Tuple[int, int], str] = {}
    if Chem is None or not smiles:
        for pair in pairs:
            descriptions[pair] = f"bond_{pair[0]}_{pair[1]}"
        return descriptions
    try:  # pragma: no cover - depends on rdkit
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        for pair in pairs:
            descriptions[pair] = f"bond_{pair[0]}_{pair[1]}"
        return descriptions
    try:  # pragma: no cover - depends on rdkit
        num_atoms = int(mol.GetNumAtoms())
    except Exception:
        num_atoms = None

    for pair in pairs:
        i, j = pair
        if num_atoms is not None and (i < 0 or j < 0 or i >= num_atoms or j >= num_atoms):
            descriptions[pair] = f"bond_{i}_{j}"
            continue
        try:  # pragma: no cover - depends on rdkit
            bond = mol.GetBondBetweenAtoms(i, j)
        except Exception:
            bond = None
        if bond is None:
            descriptions[pair] = f"bond_{i}_{j}"
            continue
        try:  # pragma: no cover - depends on rdkit
            atom_i = mol.GetAtomWithIdx(i).GetSymbol()
            atom_j = mol.GetAtomWithIdx(j).GetSymbol()
            bond_type = bond.GetBondType()
        except Exception:
            descriptions[pair] = f"bond_{i}_{j}"
            continue
        descriptions[pair] = f"{atom_i}-{atom_j}:{bond_type}"
    return descriptions


def _score_to_color(score: float, positive: bool) -> Tuple[float, float, float]:
    intensity = min(1.0, max(0.0, abs(score)))
    base = 0.2
    if positive:
        return (base + 0.8 * intensity, base, base)
    return (base, base, base + 0.8 * intensity)


def render_molecule_heatmap(
    smiles: Optional[str],
    atom_scores: Sequence[float],
    bond_scores: Mapping[Tuple[int, int], float],
    output_path: str,
) -> str:
    """Render a RDKit heatmap or fallback placeholder for the molecule."""

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if Chem is None or Draw is None or rdMolDraw2D is None or not smiles:
        with open(output_path, "wb") as handle:
            handle.write(_PLACEHOLDER_PNG)
        return output_path

    try:  # pragma: no cover - relies on rdkit availability
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        with open(output_path, "wb") as handle:
            handle.write(_PLACEHOLDER_PNG)
        return output_path

    try:  # pragma: no cover - depends on rdkit
        max_atoms = int(mol.GetNumAtoms())
    except Exception:
        max_atoms = len(atom_scores)
    highlight_atoms = {idx: _score_to_color(score, score >= 0) for idx, score in enumerate(atom_scores[:max_atoms])}
    bond_colors = {}
    for (i, j), score in bond_scores.items():
        if i < 0 or j < 0 or i >= max_atoms or j >= max_atoms:
            continue
        try:  # pragma: no cover - depends on rdkit
            bond = mol.GetBondBetweenAtoms(i, j)
        except Exception:
            bond = None
        if bond is None:
            continue
        bond_colors[bond.GetIdx()] = _score_to_color(score, score >= 0)

    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(highlight_atoms.keys()),
        highlightAtomColors=highlight_atoms,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    with open(output_path, "wb") as handle:
        handle.write(drawer.GetDrawingText())
    return output_path
