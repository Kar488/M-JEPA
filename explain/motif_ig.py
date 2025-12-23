"""Motif-level Integrated Gradients utilities."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from types import SimpleNamespace
from explain.integrated_gradients import render_molecule_heatmap

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.mdataset import GraphData
from explain.integrated_gradients import (
    aggregate_undirected_edge_scores,
    build_zero_baseline_graph,
    compute_integrated_gradients,
    normalise_attributions,
    render_molecule_heatmap,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from rdkit import Chem  # type: ignore
    _has_rdkit = True
except Exception as e:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore[assignment]
    _has_rdkit = False
    print("[RDKit import failed]", repr(e))


_DEF_MOTIF_NAME = "molecule"


def _ensure_motif_map(motif_map: Optional[Mapping[str, Iterable[int]]], num_atoms: int) -> Dict[str, List[int]]:
    if motif_map:
        cleaned: Dict[str, List[int]] = {}
        for key, atoms in motif_map.items():
            if atoms is None:
                continue
            indices = [int(idx) for idx in atoms if 0 <= int(idx) < num_atoms]
            if not indices:
                continue
            cleaned[str(key)] = sorted(set(indices))
        if cleaned:
            return cleaned
    return {_DEF_MOTIF_NAME: list(range(num_atoms))}


def _infer_num_atoms(motif_map: Mapping[str, Iterable[int]]) -> int:
    max_atom = -1
    for atoms in motif_map.values():
        if atoms is None:
            continue
        for atom in atoms:
            try:
                idx = int(atom)
            except (TypeError, ValueError):
                continue
            if idx > max_atom:
                max_atom = idx
    return max_atom + 1 if max_atom >= 0 else 0


def find_motifs(graph_or_smiles: GraphData | str) -> Dict[str, List[int]]:
    """Derive a mapping from motif names to atom indices.

    Falls back to a single "molecule" motif spanning all atoms when RDKit is
    unavailable or motifs cannot be determined.
    """

    smiles: Optional[str] = None
    num_atoms: Optional[int] = None

    if isinstance(graph_or_smiles, GraphData):
        num_atoms = graph_or_smiles.num_nodes()
        smiles = getattr(graph_or_smiles, "smiles", None)
    elif isinstance(graph_or_smiles, str):
        smiles = graph_or_smiles

    if not _has_rdkit or not smiles:
        if num_atoms is None:
            num_atoms = 0
        return _ensure_motif_map(None, num_atoms)

    try:  # pragma: no cover - depends on rdkit
        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # pragma: no cover - defensive
        mol = None

    if mol is None:
        if num_atoms is None:
            num_atoms = 0
        return _ensure_motif_map(None, num_atoms)

    try:  # pragma: no cover - depends on rdkit
        num_atoms = int(mol.GetNumAtoms())
    except Exception:
        num_atoms = num_atoms or 0

    motifs: Dict[str, List[int]] = {}

    try:  # pragma: no cover - depends on rdkit
        fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        for idx, atoms in enumerate(fragments):
            motifs[f"fragment_{idx}"] = sorted(set(int(a) for a in atoms))
    except Exception:
        pass

    ring_atoms: List[tuple[int, ...]] = []
    try:  # pragma: no cover - depends on rdkit
        ring_info = mol.GetRingInfo()
        ring_atoms = list(ring_info.AtomRings())
    except Exception:
        ring_atoms = []
    for idx, atoms in enumerate(ring_atoms):
        motifs[f"ring_{idx}"] = sorted(set(int(a) for a in atoms))

    return _ensure_motif_map(motifs, num_atoms)


def aggregate_motif_ig(
    node_scores: Sequence[float],
    edge_scores: Sequence[float],
    motif_map: Mapping[str, Iterable[int]],
    edge_index: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregate node and edge IG scores into motif-level totals."""

    node_arr = np.asarray(node_scores, dtype=np.float32)
    edge_arr = np.asarray(edge_scores, dtype=np.float32)
    num_atoms = int(node_arr.shape[0])
    motifs = _ensure_motif_map(motif_map, num_atoms)

    edge_assignments: Dict[str, List[float]] = {name: [] for name in motifs}
    if edge_index is not None and edge_arr.size > 0:
        try:
            undirected = aggregate_undirected_edge_scores(edge_index, edge_arr)
            for (i, j), score in undirected.items():
                for name, atoms in motifs.items():
                    if i in atoms and j in atoms:
                        edge_assignments[name].append(float(score))
        except Exception:
            logger.debug("Failed to aggregate edge scores for motifs", exc_info=True)

    aggregated: Dict[str, float] = {}
    for name, atoms in motifs.items():
        atom_total = float(node_arr[atoms].sum()) if atoms else 0.0
        edge_total = float(sum(edge_assignments.get(name, [])))
        aggregated[name] = atom_total + edge_total
    return aggregated


def _mask_graph(graph: GraphData, atoms: Iterable[int]) -> GraphData:
    mask_set = set(int(a) for a in atoms)
    x = np.array(graph.x, copy=True)
    for idx in mask_set:
        if 0 <= idx < x.shape[0]:
            x[idx] = 0.0

    edge_attr = getattr(graph, "edge_attr", None)
    edge_index = getattr(graph, "edge_index", None)
    masked_edge_attr = None
    if edge_attr is not None:
        masked_edge_attr = np.array(edge_attr, copy=True)
        if edge_index is not None:
            for col_idx, (src, dst) in enumerate(np.asarray(edge_index).T):
                if int(src) in mask_set or int(dst) in mask_set:
                    masked_edge_attr[col_idx] = 0.0

    pos = getattr(graph, "pos", None)
    masked_pos = None
    if pos is not None:
        masked_pos = np.array(pos, copy=True)
        for idx in mask_set:
            if 0 <= idx < masked_pos.shape[0]:
                masked_pos[idx] = 0.0

    return GraphData(
        x=x,
        edge_index=np.array(edge_index, copy=True) if edge_index is not None else np.zeros((2, 0), dtype=np.int64),
        edge_attr=masked_edge_attr,
        pos=masked_pos,
    )


def _graph_to_namespace(graph: GraphData, device: Optional[torch.device]) -> SimpleNamespace:
    ptr = torch.tensor([0, graph.num_nodes()], dtype=torch.long, device=device)
    ns = SimpleNamespace(
        x=torch.as_tensor(graph.x, dtype=torch.float32, device=device),
        edge_index=torch.as_tensor(
            getattr(graph, "edge_index", np.zeros((2, 0), dtype=np.int64)), dtype=torch.long, device=device
        ),
        graph_ptr=ptr,
    )
    if getattr(graph, "edge_attr", None) is not None:
        ns.edge_attr = torch.as_tensor(graph.edge_attr, dtype=torch.float32, device=device)
    if getattr(graph, "pos", None) is not None:
        ns.pos = torch.as_tensor(graph.pos, dtype=torch.float32, device=device)
    return ns


def compute_motif_deltas(
    model_fn: Callable[[SimpleNamespace], torch.Tensor],
    graph: GraphData,
    motif_map: Mapping[str, Iterable[int]],
    *,
    device: Optional[torch.device] = None,
    baseline_logits: Optional[Sequence[float]] = None,
) -> Dict[str, List[float]]:
    """Compute per-motif logit deltas by masking motif atoms."""

    motifs = _ensure_motif_map(motif_map, graph.num_nodes())
    graph_ns = _graph_to_namespace(graph, device)
    if baseline_logits is None:
        with torch.no_grad():
            baseline_logits = model_fn(graph_ns)
    baseline_tensor = torch.as_tensor(baseline_logits, dtype=torch.float32)
    baseline_flat = baseline_tensor.reshape(-1)

    deltas: Dict[str, List[float]] = {}
    for name, atoms in motifs.items():
        masked = _mask_graph(graph, atoms)
        masked_ns = _graph_to_namespace(masked, device)
        with torch.no_grad():
            masked_logits = torch.as_tensor(model_fn(masked_ns), dtype=torch.float32)
        masked_flat = masked_logits.reshape(-1)
        max_len = max(int(baseline_flat.numel()), int(masked_flat.numel()))
        base_np = baseline_flat.detach().cpu().numpy().astype(np.float32, copy=False)
        masked_np = masked_flat.detach().cpu().numpy().astype(np.float32, copy=False)
        if base_np.size < max_len:
            base_np = np.pad(base_np, (0, max_len - base_np.size), constant_values=0.0)
        if masked_np.size < max_len:
            masked_np = np.pad(masked_np, (0, max_len - masked_np.size), constant_values=0.0)
        delta = base_np[:max_len] - masked_np[:max_len]
        deltas[name] = [float(val) for val in delta]
    return deltas


def draw_motif_heatmap(
    smiles: str,
    motif_scores: Mapping[str, float],
    motif_map: Mapping[str, List[int]],
    output_path: str,
    assay_type: str = "NR"
):
    """Maps motif scores back to atoms to generate the 'pretty' diagnostic heatmap."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return

    num_atoms = mol.GetNumAtoms()
    atom_weights = np.zeros(num_atoms)

    for motif_name, score in motif_scores.items():
        atom_indices = motif_map.get(motif_name, [])
        for idx in atom_indices:
            if idx < num_atoms:
                atom_weights[idx] += score

    # Delegate to the main renderer to get Gaussian patches and Indices
    render_molecule_heatmap(smiles, atom_weights, {}, output_path, assay_type=assay_type)

def plot_motif_deltas(
    task_names: Sequence[str],
    motif_delta_vector: Sequence[float],
    output_path: str,
) -> str:
    """Plot a per-task bar chart for motif deltas."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    values = np.asarray(motif_delta_vector, dtype=np.float32)
    colors = ["#d62728" if val >= 0 else "#1f77b4" for val in values]
    positions = np.arange(len(values))

    plt.figure(figsize=(max(4, len(values) * 0.5), 3))
    plt.bar(positions, values, color=colors)
    plt.xticks(positions, task_names, rotation=45, ha="right")
    plt.ylabel("logit delta")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_motif_artifacts(
    *,
    smiles: Optional[str],
    motif_map: Mapping[str, Iterable[int]],
    motif_scores: Mapping[str, float],
    motif_deltas: Mapping[str, Sequence[float]],
    task_names: Sequence[str],
    output_dir: str,
    normalise_mode: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    summary_path = os.path.join(output_dir, "motif_summary.csv")

    # FIX B: Clean and align motif map before using it
    num_atoms = _infer_num_atoms(motif_map) if motif_map else 0
    if num_atoms == 0 and smiles and _has_rdkit:
        try:  # pragma: no cover - depends on rdkit
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None
        if mol is not None:
            try:  # pragma: no cover - depends on rdkit
                num_atoms = int(mol.GetNumAtoms())
            except Exception:
                num_atoms = 0
    motifs_clean = _ensure_motif_map(motif_map, num_atoms)

    # Ensure motif_scores contains all motif keys
    motif_scores = dict(motif_scores)
    for name in motifs_clean:
        motif_scores.setdefault(name, 0.0)

    # Recompute normalization using aligned motif order
    norm_scores = normalise_attributions(
        [motif_scores[name] for name in motifs_clean],
        mode=normalise_mode
    )
    norm_lookup = {
        name: float(value)
        for name, value in zip(motifs_clean.keys(), norm_scores)
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("motif,ig_score,norm_score,size\n")
        for name, atoms in motifs_clean.items():
            score = float(motif_scores.get(name, 0.0))
            size = len(atoms)
            handle.write(f"{name},{score},{norm_lookup.get(name, 0.0)},{size}\n")
            summary_rows.append({
                "motif": name,
                "ig_score": score,
                "norm_score": norm_lookup.get(name, 0.0)
            })

    deltas_path = os.path.join(output_dir, "motif_deltas.json")
    with open(deltas_path, "w", encoding="utf-8") as handle:
        json.dump({name: list(delta) for name, delta in motif_deltas.items()}, handle, indent=2)

    heatmap_path = os.path.join(output_dir, "motif_heatmap.png")
    draw_motif_heatmap(smiles, norm_lookup, motifs_clean, heatmap_path)

    bar_paths: Dict[str, str] = {}
    for name, delta in motif_deltas.items():
        bar_paths[name] = plot_motif_deltas(task_names, delta, os.path.join(output_dir, f"{name}_deltas.png"))

    return {
        "summary_csv": summary_path,
        "deltas_json": deltas_path,
        "heatmap_png": heatmap_path,
        "bar_charts": bar_paths,
        "motif_rows": summary_rows,
    }


__all__ = [
    "find_motifs",
    "aggregate_motif_ig",
    "compute_motif_deltas",
    "draw_motif_heatmap",
    "plot_motif_deltas",
    "save_motif_artifacts",
]
