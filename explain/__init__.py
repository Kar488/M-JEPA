"""Explanation utilities for M-JEPA."""

from .integrated_gradients import (
    aggregate_undirected_edge_scores,
    build_zero_baseline_graph,
    compute_integrated_gradients,
    describe_atom_types,
    describe_bond_types,
    normalise_attributions,
    render_molecule_heatmap,
)

__all__ = [
    "aggregate_undirected_edge_scores",
    "build_zero_baseline_graph",
    "compute_integrated_gradients",
    "describe_atom_types",
    "describe_bond_types",
    "normalise_attributions",
    "render_molecule_heatmap",
]
