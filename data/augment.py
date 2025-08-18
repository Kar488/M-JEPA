from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as MT

from data.mdataset import GraphData

__all__ = [
    "random_rotation",
    "mask_random_angle",
    "perturb_dihedral",
    "delete_random_bond",
    "mask_random_atom",
    "remove_random_subgraph",
    "mask_subgraph",
    "generate_views",
    "apply_graph_augmentations",
]


def random_rotation(mol: Chem.Mol, conf_id: int = 0) -> Chem.Mol:
    """Spin the molecule around like a toy top.

    Apply a random 3D rotation to an RDKit molecule's conformer by
    multiplying atomic coordinates with a random orthogonal matrix.
    """
    if mol.GetNumConformers() == 0:
        return mol
    conf = mol.GetConformer(conf_id)
    rot = np.random.randn(3, 3)
    q, _ = np.linalg.qr(rot)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = np.array([p.x, p.y, p.z])
        v = q @ v
        conf.SetAtomPosition(i, v)
    return mol


def mask_random_angle(mol: Chem.Mol, conf_id: int = 0) -> Chem.Mol:
    """Freeze one bond so the molecule can't bend there.

    Mask a randomly chosen bond angle by setting it to zero degrees in
    the specified conformer, effectively removing local flexibility.
    """
    if mol.GetNumConformers() == 0:
        return mol
    atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() >= 2]
    if not atoms:
        return mol
    j = int(np.random.choice(atoms))
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()]
    if len(nbrs) < 2:
        return mol
    i, k = np.random.choice(nbrs, size=2, replace=False)
    try:
        MT.SetAngleDeg(mol.GetConformer(conf_id), int(i), int(j), int(k), 0.0)
    except Exception:
        pass
    return mol


def perturb_dihedral(
    mol: Chem.Mol, conf_id: int = 0, max_deg: float = 20.0
) -> Chem.Mol:
    """Give the molecule a tiny twist at one bond.

    Randomly perturb a dihedral angle by up to ``max_deg`` degrees in
    the selected conformer to add small structural noise.
    """
    if mol.GetNumConformers() == 0:
        return mol
    quadruples = []
    for b in mol.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        js = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        ks = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        for i in js:
            for l in ks:
                quadruples.append((i, j, k, l))
    if not quadruples:
        return mol
    i, j, k, l = quadruples[np.random.randint(len(quadruples))]
    conf = mol.GetConformer(conf_id)
    try:
        current = MT.GetDihedralDeg(conf, int(i), int(j), int(k), int(l))
        delta = float(np.random.uniform(-max_deg, max_deg))
        MT.SetDihedralDeg(conf, int(i), int(j), int(k), int(l), current + delta)
    except Exception:
        pass
    return mol


def _pick_neighbor(mol: Chem.Mol, center: int, exclude: int) -> Optional[int]:
    """Choose a nearby atom that's not off-limits.

    Return the smallest index neighbour of ``center`` that is not
    ``exclude`` or ``None`` if no such atom exists.
    """

    ns = sorted(
        [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(center).GetNeighbors()
            if nbr.GetIdx() != exclude
        ]
    )
    return ns[0] if ns else None


def _geom_features_for_bond(
    mol: Chem.Mol, i: int, j: int, conf_id: int = 0
) -> np.ndarray:
    """Describe a bond's shape in numbers.

    Compute bond length, neighbouring angles and dihedral information for
    the directed bond ``i -> j`` and return a 10‑dimensional feature
    vector encoded as ``numpy.ndarray``.
    """
    d = np.zeros(10, dtype=np.float32)
    if mol.GetNumConformers() == 0:
        return d
    conf = mol.GetConformer(conf_id)
    try:
        d[0] = float(MT.GetBondLength(conf, int(i), int(j)))
    except Exception:
        d[0] = 0.0
    k = _pick_neighbor(mol, i, j)
    l = _pick_neighbor(mol, j, i)
    if k is not None:
        try:
            ang = float(MT.GetAngleRad(conf, int(k), int(i), int(j)))
            d[1], d[2], d[3] = np.cos(ang), np.sin(ang), 1.0
        except Exception:
            pass
    if l is not None:
        try:
            ang = float(MT.GetAngleRad(conf, int(i), int(j), int(l)))
            d[4], d[5], d[6] = np.cos(ang), np.sin(ang), 1.0
        except Exception:
            pass
    if (k is not None) and (l is not None):
        try:
            dih = float(MT.GetDihedralRad(conf, int(k), int(i), int(j), int(l)))
            d[7], d[8], d[9] = np.cos(dih), np.sin(dih), 1.0
        except Exception:
            pass
    return d


def delete_random_bond(g: GraphData) -> GraphData:
    """Remove a random bond (and its reverse) from the graph."""
    if g.edge_index.shape[1] == 0:
        return g
    edges = g.edge_index.T
    pairs: dict[tuple[int, int], list[int]] = {}
    for idx, (u, v) in enumerate(edges):
        key = (int(min(u, v)), int(max(u, v)))
        pairs.setdefault(key, []).append(idx)
    key = list(pairs.keys())[np.random.randint(len(pairs))]
    mask = np.ones(edges.shape[0], dtype=bool)
    mask[pairs[key]] = False
    g.edge_index = edges[mask].T
    if g.edge_attr is not None:
        g.edge_attr = g.edge_attr[mask]
    return g


def mask_random_atom(g: GraphData) -> GraphData:
    """Zero out features of a random atom and its incident edges."""
    n = g.num_nodes()
    if n == 0:
        return g
    idx = int(np.random.randint(n))
    g.x[idx] = 0
    if g.edge_attr is not None and g.edge_attr.shape[0] == g.edge_index.shape[1]:
        mask = (g.edge_index[0] == idx) | (g.edge_index[1] == idx)
        g.edge_attr[mask] = 0
    return g


def remove_random_subgraph(g: GraphData) -> GraphData:
    """Remove a small connected subgraph starting from a random atom."""
    n = g.num_nodes()
    if n <= 1:
        return g
    adj = [[] for _ in range(n)]
    for u, v in g.edge_index.T:
        adj[int(u)].append(int(v))
    start = int(np.random.randint(n))
    max_size = int(np.random.randint(1, min(4, n)))
    to_remove = {start}
    frontier = [start]
    while frontier and len(to_remove) < max_size:
        cur = frontier.pop()
        nbrs = adj[cur]
        if not nbrs:
            continue
        np.random.shuffle(nbrs)
        for nb in nbrs:
            if len(to_remove) >= max_size:
                break
            if nb not in to_remove:
                to_remove.add(nb)
                frontier.append(nb)
    keep = [i for i in range(n) if i not in to_remove]
    if len(keep) == n or not keep:
        return g
    mapping = {old: new for new, old in enumerate(keep)}
    g.x = g.x[keep]
    if g.edge_index.shape[1] > 0:
        edges = g.edge_index.T
        edge_mask = np.array(
            [(u not in to_remove) and (v not in to_remove) for u, v in edges],
            dtype=bool,
        )
        edges = edges[edge_mask]
        if edges.size == 0:
            g.edge_index = np.zeros((2, 0), dtype=np.int64)
            if g.edge_attr is not None:
                g.edge_attr = g.edge_attr[0:0]
        else:
            remapped = np.array(
                [[mapping[int(u)], mapping[int(v)]] for u, v in edges],
                dtype=np.int64,
            )
            g.edge_index = remapped.T
            if g.edge_attr is not None:
                g.edge_attr = g.edge_attr[edge_mask]
    return g


def _subgraph(g: GraphData, idx: List[int]) -> GraphData:
    """Return the induced subgraph on the given node indices."""
    if len(idx) == 0 or g.x.shape[0] == 0:
        x = np.zeros((0, g.x.shape[1]), dtype=np.float32)
        e = np.zeros((2, 0), dtype=np.int64)
        ea = (
            None
            if g.edge_attr is None
            else np.zeros((0, g.edge_attr.shape[1]), dtype=np.float32)
        )
        return GraphData(x=x, edge_index=e, edge_attr=ea)
    remap = {old: new for new, old in enumerate(idx)}
    mask = np.isin(g.edge_index[0], idx) & np.isin(g.edge_index[1], idx)
    e = g.edge_index[:, mask].copy()
    for t in range(e.shape[1]):
        e[0, t] = remap[int(e[0, t])]
        e[1, t] = remap[int(e[1, t])]
    x = g.x[idx]
    ea = g.edge_attr[mask] if g.edge_attr is not None else None
    return GraphData(x=x, edge_index=e, edge_attr=ea)


def mask_subgraph(
    g: GraphData, mask_ratio: float, contiguous: bool
) -> tuple[GraphData, GraphData]:
    """Split ``g`` into context and target subgraphs."""
    n = int(g.x.shape[0])
    if n == 0:
        return g, g
    k = max(1, int(np.ceil(mask_ratio * n)))
    if contiguous:
        start = np.random.randint(0, n)
        tgt = [(start + j) % n for j in range(k)]
    else:
        tgt = np.random.choice(n, size=k, replace=False).tolist()
    ctx = [i for i in range(n) if i not in set(tgt)]
    return _subgraph(g, ctx), _subgraph(g, tgt)


def apply_graph_augmentations(
    g: GraphData,
    *,
    rotate: bool = False,
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
):
    """Play with the molecule's shape using optional tricks.

    Apply random rotation, angle masking, or dihedral perturbation to a
    :class:`GraphData` instance and update coordinates and geometric edge
    attributes accordingly.
    """
    if g.x.shape[1] < 7:
        return g
    num_nodes = g.x.shape[0]
    coords = g.x[:, -3:].copy()
    mol = Chem.RWMol()
    for i in range(num_nodes):
        z = int(g.x[i, 0])
        mol.AddAtom(Chem.Atom(z))
    added = set()
    for u, v in g.edge_index.T:
        tup = tuple(sorted((int(u), int(v))))
        if tup in added:
            continue
        mol.AddBond(tup[0], tup[1], Chem.BondType.SINGLE)
        added.add(tup)
    conf = Chem.Conformer(num_nodes)
    for i in range(num_nodes):
        conf.SetAtomPosition(i, tuple(map(float, coords[i])))
    mol.AddConformer(conf, assignId=True)
    if rotate:
        random_rotation(mol)
    if mask_angle:
        mask_random_angle(mol)
    if perturb_dihedral:
        perturb_dihedral(mol)
    conf = mol.GetConformer()
    for i in range(num_nodes):
        p = conf.GetAtomPosition(i)
        g.x[i, -3:] = [p.x, p.y, p.z]
    if g.edge_attr is not None and g.edge_attr.shape[1] >= 10:
        geom = [_geom_features_for_bond(mol, int(i), int(j)) for i, j in g.edge_index.T]
        geom = np.stack(geom, axis=0)
        g.edge_attr[:, -10:] = geom
    return g


def _clone_graph(g: GraphData) -> GraphData:
    return GraphData(
        x=g.x.copy(),
        edge_index=g.edge_index.copy(),
        edge_attr=None if g.edge_attr is None else g.edge_attr.copy(),
    )


def generate_views(
    graph: GraphData,
    structural_ops: Sequence[Callable[[GraphData], GraphData | tuple[GraphData, ...]]] = (),
    geometric_ops: Sequence[Callable[[GraphData], GraphData]] = (),
) -> List[GraphData]:
    """Apply structural and geometric operations to produce graph views."""

    views: List[GraphData] = [_clone_graph(graph)]
    for op in structural_ops:
        new_views: List[GraphData] = []
        for v in views:
            out = op(_clone_graph(v))
            if isinstance(out, (list, tuple)):
                new_views.extend(out)
            else:
                new_views.append(out)
        views = new_views

    final: List[GraphData] = []
    for v in views:
        v = _clone_graph(v)
        for op in geometric_ops:
            v = op(v)
        final.append(v)
    return final
