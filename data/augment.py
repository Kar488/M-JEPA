from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, TYPE_CHECKING

try:  # optional heavy deps
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # rdkit is optional outside tests
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdMolTransforms as MT  # type: ignore
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    MT = None  # type: ignore

try:  # GraphData is optional for lightweight imports
    from data.mdataset import GraphData  # type: ignore
except Exception:  # pragma: no cover
    GraphData = Any  # type: ignore

__all__ = [
    "AugmentationConfig",
    "iter_augmentation_options",
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


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration flags for optional geometric augmentations."""

    # geometric
    random_rotate: bool = False
    mask_angle: bool = False
    perturb_dihedral: bool = False
    # graph-level (NEW)
    bond_deletion: bool = False
    atom_masking: bool = False
    subgraph_removal: bool = False

    def __init__(
        self,
        random_rotate: bool = False,
        mask_angle: bool = False,
        perturb_dihedral: bool = False,
        bond_deletion: bool = False,
        atom_masking: bool = False,
        subgraph_removal: bool = False,

        *,
        rotate: Optional[bool] = None,
        dihedral: Optional[bool] = None,
    ) -> None:
        object.__setattr__(self, "random_rotate", random_rotate or bool(rotate))
        object.__setattr__(self, "mask_angle", bool(mask_angle))
        object.__setattr__(self, "perturb_dihedral", perturb_dihedral or bool(dihedral))
        object.__setattr__(self, "bond_deletion", bool(bond_deletion))
        object.__setattr__(self, "atom_masking", bool(atom_masking))
        object.__setattr__(self, "subgraph_removal", bool(subgraph_removal))

    @property
    def rotate(self) -> bool:
        return self.random_rotate

    @property
    def dihedral(self) -> bool:
        return self.perturb_dihedral

    @classmethod
    def from_dict(cls, cfg: Optional[dict] = None) -> "AugmentationConfig":
        cfg = cfg or {}
        return cls(
            random_rotate=bool(cfg.get("random_rotate", cfg.get("rotate", False))),
            mask_angle=bool(cfg.get("mask_angle", False)),
            perturb_dihedral=bool(cfg.get("perturb_dihedral", cfg.get("dihedral", False))),
            bond_deletion=bool(cfg.get("bond_deletion", False)),
            atom_masking=bool(cfg.get("atom_masking", False)),
            subgraph_removal=bool(cfg.get("subgraph_removal", False)),
        )


def iter_augmentation_options(
    rotate_opts: Optional[Iterable[bool]] = None,
    mask_angle_opts: Optional[Iterable[bool]] = None,
    dihedral_opts: Optional[Iterable[bool]] = None,
) -> Iterator[AugmentationConfig]:
    """Yield ``AugmentationConfig`` for all flag combinations.

    Parameters accept any iterable of truthy values and default to
    ``(False, True)`` when omitted.  Passing ``[True]`` for a flag forces that
    augmentation to be enabled, mirroring the previous manual loops.
    """

    r_opts = [bool(v) for v in (rotate_opts or (False, True))]
    m_opts = [bool(v) for v in (mask_angle_opts or (False, True))]
    d_opts = [bool(v) for v in (dihedral_opts or (False, True))]

    for r, m, d in product(r_opts, m_opts, d_opts):
        yield AugmentationConfig(random_rotate=r, mask_angle=m, perturb_dihedral=d)


def random_rotation(mol: Chem.Mol, conf_id: int = 0) -> Chem.Mol:
    """Spin the molecule around like a toy top.

    Apply a random 3D rotation to an RDKit molecule's conformer by
    multiplying atomic coordinates with a random orthogonal matrix.
    """
    if Chem is None or np is None or mol.GetNumConformers() == 0:
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
    if Chem is None or np is None or mol.GetNumConformers() == 0:
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
       try:
        Chem.GetSymmSSSR(mol); mol.UpdatePropertyCache(False)
        MT.SetAngleDeg(mol.GetConformer(conf_id), int(i), int(j), int(k), 0.0)
       except Exception:
        pass
    return mol


# expects: from rdkit import Chem
#          from rdkit.Chem import rdMolTransforms as MT
#          import numpy as np

def perturb_dihedral(mol: Chem.Mol, conf_id: int = 0, max_deg: float = 20.0) -> Chem.Mol:
    """Randomly perturb a dihedral by up to `max_deg` degrees.
    Safe if RingInfo wasn't initialized (handles it and retries once)."""
    if mol is None or mol.GetNumConformers() == 0:
        return mol

    # Ensure ring info is initialized before MolTransforms calls
    try:
        Chem.GetSymmSSSR(mol)
        mol.UpdatePropertyCache(False)
        _ = mol.GetRingInfo()
    except Exception:
        pass

    # Build candidate (i, j, k, l) quadruples from bonds
    quadruples = []
    for b in mol.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        js = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        ks = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        for i in js:
            for ell in ks:
                quadruples.append((i, j, k, ell))
    if not quadruples:
        return mol

    i, j, k, ell = quadruples[np.random.randint(len(quadruples))]
    conf = mol.GetConformer(conf_id)

    def _try_set():
        cur = MT.GetDihedralDeg(conf, int(i), int(j), int(k), int(ell))
        delta = float(np.random.uniform(-max_deg, max_deg))
        MT.SetDihedralDeg(conf, int(i), int(j), int(k), int(ell), cur + delta)

    try:
        _try_set()
    except Exception:
        try:
            Chem.GetSymmSSSR(mol); mol.UpdatePropertyCache(False); mol.GetRingInfo()
            _try_set()
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
    ell = _pick_neighbor(mol, j, i)
    if k is not None:
        try:
            ang = float(MT.GetAngleRad(conf, int(k), int(i), int(j)))
            d[1], d[2], d[3] = np.cos(ang), np.sin(ang), 1.0
        except Exception:
            pass
    if ell is not None:
        try:
            ang = float(MT.GetAngleRad(conf, int(i), int(j), int(ell)))
            d[4], d[5], d[6] = np.cos(ang), np.sin(ang), 1.0
        except Exception:
            pass
    if (k is not None) and (ell is not None):
        try:
            dih = float(MT.GetDihedralRad(conf, int(k), int(i), int(j), int(ell)))
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
    g: GraphData, # type: ignore
    *,
    random_rotate: bool = False,
    rotate: bool = False,
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
    dihedral: bool = False,
    bond_deletion: bool = False,
    atom_masking: bool = False,
    subgraph_removal: bool = False,
):
    """Play with the molecule's shape using optional tricks.

    Apply random rotation, angle masking, or dihedral perturbation to a
    :class:`GraphData` instance and update coordinates and geometric edge
    attributes accordingly.
    """
    # --- Geometric transforms (coords) ---
    have_geom = (Chem is not None) and (np is not None) and (g.x.shape[1] >= 7)
    if have_geom:
        num_nodes = g.x.shape[0]
        coords = g.x[:, -3:].copy()

        # Build a temporary RDKit molecule from GraphData
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

        # initialize RingInfo once (prevents RDKit invariant crash)
        try:
            Chem.GetSymmSSSR(mol)         # compute ring info without full sanitize
            mol.UpdatePropertyCache(False) # avoid strict valence checks
            _ = mol.GetRingInfo()          # touch to ensure initialized
        except Exception:
            pass

        # Normalise alias flags
        do_rotate = bool(random_rotate or rotate)
        do_dihedral = bool(perturb_dihedral or dihedral)

        # Geometric ops
        if do_rotate:
            random_rotation(mol)
        if mask_angle:
            try:
                mask_random_angle(mol)
            except Exception:
                # belt-and-braces retry
                try:
                    Chem.GetSymmSSSR(mol); mol.UpdatePropertyCache(False); mol.GetRingInfo()
                    mask_random_angle(mol)
                except Exception:
                    pass
        if do_dihedral:
            try:
                perturb_dihedral(mol)   # use your safe version if you added it
            except Exception:
                try:
                    Chem.GetSymmSSSR(mol); mol.UpdatePropertyCache(False); mol.GetRingInfo()
                    perturb_dihedral(mol)
                except Exception:
                    pass

        # Write back coords + last 10 geom features (if present)
        conf = mol.GetConformer()
        for i in range(num_nodes):
            p = conf.GetAtomPosition(i)
            g.x[i, -3:] = [p.x, p.y, p.z]
            if g.edge_attr is not None and g.edge_attr.shape[1] >= 10:
                # number of edges (E); handle E == 0 safely
                try:
                    E = int(getattr(g.edge_index, "shape", [None, 0])[1] or 0)
                except Exception:
                    E = 0

                if E == 0:
                    # nothing to compute; keep as-is or zero out slice if you prefer
                    # g.edge_attr[:, -10:] = 0.0
                    pass
                else:
                    ei = g.edge_index.T
                    if hasattr(ei, "tolist"):
                        ei = ei.tolist()
                    geom_list = [_geom_features_for_bond(mol, int(i), int(j)) for (i, j) in ei]
                    if geom_list:
                        geom = np.stack(geom_list, axis=0).astype(np.float32, copy=False)
                        # be defensive if edge_attr rows don’t match E
                        if g.edge_attr.shape[0] == geom.shape[0]:
                            g.edge_attr[:, -10:] = geom
                        else:
                            m = min(g.edge_attr.shape[0], geom.shape[0])
                            if m > 0:
                                g.edge_attr[:m, -10:] = geom[:m]

    # --- Graph-structure transforms (always available) ---
    def b(x):  # robust 0/1/True/False → bool
        return bool(int(x)) if isinstance(x, (int, str)) else bool(x)

    if b(bond_deletion):
        g = delete_random_bond(g)
    if b(atom_masking):
        g = mask_random_atom(g)
    if b(subgraph_removal):
        g = remove_random_subgraph(g)

    return g
    

        


def _clone_graph(g: GraphData) -> GraphData:
    return GraphData(
        x=g.x.copy(),
        edge_index=g.edge_index.copy(),
        edge_attr=None if g.edge_attr is None else g.edge_attr.copy(),
    )


def generate_views(
    graph: GraphData,
    structural_ops: Sequence[
        Callable[[GraphData], GraphData | tuple[GraphData, ...]] # type: ignore
    ] = (),
    geometric_ops: Sequence[Callable[[GraphData], GraphData]] = (),
) -> List[GraphData]: # type: ignore
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
