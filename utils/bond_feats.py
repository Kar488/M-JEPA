# utils/bond_feats.py  (new helper)
from rdkit import Chem
import numpy as np

_BOND_TYPES = [
    Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
]
_STEREO = [
    Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
]

def _bond_vector(b: Chem.Bond) -> np.ndarray:
    # One-hot bond type (4)
    bt = [int(b.GetBondType() == t) for t in _BOND_TYPES]
    # Flags: conjugated, in ring, aromatic (3)
    flags = [int(b.GetIsConjugated()), int(b.IsInRing()), int(b.GetIsAromatic())]
    # One-hot stereo (6)
    st = [int(b.GetStereo() == s) for s in _STEREO]
    return np.asarray(bt + flags + st, dtype=np.float32)  # total dim = 13

def _infer_has_3d(graph) -> float:
    """Return 1.0 if the graph appears to have 3-D coordinates, else 0.0."""

    pos = getattr(graph, "pos", None)
    if pos is None:
        return 0.0
    try:
        arr = np.asarray(pos)
    except Exception:
        return 0.0
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0.0
    return 1.0 if arr.shape[1] >= 3 else 0.0


def _finalise_edge_attr(
    feats: np.ndarray,
    graph,
    *,
    target_edge_dim: int | None = None,
    has_3d: float | None = None,
) -> np.ndarray:
    """Pad or trim ``feats`` to ``target_edge_dim`` while setting the 3-D flag."""

    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)

    base_dim = feats.shape[1]
    if target_edge_dim is None or target_edge_dim <= 0:
        return feats.astype(np.float32, copy=False)

    target_dim = int(target_edge_dim)
    if target_dim == base_dim:
        return feats.astype(np.float32, copy=False)

    if target_dim < base_dim:
        return feats[:, :target_dim].astype(np.float32, copy=False)

    # target_dim > base_dim – pad with zeros and optionally set the 3-D flag.
    padded = np.zeros((feats.shape[0], target_dim), dtype=np.float32)
    padded[:, :base_dim] = feats.astype(np.float32, copy=False)

    flag_idx = base_dim
    if flag_idx < target_dim:
        flag_value = float(_infer_has_3d(graph) if has_3d is None else has_3d)
        padded[:, flag_idx] = flag_value

    return padded


def attach_bond_features_from_smiles(
    g,
    smiles: str,
    *,
    target_edge_dim: int | None = None,
    has_3d: float | None = None,
):
    """
    Ensures g.edge_attr exists and aligns with g.edge_index using RDKit bond features.
    Handles *directed* edges (u->v, v->u).

    When ``target_edge_dim`` is provided, the returned attributes are padded (or
    truncated) to that width and the column immediately after the base RDKit
    features is reserved for the 3-D availability flag.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Bad SMILES: {smiles}"

    # Build map {frozenset({u,v}): feature}
    bond_map = {}
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_map[frozenset((u, v))] = _bond_vector(b)

    ei = getattr(g, "edge_index", None)
    if ei is None:
        # If you don’t have edge_index yet, build it from RDKit bonds (both directions)
        edges = []
        feats = []
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            f = _bond_vector(b)
            edges += [(u, v), (v, u)]
            feats += [f, f]
        g.edge_index = np.asarray(edges, dtype=np.int64).T  # [2, E]
        g.edge_attr = _finalise_edge_attr(
            np.asarray(feats, dtype=np.float32),
            g,
            target_edge_dim=target_edge_dim,
            has_3d=has_3d,
        )
        return g

    # Otherwise, fill edge_attr to match existing directed edges
    import numpy as np
    E = int(np.asarray(ei).shape[1])
    feats = np.zeros((E, 13), dtype=np.float32)
    u = np.asarray(ei[0]).astype(int)
    v = np.asarray(ei[1]).astype(int)
    for k in range(E):
        f = bond_map.get(frozenset((u[k], v[k])))
        if f is not None:
            feats[k] = f  # directed duplicates share the same undirected bond features
        # else: keep zeros (e.g., if edge is synthetic)

    g.edge_attr = _finalise_edge_attr(
        feats,
        g,
        target_edge_dim=target_edge_dim,
        has_3d=has_3d,
    )
    return g
