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

def attach_bond_features_from_smiles(g, smiles: str):
    """
    Ensures g.edge_attr exists and aligns with g.edge_index using RDKit bond features.
    Handles *directed* edges (u->v, v->u).
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
        g.edge_attr  = np.asarray(feats, dtype=np.float32)  # [E, 13]
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

    g.edge_attr = feats
    return g
