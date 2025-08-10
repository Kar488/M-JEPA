from __future__ import annotations

import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as MT

from data.mdataset import GraphData


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
