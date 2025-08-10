import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms as MT

from data.augment import (
    random_rotation,
    mask_random_angle,
    perturb_dihedral,
    apply_graph_augmentations,
)
from data.dataset import GraphDataset


def _build_mol(smiles: str = "CCO") -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def _coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


def _pairwise_dists(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _angles(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    out = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) < 2:
            continue
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                out.append(MT.GetAngleDeg(conf, int(i), int(j), int(k)))
    return np.array(out, dtype=float)


def _dihedrals(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    out = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        js = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        ks = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        for i in js:
            for l in ks:
                out.append(MT.GetDihedralDeg(conf, int(i), int(j), int(k), int(l)))
    return np.array(out, dtype=float)


def _graph_to_mol(g):
    mol = Chem.RWMol()
    n = g.x.shape[0]
    for i in range(n):
        z = int(g.x[i, 0])
        mol.AddAtom(Chem.Atom(z))
    added = set()
    for u, v in g.edge_index.T:
        tup = tuple(sorted((int(u), int(v))))
        if tup in added:
            continue
        mol.AddBond(tup[0], tup[1], Chem.BondType.SINGLE)
        added.add(tup)
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(n)
    for i in range(n):
        x, y, z = map(float, g.x[i, -3:])
        conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf, assignId=True)
    return mol


def test_random_rotation_rotates_without_distortion():
    mol = _build_mol()
    coords0 = _coords(mol)
    dists0 = _pairwise_dists(coords0)
    np.random.seed(0)
    random_rotation(mol)
    coords1 = _coords(mol)
    dists1 = _pairwise_dists(coords1)
    assert not np.allclose(coords0, coords1)
    assert np.allclose(dists0, dists1, atol=1e-5)


def test_mask_random_angle_sets_angle_to_zero():
    mol = _build_mol()
    ang0 = _angles(mol)
    assert not np.any(np.isclose(ang0, 0.0, atol=1e-3))
    np.random.seed(0)
    mask_random_angle(mol)
    ang1 = _angles(mol)
    assert np.any(np.isclose(ang1, 0.0, atol=1e-3))
    assert not np.allclose(np.sort(ang0), np.sort(ang1))


def test_perturb_dihedral_changes_dihedral():
    mol = _build_mol()
    dih0 = _dihedrals(mol)
    np.random.seed(0)
    perturb_dihedral(mol, max_deg=20.0)
    dih1 = _dihedrals(mol)
    diffs = np.abs(((dih1 - dih0 + 180) % 360) - 180)
    assert np.any(diffs > 1e-3)
    assert np.max(diffs) <= 20.0 + 1e-5


def test_apply_graph_augmentations_rotate():
    g = GraphDataset.smiles_to_graph("CCO", add_3d=True, random_seed=0)
    coords0 = g.x[:, -3:].copy()
    mol0 = _graph_to_mol(g)
    ang0 = _angles(mol0)
    dih0 = _dihedrals(mol0)
    np.random.seed(0)
    apply_graph_augmentations(g, rotate=True)
    coords1 = g.x[:, -3:]
    mol1 = _graph_to_mol(g)
    ang1 = _angles(mol1)
    dih1 = _dihedrals(mol1)
    assert not np.allclose(coords0, coords1)
    assert np.allclose(_pairwise_dists(coords0), _pairwise_dists(coords1), atol=1e-5)
    assert np.allclose(np.sort(ang0), np.sort(ang1), atol=1e-5)
    assert np.allclose(np.sort(dih0), np.sort(dih1), atol=1e-5)
