import sys

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms as MT

from data.augment import (
    AugmentationConfig,
    apply_graph_augmentations,
    delete_random_bond,
    generate_views,
    mask_random_angle,
    mask_random_atom,
    mask_subgraph,
    perturb_dihedral,
    random_rotation,
    remove_random_subgraph,
)
from data.mdataset import GraphDataset, GraphData


@pytest.fixture(autouse=True)
def _silence_tqdm(monkeypatch):
    """Make tqdm think we're not in a TTY so it doesn't draw a progress bar."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)


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
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
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
        js = [
            n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k
        ]
        ks = [
            n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j
        ]
        for i in js:
            for ell in ks:
                out.append(MT.GetDihedralDeg(conf, int(i), int(j), int(k), int(ell)))
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


def test_delete_random_bond_produces_valid_view():
    g = GraphDataset.smiles_to_graph("CCO", random_seed=0)
    e0 = g.edge_index.shape[1]
    np.random.seed(0)
    view = generate_views(g, structural_ops=[delete_random_bond])[0]
    assert view.edge_index.shape[1] == e0 - 2
    if view.edge_attr is not None:
        assert view.edge_attr.shape[0] == e0 - 2
    view.to_tensors()  # should succeed


def test_mask_random_atom_produces_valid_view():
    g = GraphDataset.smiles_to_graph("CCO", random_seed=0)
    np.random.seed(0)
    view = generate_views(g, structural_ops=[mask_random_atom])[0]
    assert np.any(np.all(view.x == 0, axis=1))
    idx = int(np.where(np.all(view.x == 0, axis=1))[0][0])
    if view.edge_attr is not None:
        mask = (view.edge_index[0] == idx) | (view.edge_index[1] == idx)
        assert np.all(view.edge_attr[mask] == 0)
    view.to_tensors()


def test_remove_random_subgraph_produces_valid_view():
    g = GraphDataset.smiles_to_graph("CCCC", random_seed=0)
    n0 = g.num_nodes()
    np.random.seed(0)
    view = generate_views(g, structural_ops=[remove_random_subgraph])[0]
    assert view.num_nodes() < n0
    if view.edge_index.size > 0:
        assert view.edge_index.max() < view.num_nodes()
    view.to_tensors()


def test_generate_views_preserves_pos():
    pytest.importorskip("rdkit")
    g = GraphDataset.smiles_to_graph("CCO", add_3d=True, random_seed=0)
    np.random.seed(0)
    views = generate_views(
        g, structural_ops=[lambda x: mask_subgraph(x, mask_ratio=0.5, contiguous=False)]
    )
    assert len(views) == 2
    for v in views:
        assert v.pos is not None
        assert v.pos.shape[0] == v.x.shape[0]


def test_mask_subgraph_contiguous_respects_connectivity():
    x = np.stack(
        [np.ones(5, dtype=np.float32), np.arange(5, dtype=np.float32)], axis=1
    )
    edge_index = np.array(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=np.int64
    )
    g = GraphData(x=x, edge_index=edge_index, edge_attr=None, pos=None)
    adjacency = {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3},
    }

    for seed in range(5):
        np.random.seed(seed)
        _, tgt = mask_subgraph(g, mask_ratio=0.4, contiguous=True)
        original_nodes = {int(v) for v in np.asarray(tgt.x)[:, 1]}
        assert len(original_nodes) == 2
        start = next(iter(original_nodes))
        reachable = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in adjacency[cur]:
                if nb in original_nodes and nb not in reachable:
                    reachable.add(nb)
                    stack.append(nb)
        assert reachable == original_nodes


def test_run_ablation_forwards_augment(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    from experiments import ablation

    calls = []

    def fake_train_jepa(
        *, random_rotate=False, mask_angle=False, perturb_dihedral=False, **kwargs
    ):
        calls.append((random_rotate, mask_angle, perturb_dihedral))
        return []

    def fake_train_head(*args, **kwargs):
        return {"roc_auc": 0.0, "pr_auc": 0.0, "rmse": 0.0, "mae": 0.0}

    monkeypatch.setattr(ablation, "train_jepa", fake_train_jepa)
    monkeypatch.setattr(ablation, "train_linear_head", fake_train_head)

    df = ablation.run_ablation(
        augmentations=AugmentationConfig(
            random_rotate=True, mask_angle=True, perturb_dihedral=True
        )
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert calls and all(rr and ma and pd for rr, ma, pd in calls)
