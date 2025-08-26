import importlib
import sys
import types
import logging

import pytest

# optional torch and torch_geometric
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.augment import AugmentationConfig


def _make_graph(x_dim: int = 3, edge_dim: int = 2, label: int = 1, smiles: str = "") -> Data:
    x = torch.randn(4, x_dim)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))
    if smiles:
        g.smiles = smiles
    return g


@pytest.fixture(scope="module")
def gs_module():
    rdkit = types.ModuleType("rdkit")
    Chem = types.SimpleNamespace(
        MolFromSmiles=lambda sm: object(),
        MolToSmiles=lambda mol, isomericSmiles=True: "",
    )
    Scaffolds = types.SimpleNamespace(
        MurckoScaffold=types.SimpleNamespace(GetScaffoldForMol=lambda mol: mol)
    )
    Chem.Scaffolds = Scaffolds
    Chem.rdMolTransforms = types.SimpleNamespace()
    rdkit.Chem = Chem
    sys.modules.setdefault("rdkit", rdkit)
    sys.modules.setdefault("rdkit.Chem", Chem)
    sys.modules.setdefault("rdkit.Chem.Scaffolds", Scaffolds)
    sys.modules.setdefault("rdkit.Chem.rdMolTransforms", Chem.rdMolTransforms)

    gs = importlib.import_module("experiments.grid_search")
    yield gs

    for name in [
        "rdkit",
        "rdkit.Chem",
        "rdkit.Chem.Scaffolds",
        "rdkit.Chem.rdMolTransforms",
        "experiments.grid_search",
    ]:
        sys.modules.pop(name, None)


def test_ensure_graph_dataset_branches(gs_module):
    g1 = _make_graph(label=0, smiles="C")

    # None branch
    assert gs_module._ensure_graph_dataset(None) is None

    # Existing dataset with graphs attr but no labels
    ds_obj = types.SimpleNamespace(graphs=[g1])
    ds_with_labels = gs_module._ensure_graph_dataset(ds_obj)
    assert ds_with_labels.labels.tolist() == [0]

    # Indexable sequence -> shim
    shim = gs_module._ensure_graph_dataset([g1])
    assert isinstance(shim, gs_module._GraphDatasetShim)
    assert shim.labels.tolist() == [0]

    # Single PyG Data
    single = gs_module._ensure_graph_dataset(g1)
    assert single.labels.tolist() == [0]

    # Unrecognized object returned as-is
    obj = {"foo": 1}
    assert gs_module._ensure_graph_dataset(obj) is obj


def test_dataset_from_loader_branches(gs_module):
    g1 = _make_graph(label=1)

    # None loader
    assert gs_module._dataset_from_loader(None) is None

    # Loader with dataset attribute
    loader_with_ds = DataLoader([g1], batch_size=1)
    ds_from_loader = gs_module._dataset_from_loader(loader_with_ds)
    assert len(ds_from_loader.graphs) == 1
    assert ds_from_loader.labels.tolist() == [1]

    # Iterable loader without dataset attribute
    class IterLoader:
        def __iter__(self):
            yield g1

    shim = gs_module._dataset_from_loader(IterLoader())
    assert len(shim.graphs) == 1
    assert shim.labels.tolist() == [1]

    # Loader yielding tuples
    class TupleLoader:
        def __iter__(self):
            yield (g1, torch.tensor([1]))

    shim2 = gs_module._dataset_from_loader(TupleLoader())
    assert len(shim2.graphs) == 1
    assert shim2.labels.tolist() == [1]


def test_build_configs_cartesian_product(gs_module):
    cfgs = gs_module._build_configs(
        mask_ratios=(0.1, 0.2),
        contiguities=(False,),
        hidden_dims=(32,),
        num_layers_list=(2, 3),
        gnn_types=("gcn",),
        ema_decays=(0.9,),
        add_3d_options=(False, True),
        augmentation_options=(AugmentationConfig(False, False, False),),
        pretrain_batch_sizes=(8,),
        finetune_batch_sizes=(4,),
        pretrain_epochs_options=(1,),
        finetune_epochs_options=(1,),
        lrs=(1e-3,),
        temperatures=(0.1,),
    )
    assert len(cfgs) == 8
    assert {cfg.mask_ratio for cfg in cfgs} == {0.1, 0.2}
    assert {cfg.num_layers for cfg in cfgs} == {2, 3}
    assert {cfg.add_3d for cfg in cfgs} == {False, True}


def test_run_grid_search_integration(gs_module, monkeypatch, caplog):
    g = _make_graph()

    def dataset_fn(add_3d=False):
        return ([g], [g], None)

    calls = []

    def fake_run_one_config_method(cfg, method, *args):
        # prebuilt loaders/datasets should be passed at positions 16/17 in args
        prebuilt_loaders = args[16]
        prebuilt_datasets = args[17]
        assert prebuilt_loaders is not None
        assert prebuilt_datasets is not None
        calls.append((cfg, method))
        return {"metric": 0.5}

    monkeypatch.setattr(gs_module, "_run_one_config_method", fake_run_one_config_method)

    with caplog.at_level(logging.INFO):
        df = gs_module.run_grid_search(
            dataset_fn=dataset_fn,
            methods=("m1", "m2"),
            seeds=(1,),
            mask_ratios=(0.1,),
            contiguities=(False,),
            hidden_dims=(16,),
            num_layers_list=(2,),
            gnn_types=("gcn",),
            ema_decays=(0.9,),
            add_3d_options=(False,),
            augmentation_options=(AugmentationConfig(False, False, False),),
            pretrain_batch_sizes=(1,),
            finetune_batch_sizes=(1,),
            pretrain_epochs_options=(1,),
            finetune_epochs_options=(1,),
            lrs=(1e-3,),
            temperatures=(0.1,),
            device="cpu",
            disable_tqdm=True,
        )

    assert len(calls) == 2
    assert list(df["method"]) == ["m1", "m2"]
    assert "metric" in df.columns
    assert "Running grid search over 1 configs" in caplog.text


def test_target_pretrain_samples_caps_batches(gs_module, monkeypatch):
    dataset = [_make_graph() for _ in range(10)]

    def dataset_fn(add_3d=False):
        return dataset

    calls = []

    def fake_contrastive(*args, **kwargs):
        calls.append((kwargs.get("batch_size"), kwargs.get("max_batches")))

    monkeypatch.setattr(gs_module, "train_contrastive", fake_contrastive)
    monkeypatch.setattr(gs_module, "train_linear_head", lambda *a, **k: {})
    class Dummy:
        def to(self, device):
            return self

    monkeypatch.setattr(gs_module, "build_encoder", lambda *a, **k: Dummy())
    gs_module._HAS_PROBE = False

    gs_module.run_grid_search(
        dataset_fn=dataset_fn,
        methods=("contrastive",),
        seeds=(0,),
        mask_ratios=(0.1,),
        contiguities=(False,),
        hidden_dims=(8,),
        num_layers_list=(2,),
        gnn_types=("gcn",),
        ema_decays=(0.9,),
        add_3d_options=(False,),
        augmentation_options=(AugmentationConfig(False, False, False),),
        pretrain_batch_sizes=(2, 4),
        finetune_batch_sizes=(1,),
        pretrain_epochs_options=(3,),
        finetune_epochs_options=(1,),
        lrs=(1e-3,),
        temperatures=(0.1,),
        device="cpu",
        disable_tqdm=True,
        target_pretrain_samples=8,
    )

    bs_to_mb = {bs: mb for bs, mb in calls}
    assert bs_to_mb[2] == 4
    assert bs_to_mb[4] == 2
