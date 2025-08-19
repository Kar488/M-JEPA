import importlib
import sys
import types

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _make_graph(x_dim: int = 3, edge_dim: int = 2, label: int = 1) -> Data:
    x = torch.randn(4, x_dim)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))


@pytest.fixture(scope="module")
def gs_module():
    # Provide minimal rdkit stubs so grid_search imports
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

    # Cleanup to avoid leaking stub modules
    for name in [
        "rdkit",
        "rdkit.Chem",
        "rdkit.Chem.Scaffolds",
        "rdkit.Chem.rdMolTransforms",
        "experiments.grid_search",
    ]:
        sys.modules.pop(name, None)


def test_ensure_graph_dataset_and_dataset_from_loader(gs_module):
    g1 = _make_graph(label=0)
    g2 = _make_graph(label=1)

    ds = gs_module._ensure_graph_dataset([g1, g2])
    assert len(ds.graphs) == 2
    assert ds.labels.tolist() == [0, 1]

    class IterLoader:
        def __iter__(self):
            yield g1
            yield g2

    shim = gs_module._dataset_from_loader(IterLoader())
    assert len(shim.graphs) == 2
    assert shim.labels.tolist() == [0, 1]


def test_build_configs_and_normalize(gs_module):
    from data.augment import AugmentationConfig
    cfgs = gs_module._build_configs(
        mask_ratios=(0.1,),
        contiguities=(False,),
        hidden_dims=(32, 64),
        num_layers_list=(2,),
        gnn_types=("gcn",),
        ema_decays=(0.9,),
        add_3d_options=(False,),
        augmentation_options=(AugmentationConfig(),),
        pretrain_batch_sizes=(8,),
        finetune_batch_sizes=(4,),
        pretrain_epochs_options=(1,),
        finetune_epochs_options=(1,),
        lrs=(1e-3,),
    )
    assert len(cfgs) == 2
    assert isinstance(cfgs[0], gs_module.Config)

    g1 = _make_graph()
    g2 = _make_graph()
    ds_dict = {"train": [g1], "val": [g2]}
    tr, va, te = gs_module._normalize_ds(ds_dict)
    assert len(tr) == 1 and len(va) == 1 and te is None

    tr_loader, va_loader, te_loader = gs_module._normalize_ds_to_loaders(( [g1], [g2], None ), 2, 1)
    assert len(list(tr_loader)) == 1
    assert len(list(va_loader)) == 1
    assert te_loader is None


def test_feat_dim_and_infer_dims_from_loader(gs_module):
    x = torch.randn(5, 7)
    ea = torch.randn(5, 4)
    assert gs_module._feat_dim(x) == 7
    assert gs_module._feat_dim(ea) == 4

    data_list = [_make_graph(x_dim=7, edge_dim=4), _make_graph(x_dim=7, edge_dim=4)]
    loader = DataLoader(data_list, batch_size=2)
    in_dim, edge_dim = gs_module._infer_dims_from_loader(loader)
    assert in_dim == 7 and edge_dim == 4


def test_aggregate_seed_metrics(gs_module):
    metrics = [{"acc": 0.5, "loss": 1.0}, {"acc": 0.7, "loss": 0.8}]
    agg = gs_module._aggregate_seed_metrics(metrics)
    assert agg["acc_mean"] == pytest.approx(0.6)
    assert "acc_std" in agg and "loss_ci95" in agg