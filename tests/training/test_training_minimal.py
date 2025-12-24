import importlib
import sys
import types
import logging
from dataclasses import dataclass

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from sklearn.metrics import brier_score_loss
from training.train_on_embeddings import train_linear_on_embeddings_classification


@dataclass
class GraphData:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None
    pos: np.ndarray | None = None


class GraphDataset:
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = np.asarray(labels) if labels is not None else None

    def __len__(self):
        return len(self.graphs)


def global_mean_pool(node_embeddings, graph_ptr=None):
    if graph_ptr is None:
        return node_embeddings.mean(dim=0)
    start = 0
    out = []
    for end in graph_ptr:
        out.append(node_embeddings[start:end].mean(dim=0))
        start = int(end)
    return torch.stack(out, dim=0)


@dataclass(frozen=True)
class AugmentationConfig:
    rotate: bool = False
    mask_angle: bool = False
    dihedral: bool = False

    @classmethod
    def from_dict(cls, cfg=None):
        cfg = cfg or {}
        return cls(
            rotate=bool(cfg.get("rotate", False)),
            mask_angle=bool(cfg.get("mask_angle", False)),
            dihedral=bool(cfg.get("dihedral", False)),
        )


def apply_graph_augmentations(g, **kwargs):
    return g


def delete_random_bond(g):
    return g


def mask_random_atom(g):
    return g


def remove_random_subgraph(g):
    return g


def mask_subgraph(g, mask_ratio, contiguous):
    return g, g


def generate_views(graph, structural_ops=None, geometric_ops=None):
    if structural_ops:
        out = structural_ops[0](graph)
        if isinstance(out, tuple):
            return list(out)
        return [out]
    return [graph]


@pytest.fixture()
def stub_data_modules(monkeypatch):
    data_dataset = types.ModuleType("data.mdataset")
    data_dataset.GraphData = GraphData
    data_dataset.GraphDataset = GraphDataset
    monkeypatch.setitem(sys.modules, "data.mdataset", data_dataset)

    utils_pooling = types.ModuleType("utils.pooling")
    utils_pooling.global_mean_pool = global_mean_pool
    monkeypatch.setitem(sys.modules, "utils.pooling", utils_pooling)

    data_augment = types.ModuleType("data.augment")
    data_augment.apply_graph_augmentations = apply_graph_augmentations
    data_augment.delete_random_bond = delete_random_bond
    data_augment.mask_random_atom = mask_random_atom
    data_augment.remove_random_subgraph = remove_random_subgraph
    data_augment.mask_subgraph = mask_subgraph
    data_augment.generate_views = generate_views
    data_augment.AugmentationConfig = AugmentationConfig
    monkeypatch.setitem(sys.modules, "data.augment", data_augment)

    yield

    for mod in [
        "data.mdataset",
        "utils.pooling",
        "data.augment",
        "training.unsupervised",
    ]:
        sys.modules.pop(mod, None)

    try:
        import data.mdataset as real_ds
        importlib.reload(real_ds)
    except Exception:
        pass


def make_graph():
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)

def test_train_contrastive_requires_two_graphs(stub_data_modules):
    from training.unsupervised import train_contrastive

    g = make_graph()
    dataset = GraphDataset([g])

    class DummyEncoder(torch.nn.Module):
        def forward(self, x, adj, edge_attr=None):
            return torch.as_tensor(x)

    encoder = DummyEncoder()

    with pytest.raises(ValueError, match="at least two graphs"):
        train_contrastive(
            dataset=dataset,
            encoder=encoder,
            epochs=1,
            batch_size=1,
            mask_ratio=0.0,
            lr=0.0,
            device="cpu",
            temperature=0.1,
            use_amp=False,
        )


def test_train_jepa_minimal_epoch(stub_data_modules):
    from training.unsupervised import train_jepa

    g = make_graph()
    dataset = GraphDataset([g])

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x, adj, edge_attr=None):
            return x @ self.weight + self.bias

    encoder = DummyEncoder()
    ema_encoder = DummyEncoder()

    class DummyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, h):
            return h @ self.weight + self.bias

    predictor = DummyPredictor()

    class DummyEMA:
        def update(self, model):
            pass

    ema = DummyEMA()

    losses = train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=1,
        batch_size=1,
        lr=0.0,
        reg_lambda=0.0,
        mask_ratio=0.5,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )
    assert losses == [pytest.approx(1.0)]


def test_train_jepa_respects_max_batches(stub_data_modules):
    from training.unsupervised import train_jepa

    graphs = [make_graph() for _ in range(4)]
    dataset = GraphDataset(graphs)

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x, adj, edge_attr=None):
            return x @ self.weight + self.bias

    encoder = DummyEncoder()
    ema_encoder = DummyEncoder()

    class CountingPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))
            self.calls = 0

        def forward(self, h):
            self.calls += 1
            return h @ self.weight + self.bias

    predictor = CountingPredictor()

    class DummyEMA:
        def update(self, model):
            pass

    ema = DummyEMA()

    losses = train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=3,
        batch_size=1,
        lr=0.0,
        reg_lambda=0.0,
        mask_ratio=0.5,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
        max_batches=2,
    )

    assert predictor.calls == 2
    assert len(losses) == 1


def test_train_jepa_runs_probe_interval(stub_data_modules, monkeypatch):
    from training.unsupervised import train_jepa

    graphs = [make_graph(), make_graph()]
    dataset = GraphDataset(graphs, labels=[0, 1])

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x, adj, edge_attr=None):
            return x @ self.weight + self.bias

    class DummyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, h):
            return h @ self.weight + self.bias

    class DummyEMA:
        def update(self, model):
            pass

    calls = {"compute_embeddings": 0, "linear_probe_classification": 0}

    probing = types.ModuleType("experiments.probing")

    def compute_embeddings_stub(*args, **kwargs):
        calls["compute_embeddings"] += 1
        return np.ones((len(dataset.graphs), 2), dtype=np.float32)

    def linear_probe_stub(*args, **kwargs):
        calls["linear_probe_classification"] += 1
        return {
            "probe_roc_auc": 0.5,
            "probe_pr_auc": 0.4,
            "probe_acc": 0.6,
            "probe_brier": 0.3,
        }

    probing.compute_embeddings = compute_embeddings_stub
    probing.linear_probe_classification = linear_probe_stub
    monkeypatch.setitem(sys.modules, "experiments.probing", probing)

    losses = train_jepa(
        dataset=dataset,
        encoder=DummyEncoder(),
        ema_encoder=DummyEncoder(),
        predictor=DummyPredictor(),
        ema=DummyEMA(),
        epochs=1,
        batch_size=1,
        lr=0.0,
        reg_lambda=0.0,
        mask_ratio=0.5,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
        probe_dataset=dataset,
        probe_interval=1,
    )

    assert losses == [pytest.approx(1.0)]
    assert calls["compute_embeddings"] == 1
    assert calls["linear_probe_classification"] == 1


def test_train_jepa_probe_disabled_matches_default(stub_data_modules, monkeypatch):
    from training.unsupervised import train_jepa

    graphs = [make_graph(), make_graph()]
    dataset = GraphDataset(graphs, labels=[0, 1])

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x, adj, edge_attr=None):
            return x @ self.weight + self.bias

    class DummyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, h):
            return h @ self.weight + self.bias

    class DummyEMA:
        def update(self, model):
            pass

    probing = types.ModuleType("experiments.probing")

    def compute_embeddings_stub(*args, **kwargs):
        raise AssertionError("Probe should not run when disabled.")

    def linear_probe_stub(*args, **kwargs):
        raise AssertionError("Probe should not run when disabled.")

    probing.compute_embeddings = compute_embeddings_stub
    probing.linear_probe_classification = linear_probe_stub
    monkeypatch.setitem(sys.modules, "experiments.probing", probing)

    torch.manual_seed(0)
    losses_with_disabled_probe = train_jepa(
        dataset=dataset,
        encoder=DummyEncoder(),
        ema_encoder=DummyEncoder(),
        predictor=DummyPredictor(),
        ema=DummyEMA(),
        epochs=1,
        batch_size=1,
        lr=0.0,
        reg_lambda=0.0,
        mask_ratio=0.5,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
        probe_dataset=dataset,
        probe_interval=0,
    )

    torch.manual_seed(0)
    losses_default = train_jepa(
        dataset=dataset,
        encoder=DummyEncoder(),
        ema_encoder=DummyEncoder(),
        predictor=DummyPredictor(),
        ema=DummyEMA(),
        epochs=1,
        batch_size=1,
        lr=0.0,
        reg_lambda=0.0,
        mask_ratio=0.5,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )

    assert losses_with_disabled_probe == losses_default


def test_train_jepa_falls_back_to_cpu(stub_data_modules, monkeypatch, caplog):
    torch = pytest.importorskip("torch")

    from training import unsupervised as unsup

    monkeypatch.setattr(unsup, "_DEVICE_FALLBACK_WARNED", False, raising=False)

    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        monkeypatch.setattr(cuda, "is_available", lambda: False, raising=False)

    g = make_graph()
    dataset = GraphDataset([g])

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x, adj, edge_attr=None):
            return x @ self.weight + self.bias

    encoder = DummyEncoder()
    ema_encoder = DummyEncoder()

    class DummyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, h):
            return h @ self.weight + self.bias

    predictor = DummyPredictor()

    class DummyEMA:
        def update(self, model):
            pass

    ema = DummyEMA()

    moved_devices: list[str] = []
    original_to = unsup.GraphBatch.to

    def tracking_to(self, device):
        moved_devices.append(str(torch.device(device)))
        return original_to(self, device)

    monkeypatch.setattr(unsup.GraphBatch, "to", tracking_to, raising=False)

    caplog.set_level(logging.WARNING, logger="training.unsupervised")

    losses = unsup.train_jepa(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=1,
        batch_size=1,
        lr=0.0,
        reg_lambda=0.0,
        mask_ratio=0.5,
        device="cuda",
        use_scheduler=False,
        use_amp=False,
        pin_memory=False,
        persistent_workers=False,
        num_workers=0,
    )

    assert losses == [pytest.approx(1.0)]
    assert moved_devices, "No graph batches were moved to a device"
    assert all(dev.startswith("cpu") for dev in moved_devices)
    assert "falling back to cpu" in caplog.text.lower()


def test_should_compile_models_requires_budget():
    import importlib

    unsup = importlib.import_module("training.unsupervised")
    device_cuda = torch.device("cuda")
    small_budget = max(1, unsup._COMPILE_WARMUP_BATCHES // 2)
    large_budget = max(1, unsup._COMPILE_WARMUP_BATCHES * 2)

    assert not unsup._should_compile_models(True, device_cuda, small_budget)
    assert unsup._should_compile_models(True, device_cuda, large_budget)
    assert not unsup._should_compile_models(False, device_cuda, large_budget)
    assert not unsup._should_compile_models(True, torch.device("cpu"), large_budget)


def test_train_contrastive_respects_max_batches(stub_data_modules):
    from training.unsupervised import train_contrastive

    graphs = [make_graph() for _ in range(6)]
    dataset = GraphDataset(graphs)

    class DummyEncoder(torch.nn.Module):
        def forward(self, x, adj, edge_attr=None):  # noqa: ARG002
            return torch.as_tensor(x, dtype=torch.float32)

    encoder = DummyEncoder()

    losses = train_contrastive(
        dataset=dataset,
        encoder=encoder,
        epochs=3,
        batch_size=2,
        mask_ratio=0.1,
        lr=0.0,
        temperature=0.1,
        device="cpu",
        use_amp=False,
        use_scheduler=False,
        max_batches=2,
    )

    assert len(losses) == 1


def test_train_linear_on_embeddings():
    X = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    metrics = train_linear_on_embeddings_classification(X, y, max_iter=100)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(1.0)
    
    p = float(np.mean(y))  # class prevalence
    baseline = brier_score_loss(y, np.full_like(y, p, dtype=float))
    assert metrics["brier"] < baseline + 1e-6
