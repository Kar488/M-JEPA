import json
import math
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import experiments.case_study as cs


class DummyGraph:
    def __init__(self, value: float = 1.0, edge_dim: int = 2):
        self.x = torch.full((4, 3), value, dtype=torch.float32)
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        self.edge_attr = torch.zeros(self.edge_index.shape[1], edge_dim, dtype=torch.float32)
        self.smiles = "C"


class DummyDataset:
    def __init__(self, values: list[float], edge_dim: int = 2):
        self.graphs = [DummyGraph(v, edge_dim=edge_dim) for v in values]
        self.labels = np.asarray(values, dtype=float)

    def __len__(self) -> int:
        return len(self.graphs)


class MeanEncoder(torch.nn.Module):
    def forward(self, x, adj):  # noqa: D401 - simple helper
        return x


class LinearHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 0.5)

    def forward(self, batch):
        return self.linear(batch)


def test_to_list_and_metric_helpers():
    assert cs._to_list([1, 2]) == [1, 2]
    assert cs._to_list((1, 2)) == [1, 2]
    assert cs._to_list(np.array([1, 2])) == [1, 2]
    assert cs._to_list(torch.tensor([1, 2])) == [1, 2]
    assert cs._canonical_metric_name("AUROC") == "roc_auc"
    assert cs._metric_is_higher_better("roc_auc") is True
    assert cs._metric_is_higher_better("rmse") is False
    assert cs._compute_met_benchmark("roc_auc", 0.7, 0.6) is True
    assert cs._compute_met_benchmark("rmse", 0.4, 0.5) is True


def test_manifest_and_state_helpers(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"hyperparameters": {"foo": 1}}))
    assert cs._load_manifest_config(str(manifest)) == {"foo": 1}
    bad = tmp_path / "bad.json"
    bad.write_text("not-json")
    assert cs._load_manifest_config(str(bad)) == {}
    assert cs._load_manifest_config(None) == {}

    assert cs._extract_state_config({"encoder_cfg": {"dim": 64}}) == {"dim": 64}
    assert cs._extract_state_config(None) == {}


def test_resolve_threshold_rule(monkeypatch):
    calls: list[tuple[str | None, str | None]] = []

    class DummyRule(SimpleNamespace):
        metric: str
        threshold: float

    rule = DummyRule(metric="roc_auc", threshold=0.7)

    def fake_resolve(dataset, task):
        calls.append((dataset, task))
        if dataset == "tox21":
            return rule
        raise KeyError("missing")

    monkeypatch.setattr(cs, "resolve_metric_threshold", fake_resolve)
    assert cs._resolve_threshold_rule("tox21", "NR-AR") is rule
    assert cs._resolve_threshold_rule("other", "task") is None
    assert calls == [("tox21", "NR-AR"), ("other", "task")]


def test_predict_logits_probs_in_chunks():
    dataset = DummyDataset([0.0, 1.0, 2.0])
    encoder = MeanEncoder()
    head = LinearHead()
    logits, probs = cs._predict_logits_probs_in_chunks(
        dataset,
        indices=[0, 1, 2],
        encoder=encoder,
        head=head,
        device="cpu",
        edge_dim=2,
        batch_size=2,
    )
    assert logits.shape == (3, 1)
    assert probs.shape == (3, 1)
    assert torch.all(probs.ge(0.0) & probs.le(1.0))


def test_predict_logits_probs_empty_batch():
    dataset = DummyDataset([])
    encoder = MeanEncoder()
    head = LinearHead()
    logits, probs = cs._predict_logits_probs_in_chunks(
        dataset,
        indices=[],
        encoder=encoder,
        head=head,
        device="cpu",
        edge_dim=2,
    )
    assert logits.numel() == 0 and probs.numel() == 0


class DummyRidge:
    def __init__(self, alpha: float, random_state: int):
        self.alpha = alpha
        self.random_state = random_state
        self._fit_mean = 0.0

    def fit(self, X, y):
        self._fit_mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._fit_mean, dtype=float)


def test_evaluate_case_study_with_baseline(tmp_path, monkeypatch):
    values = [0.0, 1.0, 0.0, 1.0]
    dataset = DummyDataset(values)
    encoder = MeanEncoder()
    head = LinearHead()
    all_labels = np.asarray(values, dtype=float)
    baseline_path = tmp_path / "emb.npy"
    np.save(baseline_path, np.arange(len(values) * 2, dtype=float).reshape(len(values), 2))

    monkeypatch.setattr("sklearn.linear_model.Ridge", DummyRidge)

    mean_true, mean_rand, mean_pred, baselines, metrics, calibrator = cs._evaluate_case_study(
        dataset=dataset,
        encoder=encoder,
        head=head,
        all_labels=all_labels,
        train_idx=[0, 1],
        val_idx=[2],
        test_idx=[3],
        triage_pct=0.5,
        calibrate=False,
        device="cpu",
        edge_dim=2,
        seed=42,
        baseline_embeddings={"ridge": str(baseline_path)},
    )

    assert math.isclose(mean_true, 1.0)
    assert math.isfinite(mean_rand)
    assert math.isfinite(mean_pred)
    assert "ridge" in baselines
    assert set(metrics) >= {"roc_auc", "pr_auc", "brier", "ece"}
    assert calibrator == {"enabled": False, "fit_split": "val", "status": "disabled"}


def test_tox21_cache_name_builds_path(tmp_path, monkeypatch):
    csv_path = tmp_path / "tox21.csv"
    csv_path.write_text("smiles,NR-AR\nC,1\nN,0\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"

    class HaltExecution(RuntimeError):
        pass

    class StubDataset:
        @classmethod
        def from_smiles_list(cls, *args, **kwargs):
            raise HaltExecution

    monkeypatch.setattr(cs, "_load_real_graphdataset", lambda: StubDataset)
    monkeypatch.setattr(cs, "set_seed", lambda *args, **kwargs: None)

    with pytest.raises(HaltExecution):
        cs.run_tox21_case_study(
            csv_path=str(csv_path),
            task_name="NR-AR",
            cache_dir=str(cache_dir),
            encoder_checkpoint=None,
        )

    assert cache_dir.exists()

