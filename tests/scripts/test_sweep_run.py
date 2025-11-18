import argparse
import os
import sys
import types

import pytest

import models.encoder  # noqa: F401
import models.factory  # noqa: F401
from scripts import train_jepa as tj

torch = pytest.importorskip("torch")


class DummyDataset:
    def __init__(self, name: str):
        self.name = name
        self.graphs = [name]


def test_cmd_sweep_run_invokes_run_one(monkeypatch, tmp_path):
    def fake_loader(dirpath, **kwargs):
        return f"ds:{dirpath}"

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)

    calls = {}

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_run_one_config_method(**kwargs):
        calls["called"] = True
        assert kwargs["unlabeled_dataset_fn"] is None
        assert kwargs["eval_dataset_fn"] is None
        ds_tuple = kwargs.get("prebuilt_datasets")
        assert ds_tuple is not None
        assert isinstance(ds_tuple, tuple)
        assert len(ds_tuple) == 3
        calls["datasets"] = ds_tuple
        return {"metric": 1.0}

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = DummyConfig
    grid_mod._run_one_config_method = fake_run_one_config_method
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

    wb_mod = types.ModuleType("wandb_safety")
    wb_mod.wb_get_or_init = lambda *a, **k: None
    wb_mod.wb_summary_update = lambda *a, **k: None
    wb_mod.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", wb_mod)

    args = argparse.Namespace(
        add_3d=1,
        contiguity=0,
        mask_ratio=0.1,
        hidden_dim=32,
        num_layers=2,
        gnn_type="gcn",
        ema_decay=0.99,
        pretrain_batch_size=1,
        finetune_batch_size=1,
        pretrain_epochs=1,
        finetune_epochs=1,
        learning_rate=0.001,
        temperature=0.1,
        training_method="jepa",
        labeled_dir=str(tmp_path),
        unlabeled_dir=str(tmp_path),
        label_col="y",
        sample_unlabeled=0,
        sample_labeled=0,
        task_type="classification",
        seed=0,
        max_pretrain_batches=1,
        max_finetune_batches=1,
        num_workers=0,
        cache_dir=None,
        use_wandb=0,
        cache_datasets=0,
    )

    tj.cmd_sweep_run(args)
    assert calls.get("called")
    assert isinstance(calls["datasets"][0], str)
    assert isinstance(calls["datasets"][1], str)

def test_cmd_sweep_run_handles_float_return(monkeypatch, tmp_path):
    def fake_loader(dirpath, **kwargs):
        return f"ds:{dirpath}"

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)

    captures = {}

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyAugConfig:
        def __init__(self, *a, **k):
            pass

    def fake_run_one_config_method(**kwargs):
        return 0.5

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = DummyConfig
    grid_mod.AugmentationConfig = DummyAugConfig
    grid_mod._run_one_config_method = fake_run_one_config_method
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

    wb_mod = types.ModuleType("wandb_safety")
    wb_mod.wb_get_or_init = lambda *a, **k: None
    wb_mod.wb_summary_update = lambda payload: captures.update(payload)
    wb_mod.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", wb_mod)

    args = argparse.Namespace(
        add_3d=1,
        contiguity=0,
        mask_ratio=0.1,
        hidden_dim=32,
        num_layers=2,
        gnn_type="gcn",
        ema_decay=0.99,
        pretrain_batch_size=1,
        finetune_batch_size=1,
        pretrain_epochs=1,
        finetune_epochs=1,
        learning_rate=0.001,
        temperature=0.1,
        training_method="jepa",
        labeled_dir=str(tmp_path),
        unlabeled_dir=str(tmp_path),
        label_col="y",
        sample_unlabeled=0,
        sample_labeled=0,
        task_type="classification",
        seed=0,
        max_pretrain_batches=1,
        max_finetune_batches=1,
        num_workers=0,
        cache_dir=None,
        use_wandb=0,
        cache_datasets=0,
    )

    tj.cmd_sweep_run(args)
    assert captures.get("val_rmse") == 0.5
    assert "pair_id" in captures
    assert captures.get("training_method") == "jepa"


def test_cmd_sweep_run_dataset_cache(monkeypatch, tmp_path):
    def fake_loader(dirpath, **kwargs):
        if kwargs.get("label_col"):
            loads["labeled"] += 1
            return DummyDataset("labeled")
        loads["unlabeled"] += 1
        return DummyDataset("unlabeled")

    loads = {"unlabeled": 0, "labeled": 0}

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyAugConfig:
        def __init__(self, *a, **k):
            pass

    def fake_run_one_config_method(**kwargs):
        return {"metric": 1.0}

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = DummyConfig
    grid_mod.AugmentationConfig = DummyAugConfig
    grid_mod._run_one_config_method = fake_run_one_config_method
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

    wb_mod = types.ModuleType("wandb_safety")
    wb_mod.wb_get_or_init = lambda *a, **k: None
    wb_mod.wb_summary_update = lambda *a, **k: None
    wb_mod.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", wb_mod)

    base_cache = tmp_path / "cache"
    args = argparse.Namespace(
        add_3d=1,
        contiguity=0,
        mask_ratio=0.1,
        hidden_dim=32,
        num_layers=2,
        gnn_type="gcn",
        ema_decay=0.99,
        pretrain_batch_size=1,
        finetune_batch_size=1,
        pretrain_epochs=1,
        finetune_epochs=1,
        learning_rate=0.001,
        temperature=0.1,
        training_method="jepa",
        labeled_dir=str(tmp_path),
        unlabeled_dir=str(tmp_path),
        label_col="y",
        sample_unlabeled=0,
        sample_labeled=0,
        task_type="classification",
        seed=0,
        max_pretrain_batches=1,
        max_finetune_batches=1,
        num_workers=0,
        cache_dir=str(base_cache),
        use_wandb=0,
        cache_datasets=1,
    )

    tj.cmd_sweep_run(args)
    assert loads == {"unlabeled": 1, "labeled": 1}

    tj.cmd_sweep_run(args)
    assert loads == {"unlabeled": 1, "labeled": 1}

def test_cmd_sweep_run_expands_cache_dir_env(monkeypatch, tmp_path):
    loads = []

    def fake_loader(dirpath, **kwargs):
        loads.append(kwargs.get("cache_dir"))
        return DummyDataset(dirpath)

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyAugConfig:
        def __init__(self, *args, **kwargs):
            pass

    def fake_run_one_config_method(**kwargs):
        return {"metric": 1.0}

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = DummyConfig
    grid_mod.AugmentationConfig = DummyAugConfig
    grid_mod._run_one_config_method = fake_run_one_config_method
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

    wb_mod = types.ModuleType("wandb_safety")
    wb_mod.wb_get_or_init = lambda *a, **k: None
    wb_mod.wb_summary_update = lambda *a, **k: None
    wb_mod.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", wb_mod)

    cache_root = tmp_path / "resolved"
    monkeypatch.setenv("SWEEP_CACHE_DIR", str(cache_root))

    args = argparse.Namespace(
        add_3d=1,
        contiguity=0,
        mask_ratio=0.1,
        hidden_dim=32,
        num_layers=2,
        gnn_type="gcn",
        ema_decay=0.99,
        pretrain_batch_size=1,
        finetune_batch_size=1,
        pretrain_epochs=1,
        finetune_epochs=1,
        learning_rate=0.001,
        temperature=0.1,
        training_method="jepa",
        labeled_dir=str(tmp_path),
        unlabeled_dir=str(tmp_path),
        label_col="y",
        sample_unlabeled=0,
        sample_labeled=0,
        task_type="classification",
        seed=0,
        max_pretrain_batches=1,
        max_finetune_batches=1,
        num_workers=0,
        cache_dir="${env:SWEEP_CACHE_DIR}/inner",
        use_wandb=0,
        cache_datasets=0,
    )

    tj.cmd_sweep_run(args)

    expected = os.path.abspath(cache_root / "inner")
    assert loads == [expected, expected]
    assert args.cache_dir == expected
    assert os.path.isdir(expected)


def test_cmd_sweep_run_keeps_wandb_run_active_for_summary(monkeypatch, tmp_path):
    """Regression test for the missing summary/pair_id payload in sweeps."""

    def fake_loader(dirpath, **kwargs):
        return DummyDataset(f"ds:{dirpath}")

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)
    monkeypatch.setattr(tj, "resolve_device", lambda *_a, **_k: "cpu")

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyAugConfig:
        def __init__(self, *a, **k):
            pass

    fake_wandb = types.ModuleType("wandb")
    finish_calls = []

    class FakeConfig(dict):
        def as_dict(self):
            return dict(self)

    class FakeRun:
        def __init__(self):
            self.config = FakeConfig()
            self.summary = {}
            self.logged = []
            self.name = None

        def save(self):
            return None

    active_run = FakeRun()

    def fake_finish(**kwargs):
        finish_calls.append(kwargs)
        fake_wandb.run = None

    def fake_log(payload):
        if fake_wandb.run is not None:
            fake_wandb.run.logged.append(dict(payload))

    fake_wandb.run = active_run
    fake_wandb.config = FakeConfig()
    fake_wandb.finish = fake_finish
    fake_wandb.log = fake_log
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    # Ensure wandb_safety resolves from the real helper module so the regression
    # exercises the production summary writer.
    import importlib

    wandb_safety = importlib.import_module("scripts.wandb_safety")
    monkeypatch.setitem(sys.modules, "wandb_safety", wandb_safety)
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    dataset_cache = types.ModuleType("scripts.commands.dataset_cache")
    dataset_cache.resolve_env_path = lambda path: os.path.abspath(str(path))
    dataset_cache.prepare_cache_root = lambda base, enabled=True: str(tmp_path / "cache") if enabled else None

    def _load_or_build(kind, payload, builder, cache_root, **_):
        return builder()

    dataset_cache.load_or_build_dataset = _load_or_build
    monkeypatch.setitem(sys.modules, "scripts.commands.dataset_cache", dataset_cache)

    def fake_run_one_config_method(**kwargs):
        if not kwargs.get("defer_wandb_finish", False):
            fake_finish()
        return {"rmse_mean": 0.42, "best_step": 7}

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = DummyConfig
    grid_mod.AugmentationConfig = DummyAugConfig
    grid_mod._run_one_config_method = fake_run_one_config_method
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

    args = argparse.Namespace(
        add_3d=0,
        contiguity=0,
        mask_ratio=0.1,
        hidden_dim=16,
        num_layers=2,
        gnn_type="gine",
        ema_decay=0.99,
        pretrain_batch_size=1,
        finetune_batch_size=1,
        pretrain_epochs=1,
        finetune_epochs=1,
        learning_rate=0.001,
        temperature=0.1,
        training_method="jepa",
        labeled_dir=str(tmp_path),
        unlabeled_dir=str(tmp_path),
        label_col="y",
        sample_unlabeled=0,
        sample_labeled=0,
        task_type="regression",
        seed=0,
        max_pretrain_batches=1,
        max_finetune_batches=1,
        num_workers=0,
        cache_dir=None,
        use_wandb=1,
        cache_datasets=0,
        devices=0,
    )

    tj.cmd_sweep_run(args)
    assert finish_calls, "cmd_sweep_run should close the run via wb_finish_safely"
    assert "pair_id" in active_run.summary
    assert active_run.summary.get("training_method") == "jepa"
    assert pytest.approx(active_run.summary.get("val_rmse"), rel=1e-6) == 0.42
