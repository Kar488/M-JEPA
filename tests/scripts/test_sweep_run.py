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