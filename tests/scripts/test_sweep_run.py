import argparse
import os
import sys
import types

import pytest

pytest.importorskip("yaml")

import models.encoder  # noqa: F401
import models.factory  # noqa: F401
from scripts import train_jepa as tj

torch = pytest.importorskip("torch")


class DummyDataset:
    def __init__(self, name: str):
        self.name = name
        self.graphs = [name]


def test_resolve_augmentation_profile_maps_geometric_flags():
    from scripts.commands import sweep_run

    resolved = sweep_run._resolve_augmentation_profile("geom_only", seed=7)

    assert resolved["contiguity"] is True
    assert resolved["add_3d"] is True
    aug = resolved["augmentations"]
    assert aug["random_rotate"] and aug["mask_angle"] and aug["perturb_dihedral"]
    assert not aug["bond_deletion"]
    assert not aug["atom_masking"]
    assert not aug["subgraph_removal"]


def test_resolve_augmentation_profile_draws_single_corruption():
    from scripts.commands import sweep_run

    resolved = sweep_run._resolve_augmentation_profile("graph_light", seed=0)

    aug = resolved["augmentations"]
    corruption_total = sum(
        int(aug[key])
        for key in ("bond_deletion", "atom_masking", "subgraph_removal")
    )

    assert corruption_total == 1
    assert resolved["selected_corruption"] in {
        "bond_deletion",
        "atom_masking",
        "subgraph_removal",
    }
    assert not aug["random_rotate"]
    assert not aug["mask_angle"]
    assert not aug["perturb_dihedral"]


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


def _sweep_args(tmp_path, **overrides):
    params = dict(
        add_3d=0,
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
    params.update(overrides)
    return argparse.Namespace(**params)


def _prepare_sweep_module(monkeypatch, tmp_path, *, result_payload):
    import importlib

    def fake_loader(dirpath, **kwargs):
        return DummyDataset(f"ds:{dirpath}")

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)
    monkeypatch.setattr(tj, "resolve_device", lambda *_a, **_k: "cpu")

    dataset_cache = types.SimpleNamespace()
    dataset_cache.resolve_env_path = lambda path: os.path.abspath(str(path))
    dataset_cache.prepare_cache_root = lambda base, enabled=True: str(tmp_path / "cache") if enabled else None
    dataset_cache.load_or_build_dataset = (
        lambda _kind, _payload, builder, _cache_root=None, **__: builder()
    )

    sweep_mod = importlib.import_module("scripts.commands.sweep_run")
    monkeypatch.setattr(sweep_mod, "dataset_cache", dataset_cache, raising=False)

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyAugConfig:
        def __init__(self, *args, **kwargs):
            pass

    def fake_run_one_config_method(**kwargs):
        return dict(result_payload)

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = DummyConfig
    grid_mod.AugmentationConfig = DummyAugConfig
    grid_mod._run_one_config_method = fake_run_one_config_method
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

    return sweep_mod


def test_sweep_run_promotes_nested_training_method(monkeypatch, tmp_path):
    sweep_mod = _prepare_sweep_module(
        monkeypatch, tmp_path, result_payload={"rmse_mean": 0.42, "best_step": 3}
    )

    import importlib

    # Capture the method passed into the grid runner so we can assert the nested
    # sweep config overrides the CLI default.
    calls = {}

    def tracking_run_one_config_method(**kwargs):
        calls["method"] = kwargs.get("method")
        return {"rmse_mean": 0.42, "best_step": 3}

    grid_mod = importlib.import_module("experiments.grid_search")
    grid_mod._run_one_config_method = tracking_run_one_config_method

    class FakeConfig(dict):
        def as_dict(self):
            return dict(self)

    fake_wandb = types.ModuleType("wandb")
    fake_run = types.SimpleNamespace(
        config=FakeConfig({"training_method": {"value": "jepa"}}), summary={}, sweep=None
    )
    fake_wandb.run = fake_run
    fake_wandb.config = fake_run.config
    fake_wandb.finish = lambda **kwargs: None
    fake_wandb.log = lambda payload: payload
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    stub_ws = types.ModuleType("wandb_safety")
    stub_ws.wb_get_or_init = lambda *a, **k: fake_run
    stub_ws.wb_summary_update = lambda payload: fake_run.summary.update(payload)
    stub_ws.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", stub_ws)

    args = _sweep_args(tmp_path, training_method="contrastive", use_wandb=1)
    sweep_mod.cmd_sweep_run(args)

    assert calls.get("method") == "jepa"


def test_cmd_sweep_run_initializes_wandb_run_when_missing(monkeypatch, tmp_path):
    _prepare_sweep_module(
        monkeypatch,
        tmp_path,
        result_payload={"rmse_mean": 0.42, "best_step": 9},
    )

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None
    fake_wandb.config = {}
    fake_wandb.finish = lambda **kwargs: None
    fake_wandb.log = lambda payload: payload
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    import importlib

    wandb_safety = importlib.import_module("scripts.wandb_safety")

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.name = None

        def save(self):
            return None

    init_calls = []

    def fake_get_or_init(args):
        run = FakeRun()
        fake_wandb.run = run
        init_calls.append(args)
        return run

    stub_ws = types.ModuleType("wandb_safety")
    stub_ws.wb_get_or_init = fake_get_or_init
    stub_ws.wb_summary_update = wandb_safety.wb_summary_update
    stub_ws.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", stub_ws)
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    args = _sweep_args(tmp_path)
    tj.cmd_sweep_run(args)

    assert init_calls, "sweep-run should initialize a W&B run when none exists"
    assert fake_wandb.run is not None
    assert "pair_id" in fake_wandb.run.summary
    assert pytest.approx(fake_wandb.run.summary.get("val_rmse"), rel=1e-6) == 0.42


def test_cmd_sweep_run_logs_wandb_urls(monkeypatch, tmp_path, capsys):
    sweep_mod = _prepare_sweep_module(
        monkeypatch, tmp_path, result_payload={"rmse_mean": 0.42, "best_step": 2}
    )

    class FakeConfig(dict):
        def as_dict(self):
            return dict(self)

    fake_run = types.SimpleNamespace(
        config=FakeConfig(),
        summary={},
        sweep=types.SimpleNamespace(url="https://wandb.ai/ent/proj/sweeps/123"),
        url="https://wandb.ai/ent/proj/runs/456",
        entity="ent",
        project="proj",
        name=None,
    )

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = fake_run
    fake_wandb.config = fake_run.config
    fake_wandb.finish = lambda **kwargs: None
    fake_wandb.log = lambda payload: payload
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    stub_ws = types.ModuleType("wandb_safety")
    stub_ws.wb_get_or_init = lambda *a, **k: fake_run
    stub_ws.wb_summary_update = lambda payload: fake_run.summary.update(payload)
    stub_ws.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", stub_ws)

    args = _sweep_args(tmp_path, task_type="classification")
    sweep_mod.cmd_sweep_run(args)

    out = capsys.readouterr().out
    assert "[sweep-run] start sweep link: https://wandb.ai/ent/proj/sweeps/123" in out
    assert "[sweep-run] finish run link:   https://wandb.ai/ent/proj/runs/456" in out
    assert out.count("project:    https://wandb.ai/ent/proj") >= 2


def test_cmd_sweep_run_logs_env_urls_when_run_missing(monkeypatch, tmp_path, capsys):
    sweep_mod = _prepare_sweep_module(
        monkeypatch, tmp_path, result_payload={"rmse_mean": 0.24, "best_step": 1}
    )

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None
    fake_wandb.config = {}
    fake_wandb.finish = lambda **kwargs: None
    fake_wandb.log = lambda payload: payload
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    stub_ws = types.ModuleType("wandb_safety")
    stub_ws.wb_get_or_init = lambda *a, **k: None
    stub_ws.wb_summary_update = lambda payload: None
    stub_ws.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", stub_ws)

    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.ai")
    monkeypatch.setenv("SWEEP_ID", "abcd1234")
    monkeypatch.setenv("WANDB_RUN_ID", "run-42")

    args = _sweep_args(tmp_path)
    args.use_wandb = 0
    sweep_mod.cmd_sweep_run(args)

    out = capsys.readouterr().out
    assert "[sweep-run] start sweep link: https://wandb.ai/ent/proj/sweeps/abcd1234" in out
    assert "[sweep-run] start run link:   https://wandb.ai/ent/proj/runs/run-42" in out
    assert out.count("project:    https://wandb.ai/ent/proj") >= 2


def test_cmd_sweep_run_promotes_mean_auc(monkeypatch, tmp_path):
    sweep_mod = _prepare_sweep_module(
        monkeypatch, tmp_path, result_payload={"roc_auc_mean": 0.81}
    )

    captures = {}
    stub_ws = types.ModuleType("wandb_safety")
    stub_ws.wb_get_or_init = lambda *a, **k: None
    stub_ws.wb_summary_update = lambda payload: captures.update(payload)
    stub_ws.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", stub_ws)

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = types.SimpleNamespace(config={}, summary={})
    fake_wandb.finish = lambda **kwargs: None
    fake_wandb.log = lambda payload: payload
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    args = _sweep_args(tmp_path, task_type="classification")
    sweep_mod.cmd_sweep_run(args)

    assert captures["val_auc"] == pytest.approx(0.81)


def test_phase2_sweep_waits_for_wandb_run_before_summary(monkeypatch, tmp_path):
    _prepare_sweep_module(
        monkeypatch,
        tmp_path,
        result_payload={"val_auc": 0.88, "best_step": 5},
    )

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None
    fake_wandb.finish = lambda **kwargs: None
    fake_wandb.log = lambda payload: payload
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.name = None

    init_calls = []

    def fake_get_or_init(args):
        run = FakeRun()
        fake_wandb.run = run
        init_calls.append(args)
        return run

    def guarded_summary_update(payload):
        assert fake_wandb.run is not None, "summary update should only occur when a run is active"
        fake_wandb.run.summary.update(payload)

    stub_ws = types.ModuleType("wandb_safety")
    stub_ws.wb_get_or_init = fake_get_or_init
    stub_ws.wb_summary_update = guarded_summary_update
    stub_ws.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", stub_ws)
    monkeypatch.setenv("MJEPACI_STAGE", "phase2")

    args = _sweep_args(
        tmp_path,
        training_method="contrastive",
        task_type="classification",
    )
    tj.cmd_sweep_run(args)

    assert init_calls, "phase-2 sweeps should initialize W&B before logging"
    assert fake_wandb.run is not None
    assert "pair_id" in fake_wandb.run.summary
    assert pytest.approx(fake_wandb.run.summary.get("val_auc"), rel=1e-6) == 0.88


def test_cmd_sweep_run_updates_full_config(monkeypatch, tmp_path):
    sweep_cfg = {
        "training_method": {"value": "jepa"},
        "model": {"gnn_type": {"value": "gat"}},
        "optim": {"lr": {"value": 0.1}},
    }

    def fake_loader(dirpath, **kwargs):
        return DummyDataset(dirpath)

    monkeypatch.setattr(tj, "load_directory_dataset", fake_loader)

    class DummyConfig:
        def __init__(self, payload):
            self._payload = payload
            self.updated_payload = None

        def as_dict(self):
            return self._payload

        def update(self, payload, allow_val_change=True):
            self.updated_payload = payload

        def __setitem__(self, key, value):
            self.update({key: value})

    class DummyRun:
        def __init__(self, cfg):
            self.config = cfg
            self.project = "proj"
            self.entity = "ent"
            self.url = "http://example"

    dummy_config = DummyConfig(sweep_cfg)
    dummy_run = DummyRun(dummy_config)

    wandb_mod = types.SimpleNamespace(run=dummy_run, config=dummy_config)
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)

    wb_mod = types.ModuleType("wandb_safety")
    wb_mod.wb_get_or_init = lambda *a, **k: dummy_run
    wb_mod.wb_summary_update = lambda *a, **k: None
    wb_mod.wb_finish_safely = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb_safety", wb_mod)

    grid_mod = types.ModuleType("experiments.grid_search")
    grid_mod.Config = lambda **kwargs: kwargs
    grid_mod.AugmentationConfig = lambda *a, **k: None
    grid_mod._run_one_config_method = lambda **kwargs: {"val_rmse": 0.1}
    monkeypatch.setitem(sys.modules, "experiments.grid_search", grid_mod)

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
        task_type="regression",
        seed=0,
        max_pretrain_batches=1,
        max_finetune_batches=1,
        num_workers=0,
        cache_dir=None,
        use_wandb=1,
        cache_datasets=0,
    )

    def _flatten(d, parent_key: str = "", sep: str = "."):
        out = {}
        for k, v in (d or {}).items():
            nk = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                if "value" in v:
                    out[k] = v["value"]
                    out[nk] = v["value"]
                    out.update(_flatten(v, nk, sep))
                    continue
                out.update(_flatten(v, nk, sep))
            else:
                out[nk] = v
                out[k] = v
        return out

    expected = {k: v for k, v in _flatten(sweep_cfg).items() if not isinstance(v, dict)}

    tj.cmd_sweep_run(args)

    for key, value in expected.items():
        assert dummy_config.updated_payload.get(key) == value
