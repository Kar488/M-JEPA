import argparse
import sys
import types

try:  # pragma: no cover - torch may be unavailable
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = types.ModuleType("torch")
    torch.load = lambda *a, **k: {}
    torch.manual_seed = lambda *a, **k: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__path__ = []
    nn_mod.Module = object
    torch.nn = nn_mod
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn_mod

sys.modules.setdefault(
    "models.factory", types.SimpleNamespace(build_encoder=lambda **k: None)
)
sys.modules.setdefault("models.encoder", types.SimpleNamespace(GNNEncoder=object))

from scripts import train_jepa as tj  # noqa: E402


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
    )

    tj.cmd_sweep_run(args)
    assert calls.get("called")
