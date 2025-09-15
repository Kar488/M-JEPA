import argparse
import pytest
import scripts.commands.finetune as ft


def test_cmd_finetune_missing_modules_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "load_directory_dataset", None, raising=False)
    monkeypatch.setattr(ft, "build_encoder", None, raising=False)
    monkeypatch.setattr(ft, "train_linear_head", None, raising=False)
    class DummyLogger:
        def info(self, *a, **k): pass
        def error(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def exception(self, *a, **k): pass
    monkeypatch.setattr(ft, "logger", DummyLogger(), raising=False)
    args = argparse.Namespace(
        labeled_dir=str(tmp_path),
        gnn_type="gcn",
        hidden_dim=4,
        num_layers=2,
        task_type="regression",
        epochs=1,
        batch_size=1,
        lr=0.001,
        patience=1,
        devices=1,
        device="cpu",
        ema_decay=0.99,
        seeds=[0],
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        label_col="y",
    )
    with pytest.raises(SystemExit) as ex:
        ft.cmd_finetune(args)
    assert ex.value.code == 3
