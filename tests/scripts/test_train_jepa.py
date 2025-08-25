import sys
import argparse
import types
import pytest

try:  # pragma: no cover - torch may be unavailable
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda *a, **k: None,
        save=lambda *a, **k: None,
        load=lambda *a, **k: {},
        nn=types.SimpleNamespace(Module=object),
    )
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn

try:  # pragma: no cover - optional dependency
    import models.factory  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault(
        "models.factory", types.SimpleNamespace(build_encoder=lambda *a, **k: None)
    )
try:  # pragma: no cover - optional dependency
    import models.encoder  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("models.encoder", types.SimpleNamespace(GNNEncoder=object))

from scripts import train_jepa as tj


# ---------------------------------------------------------------------------
# iter_augmentation_options
# ---------------------------------------------------------------------------

def test_iter_augmentation_options_all_combinations():
    opts = list(tj.iter_augmentation_options([0, 1], [0, 1], [0, 1]))
    combos = {(c.random_rotate, c.mask_angle, c.perturb_dihedral) for c in opts}
    assert combos == {
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    }


def test_iter_augmentation_options_forced_flags():
    cfgs = list(tj.iter_augmentation_options([1], [True], [False]))
    assert len(cfgs) == 1
    cfg = cfgs[0]
    assert cfg.random_rotate is True
    assert cfg.mask_angle is True
    assert cfg.perturb_dihedral is False


# ---------------------------------------------------------------------------
# Fallback imports for optional dependencies
# ---------------------------------------------------------------------------

def test_load_directory_dataset_missing_graphdataset(monkeypatch):
    monkeypatch.setattr(tj, "GraphDataset", None)
    monkeypatch.setattr(tj, "_GRAPH_DATASET_IMPORT_ERROR", ImportError("boom"))
    with pytest.raises(ImportError) as exc:
        tj.load_directory_dataset("/tmp")
    assert "GraphDataset is unavailable" in str(exc.value)


def test_cmd_pretrain_missing_ema(monkeypatch, tmp_path):
    class DummyDataset:
        graphs = [types.SimpleNamespace(x=types.SimpleNamespace(shape=(1, 3)), edge_attr=None)]

        def __len__(self):
            return 1

    monkeypatch.setattr(tj, "load_directory_dataset", lambda *a, **k: DummyDataset())
    monkeypatch.setattr(tj, "build_encoder", lambda **k: object())
    monkeypatch.setattr(tj, "train_jepa", lambda **k: [0.0])
    monkeypatch.setattr(tj, "MLPPredictor", lambda *a, **k: object())
    monkeypatch.setattr(
        tj,
        "plot_training_curves",
        lambda *a, **k: types.SimpleNamespace(savefig=lambda p, dpi=200: open(p, "wb").close()),
    )
    monkeypatch.setattr(
        tj,
        "maybe_init_wandb",
        lambda *a, **k: types.SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None),
    )
    monkeypatch.setattr(tj, "EMA", None)

    args = argparse.Namespace(
        unlabeled_dir=str(tmp_path),
        gnn_type="gcn",
        seeds=(42,),
        hidden_dim=16,
        num_layers=2,
        mask_ratio=0.15,
        epochs=1,
        batch_size=1,
        lr=0.001,
        ema_decay=0.99,
        contrastive=False,
        output=str(tmp_path / "encoder.pt"),
        contiguous=False,
        device="cpu",
        aug_rotate=False,
        aug_mask_angle=False,
        aug_dihedral=False,
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        add_3d=False,
        plot_dir=str(tmp_path),
    )

    with pytest.raises(TypeError) as exc:
        tj.cmd_pretrain(args)
    assert "NoneType" in str(exc.value)


# ---------------------------------------------------------------------------
# CLI parsing and exit codes
# ---------------------------------------------------------------------------

def test_cli_pretrain_exit_code(tmp_path, monkeypatch):
    parser = tj.build_parser()
    args = parser.parse_args(["pretrain", f"--unlabeled-dir={tmp_path}", f"--plot-dir={tmp_path}"])
    assert args.epochs == tj.CONFIG["pretrain"]["epochs"]
    monkeypatch.setattr(tj, "load_directory_dataset", None)
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 2


def test_cli_finetune_exit_code(tmp_path, monkeypatch):
    parser = tj.build_parser()
    args = parser.parse_args(["finetune", "--labeled-dir", str(tmp_path), "--encoder", "enc.pt"])
    assert args.epochs == tj.CONFIG["finetune"]["epochs"]
    monkeypatch.setattr(tj, "load_directory_dataset", None)
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 3


def test_cli_evaluate_exit_code(tmp_path, monkeypatch):
    parser = tj.build_parser()
    args = parser.parse_args(["evaluate", "--labeled-dir", str(tmp_path), "--encoder", "enc.pt"])
    assert args.epochs == tj.CONFIG["evaluate"]["epochs"]
    monkeypatch.setattr(tj, "load_directory_dataset", None)
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 3
