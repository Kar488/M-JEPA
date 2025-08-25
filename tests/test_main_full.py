import argparse
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from data.mdataset import GraphDataset, GraphData
import main


def _make_graph(n_nodes: int) -> GraphData:
    x = np.ones((n_nodes, 2), dtype=np.float32)
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)


def _make_dataset(n: int, labeled: bool) -> GraphDataset:
    graphs = [_make_graph(3 + i) for i in range(n)]
    labels = np.arange(n) % 2 if labeled else None
    return GraphDataset(graphs, labels=labels)


def test_run_full_mode(monkeypatch, tmp_path):
    """Integration-style test for the full training pipeline."""

    unlabeled = _make_dataset(2, labeled=False)
    train = _make_dataset(2, labeled=True)
    val = _make_dataset(1, labeled=True)
    test = _make_dataset(1, labeled=True)

    datasets = {
        "unlabeled": unlabeled,
        "train": train,
        "val": val,
        "test": test,
    }

    def _fake_loader(dirpath, *args, **kwargs):
        return datasets[dirpath]

    monkeypatch.setattr(main, "load_directory_dataset", _fake_loader)
    monkeypatch.setattr(main, "save_checkpoint", lambda *a, **k: None)

    args = argparse.Namespace(
        unlabeled_dir="unlabeled",
        cache_dir=str(tmp_path),
        add_3d=False,
        gnn_type="mpnn",
        hidden_dim=16,
        num_layers=1,
        ema_decay=0.99,
        method="jepa",
        pretrain_epochs=1,
        pretrain_bs=2,
        mask_ratio=0.1,
        contiguous=False,
        pretrain_lr=1e-3,
        device="cpu",
        devices=1,
        use_wandb=False,
        wandb_project="test",
        wandb_tags=[],
        ckpt_dir=str(tmp_path / "ckpt"),
        ckpt_every=5,
        warmup_steps=0,
        aug_rotate=False,
        aug_mask_angle=False,
        aug_dihedral=False,
        label_train_dir="train",
        label_val_dir="val",
        label_test_dir="test",
        label_col="label",
        task_type="classification",
        finetune_epochs=1,
        finetune_lr=1e-3,
        finetune_bs=2,
        val_patience=1,
    )

    main.run_full(args)
