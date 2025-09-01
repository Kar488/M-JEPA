from __future__ import annotations
import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from models.ema import EMA
from training.unsupervised import train_jepa, train_contrastive


# Runtime import: needed because we *instantiate* GraphDataset
from data.mdataset import GraphDataset, GraphData 

# Type-only import: avoids circular import / “variable in type expr”
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.mdataset import GraphData
    from torch import Tensor
    

class ConstantEncoder(torch.nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 4) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.constant_(self.fc.weight, 1.0)

    def forward(self, g: "GraphData") -> Tensor:
        x = torch.from_numpy(g.x).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = x.mean(dim=0, keepdim=True)
        return self.fc(h)


def _make_graph(n_nodes: int) -> GraphData:
    x = np.ones((n_nodes, 2), dtype=np.float32)
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return GraphData(x=x, edge_index=edge_index)


def _make_dataset() -> GraphDataset:
    graphs = [_make_graph(3), _make_graph(4)]
    return GraphDataset(graphs)


def test_jepa_vs_contrastive():
    dataset = _make_dataset()

    # --- JEPA ---
    jepa_encoder = ConstantEncoder()
    ema_encoder = ConstantEncoder()
    ema = EMA(jepa_encoder)
    ema.copy_to(ema_encoder)
    predictor = torch.nn.Identity()
    jepa_losses = train_jepa(
        dataset=dataset,
        encoder=jepa_encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema,
        epochs=2,
        batch_size=2,
        lr=0.0,
        reg_lambda=0.0,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )

    # --- Contrastive baseline ---
    contrastive_encoder = ConstantEncoder()
    contrastive_losses = train_contrastive(
        dataset=dataset,
        encoder=contrastive_encoder,
        projection_dim=4,
        epochs=2,
        batch_size=2,
        lr=0.0,        
        temperature=0.1,
        device="cpu",
        use_scheduler=False,
        use_amp=False,
    )

    assert jepa_losses and contrastive_losses
    assert jepa_losses[-1] <= contrastive_losses[-1]


def test_grid_search_best_config(tmp_path, monkeypatch):
    """Ensure that cmd_grid_search writes the best config to JSON."""
    import argparse
    import json
    import pandas as pd
    from scripts import train_jepa

    # Fake run_grid_search that returns a one‑row DataFrame
    def fake_run_grid_search(**kwargs):
        return pd.DataFrame([
            {
                'mask_ratio': 0.1,
                'contiguous': False,
                'hidden_dim': 64,
                'num_layers': 2,
                'gnn_type': 'mpnn',
                'ema_decay': 0.99,
                'lr': 1e-4,
                'temperature': 0.1,
                'pretrain_batch_size': 32,
                'finetune_batch_size': 16,
                'pretrain_epochs': 1,
                'finetune_epochs': 1,
            }
        ])

    # Monkeypatch run_grid_search in the train_jepa module
    monkeypatch.setattr(train_jepa, 'run_grid_search', fake_run_grid_search)
    # Build dummy args namespace.  Many fields are unused because the fake
    # run_grid_search ignores them; they are included for completeness.
    args = argparse.Namespace(
        dataset_dir=None,
        unlabeled_dir=None,
        labeled_dir=None,
        label_col='label',
        task_type='classification',
        methods=['jepa'],
        mask_ratios=[0.1],
        contiguities=[0],
        hidden_dims=[64],
        num_layers_list=[2],
        gnn_types=['mpnn'],
        ema_decays=[0.99],
        add_3d_options=[0],
        aug_rotate_options=[0],
        aug_mask_angle_options=[0],
        aug_dihedral_options=[0],
        pretrain_batch_sizes=[32],
        finetune_batch_sizes=[16],
        pretrain_epochs_options=[1],
        finetune_epochs_options=[1],
        learning_rates=[1e-4],        
        temperatures=[0.1],
        seeds=[42],
        device='cpu',
        out_csv=None,
        ckpt_dir=str(tmp_path),
        ckpt_every=1,
        use_scheduler=False,
        warmup_steps=1000,
        use_wandb=False,
        wandb_project='test',
        wandb_tags=[],
        best_config_out=str(tmp_path / 'best_config.json'),
        force_refresh=True, 
    )
    # Invoke grid search
    train_jepa.cmd_grid_search(args)
    # Check that the JSON file is created and contains the expected keys
    with open(args.best_config_out, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    assert conf.get('hidden_dim') == 64
    assert conf.get('gnn_type') == 'mpnn'
