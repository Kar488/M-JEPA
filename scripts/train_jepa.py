import argparse
import os
import sys

import yaml
import numpy as np
import torch

# Allow running as a script without installing the package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataset import GraphData, GraphDataset
from models.base import EncoderBase
from models.ema import EMA
from models.predictor import MLPPredictor
from training.unsupervised import train_jepa


def make_synthetic_dataset(num_graphs: int, num_nodes: int, feat_dim: int) -> GraphDataset:
    graphs = []
    for _ in range(num_graphs):
        x = np.random.randn(num_nodes, feat_dim).astype(np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 1), dtype=np.float32)
        graphs.append(GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return GraphDataset(graphs)


class ToyEncoder(EncoderBase):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(input_dim, hidden_dim)

    def encode_graph(self, g: GraphData, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
        h = torch.relu(self.lin(x))
        return h.mean(dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JEPA on a synthetic dataset")
    p.add_argument("--config", type=str, help="Path to YAML config with training options", required=False)
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = {
        "num_graphs": 8,
        "num_nodes": 5,
        "feat_dim": 4,
        "hidden_dim": 64,
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "ema_decay": 0.99,
    }
    if path:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(getattr(args, "config", None))

    ds = make_synthetic_dataset(cfg["num_graphs"], cfg["num_nodes"], cfg["feat_dim"])
    enc = ToyEncoder(input_dim=cfg["feat_dim"], hidden_dim=cfg["hidden_dim"])
    ema_enc = ToyEncoder(input_dim=cfg["feat_dim"], hidden_dim=cfg["hidden_dim"])
    ema = EMA(enc, decay=cfg["ema_decay"])
    predictor = MLPPredictor(embed_dim=cfg["hidden_dim"], hidden_dim=cfg["hidden_dim"] * 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = train_jepa(
        dataset=ds,
        encoder=enc,
        ema_encoder=ema_enc,
        predictor=predictor,
        ema=ema,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        device=device,
        devices=1,
    )
    if losses:
        print(f"final_loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
