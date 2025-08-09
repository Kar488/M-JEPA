from __future__ import annotations

"""Simple pretraining entry point using Parquet graph datasets.

This script demonstrates how to build :class:`torch_geometric.loader.DataLoader`
objects from preprocessed graph data stored in Parquet files.  The actual
training logic is intentionally omitted – in real usage you would pass the
returned loaders to your model's training loop.
"""

import argparse

from data.parquet_loader import load_dataloaders
from utils.logging import maybe_init_wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain a model on Parquet graphs")
    p.add_argument("--parquet-root", required=True, help="Directory with train/val/test parquet files")
    p.add_argument("--batch-size", type=int, default=32, help="Graphs per batch")
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="m-jepa", help="W&B project name")
    p.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional list of W&B tags",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, val_loader, test_loader = load_dataloaders(
        args.parquet_root, batch_size=args.batch_size
    )
    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config={"batch_size": args.batch_size, "parquet_root": args.parquet_root},
    )
    if args.use_wandb:
        wb.log(
            {
                "dataset/train_graphs": len(train_loader.dataset),
                "dataset/val_graphs": len(val_loader.dataset),
                "dataset/test_graphs": len(test_loader.dataset),
            }
        )
        wb.finish()


if __name__ == "__main__":  # pragma: no cover
    main()
