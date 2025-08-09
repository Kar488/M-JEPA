from __future__ import annotations

"""Simple pretraining entry point using Parquet graph datasets.

This script demonstrates how to build :class:`torch_geometric.loader.DataLoader`
objects from preprocessed graph data stored in Parquet files.  The actual
training logic is intentionally omitted – in real usage you would pass the
returned loaders to your model's training loop.
"""

import argparse

from data.parquet_loader import load_dataloaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain a model on Parquet graphs")
    p.add_argument("--parquet-root", required=True, help="Directory with train/val/test parquet files")
    p.add_argument("--batch-size", type=int, default=32, help="Graphs per batch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, val_loader, test_loader = load_dataloaders(
        args.parquet_root, batch_size=args.batch_size
    )
    # For now we simply print dataset sizes as a minimal smoke test.
    print(
        f"Loaded {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test graphs"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
