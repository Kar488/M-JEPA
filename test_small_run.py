import pandas as pd
from pathlib import Path

from data.dataset import GraphDataset
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from models.ema import EMA
from training.unsupervised import train_jepa
from training.supervised import train_linear_head

# Point to one of your big Parquet shards
SOURCE = Path("data/ZINC_canonicalized/train-0000.parquet")
TMP = Path("data/tmp_small.parquet")

# 1. Load just a handful of molecules from the source parquet
df = pd.read_parquet(SOURCE).head(50)   # change 50 to whatever is fast for you
df.to_parquet(TMP, index=False)

# 2. Use our existing loader
dataset = GraphDataset.from_parquet(
    filepath=str(TMP),
    smiles_col="smiles",
    cache_dir="cache/tmp_small",
    add_3d_features=False  # turn on if you want to test 3D branch
)

# 3. Build the JEPA model as normal
input_dim = dataset.graphs[0].x.shape[1]
encoder = GNNEncoder(input_dim=input_dim, hidden_dim=64, num_layers=2, gnn_type="mpnn")
ema_encoder = GNNEncoder(input_dim=input_dim, hidden_dim=64, num_layers=2, gnn_type="mpnn")
ema = EMA(encoder, decay=0.99)
predictor = MLPPredictor(embed_dim=64, hidden_dim=128)

# 4. Run a tiny JEPA pretrain
_ = train_jepa(
    dataset=dataset,
    encoder=encoder,
    ema_encoder=ema_encoder,
    predictor=predictor,
    ema=ema,
    epochs=2,           # keep it tiny
    batch_size=8,
    mask_ratio=0.2,
    contiguous=False,
    lr=1e-3,
    device="cpu"
)

# 5. Tiny fine-tune
labels = [0, 1] * (len(dataset.graphs) // 2)  # fake labels just for test
dataset.labels = labels
metrics = train_linear_head(
    dataset=dataset,
    encoder=encoder,
    task_type="classification",
    epochs=2,
    lr=1e-3,
    batch_size=8,
    device="cpu"
)
print("Test metrics:", {k: v for k, v in metrics.items() if k != "head"})
