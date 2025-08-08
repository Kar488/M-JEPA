#Demo + tiny grid (local CPU):
python main.py --mode demo

#Grid on your GPU farm with YAML:
python main.py --mode grid --sweep sweeps/zinc_small.yaml

#Grid sweep from YAML:
python main.py --mode grid --sweep sweeps/zinc_small.yaml


#Grid on your GPU farm with YAML:
python main.py --mode full --device cuda \
  --method jepa --gnn_type edge_mpnn --add_3d \
  --unlabeled_dir data/ZINC_canonicalized \
  --label_train_dir data/esol_scaffold/train \
  --label_val_dir data/esol_scaffold/val \
  --label_test_dir data/esol_scaffold/test \
  --label_col ESOL --task_type regression \
  --pretrain_epochs 100 --pretrain_bs 256 --pretrain_lr 1e-4 \
  --mask_ratio 0.15 --contiguous \
  --finetune_epochs 50 --finetune_bs 64 --finetune_lr 5e-3 \
  --use_wandb --wandb_project m-jepa --ckpt_dir outputs/checkpoints

#Baselines (full mode) — train, embed, evaluate
python main.py --mode full --device cuda --method molclr \
  --baseline_cfg adapters/config.yaml \
  --baseline_unlabeled_file data/zinc_pubchem_train.csv \
  --label_train_dir data/esol_scaffold/train \
  --label_val_dir   data/esol_scaffold/val \
  --label_test_dir  data/esol_scaffold/test \
  --label_col ESOL --task_type regression

#Grid (JEPA + contrastive + baselines) + visuals
python main.py --mode grid --sweep sweeps/zinc_small.yaml

# Example post-processing usage
from experiments.reporting import plot_topn_bar, pivot_heatmap, save_ranked_csv
import pandas as pd
df = pd.read_csv("outputs/grid_results.csv")
save_ranked_csv(df, metric="roc_auc_mean", out_csv="outputs/top50.csv", top_n=50)
plot_topn_bar(df[df["method"]=="jepa"], metric="roc_auc_mean", top_n=15, out_path="outputs/top15_jepa.png",
              title="Top‑15 JEPA configs")
pivot_heatmap(df[df["method"]=="jepa"], metric="roc_auc_mean", x="mask_ratio", y="gnn_type",
              out_path="outputs/heatmap_jepa_mask_gnn.png", title="JEPA ROC‑AUC vs mask×GNN")

#Create scaffold splits from a single file:
python scripts/make_scaffold_splits.py \
  --input data/tox21.csv \
  --out_dir data/tox21_scaffold \
  --smiles_col smiles --format parquet

  # (optional) activate your env first
pytest         # run everything
pytest -m "not slow"   # skip the tiny grid if you want it even faster