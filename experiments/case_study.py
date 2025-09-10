"""Case study using real Tox21 toxicity labels.

This module demonstrates how JEPA embeddings can prioritise molecules by
ranking predictions on the Tox21 dataset. A small encoder is pretrained on
unlabelled molecules, a classification head is fitted on a chosen toxicity task
and the most toxic predictions are compared against a random exclusion
baseline.
"""

from __future__ import annotations
from typing import Iterable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # type-only import; not evaluated at runtime
    from data.mdataset import GraphDataset as GraphDatasetT

import logging
import os
import random
from typing import Iterable, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.unsupervised import train_jepa
from utils.pooling import global_mean_pool
from utils.seed import set_seed

import inspect
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

import sys
import importlib, types
from pathlib import Path
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.mdataset import GraphDataset as GraphDatasetT

def _load_real_graphdataset():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    mod_name = "data.mdataset"              # the real module name
    file_path = data_dir / "mdataset.py"

    # 1) Ensure 'data' package exists and points at your repo's data/ dir
    if "data" not in sys.modules:
        pkg = types.ModuleType("data")
        pkg.__path__ = [str(data_dir)]
        sys.modules["data"] = pkg
    else:
        # make sure its __path__ points to your repo
        sys.modules["data"].__path__ = [str(data_dir)]

    # 2) Build spec for the correct qualified name, create module, and
    #    register it in sys.modules BEFORE exec_module (needed for dataclasses)
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module.GraphDataset

def _import_graphdataset():
    from data.mdataset import GraphDataset
    return GraphDataset


def run_tox21_case_study(
    csv_path: str,
    task_name: str,
    smiles_col: str = "smiles",
    train_fraction: float = 0.8,# only used if scaffold split not possible
    val_fraction: float = 0.1,# only used if scaffold split not possible
    seed: int = 42,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    num_layers: int = 3,
    gnn_type: str = "mpnn",
    contiguous: bool = False,
    mask_ratio: float = 0.15,
    contrastive: bool = False,
    triage_pct: float = 0.10,       # proportion of TEST to exclude (e.g., 0.10 = 10%)
    calibrate: bool = True,         # fit Platt scaling on VAL logits and apply to TEST
    use_pos_weight: bool = True,    # pass pos_weight for imbalance if trainer supports it
    device: str = "cpu",
    baseline_embeddings: dict[str, str] | None = None,
) -> Tuple[float, float, float, dict[str, float]]:
    """Run a Tox21 ranking experiment using real labels.

    Args:
        csv_path: Path to a CSV file containing Tox21 data with SMILES and
            toxicity columns.
        task_name: Column name of the toxicity task to predict.
        smiles_col: Name of the column containing SMILES strings.
        train_fraction: Fraction of data used for training.
        val_fraction: Fraction of data used for validation.
        seed: Random seed for reproducibility.
        pretrain_epochs: Number of epochs for JEPA pretraining.
        finetune_epochs: Number of epochs for the classification head.
        lr: float = The learning rate.
        triage_pct: Fraction of TEST to exclude (e.g., 0.10 = 10%).
        calibrate: If True, fit Platt scaling on VAL logits and apply to TEST.
        device: Device on which to run computations.
        baseline_embeddings: Optional mapping of baseline name to a file
            containing precomputed embeddings (``.npy`` or ``.csv``) in the
            same order as ``csv_path``.

    Returns:
        ``(mean_true, mean_random_after, mean_jepa_after, baseline_means)``
        where ``baseline_means`` maps each baseline name to its post-exclusion
        mean toxicity.

    Raises:
        FileNotFoundError: If the CSV file cannot be located.
        ValueError: If the required columns are missing.
    """

    logger.info("Running Tox21 case study on %s", csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {csv_path}")
    if task_name not in df.columns:
        raise ValueError(f"Task column '{task_name}' not found in {csv_path}")

    df = df[[smiles_col, task_name]].dropna(subset=[task_name])
    smiles_list = df[smiles_col].astype(str).tolist()
    labels_list = df[task_name].astype(float).tolist()
    logger.debug("Loaded %d molecules", len(smiles_list))

    set_seed(seed)

    GraphDatasetCls  = _load_real_graphdataset()

    def subset_dataset(ds: 'GraphDatasetT', idxs: Iterable[int]) -> 'GraphDatasetT':
        sub_graphs = [ds.graphs[i] for i in idxs]
        sub_labels = ds.labels[idxs] if ds.labels is not None else None
        return GraphDatasetCls(sub_graphs, sub_labels)


    dataset = GraphDatasetCls.from_smiles_list(smiles_list, labels=labels_list)
    if len(dataset) == 0:
        raise ValueError("No valid molecules could be parsed from the dataset.")

    all_labels = dataset.labels.astype(float)
    num_total = len(dataset)
    # -------------------------------
    # Scaffold split (fallback to random)
    # -------------------------------
    def _scaffold(smiles: str) -> Optional[str]:
        if not _HAS_RDKIT:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scf, isomericSmiles=True) if scf is not None else None

    rng = np.random.default_rng(seed)
    if _HAS_RDKIT:
        scaffolds: Dict[str, List[int]] = {}
        for i, s in enumerate(smiles_list):
            key = _scaffold(s) or f"NOSCAF_{i}"
            scaffolds.setdefault(key, []).append(i)
        # Sort scaffold bins by descending size for deterministic fill
        bins = sorted(scaffolds.values(), key=len, reverse=True)
        train_idx, val_idx, test_idx = [], [], []
        n_train = int(0.8 * num_total)     # standard 80/10/10 for case study
        n_val   = int(0.1 * num_total)
        for bin_idx in bins:
            # place this whole scaffold bin into the first split that has room
            if len(train_idx) + len(bin_idx) <= n_train:
                train_idx.extend(bin_idx)
            elif len(val_idx) + len(bin_idx) <= n_val:
                val_idx.extend(bin_idx)
            else:
                test_idx.extend(bin_idx)
        # If any leftovers due to rounding, push into test
        rest = [i for i in range(num_total) if i not in train_idx+val_idx+test_idx]
        test_idx.extend(rest)
        logger.info("Scaffold split: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))
    else:
        logger.warning("RDKit not available; using RANDOM split for case study.")
        idx = np.arange(num_total)
        rng.shuffle(idx)
        n_train = int(0.8 * num_total)
        n_val   = int(0.1 * num_total)
        train_idx = idx[:n_train].tolist()
        val_idx   = idx[n_train:n_train+n_val].tolist()
        test_idx  = idx[n_train+n_val:].tolist()

    input_dim = dataset.graphs[0].x.shape[1]
    
    logger.info(
        "Pretraining for %d epochs then finetuning for %d epochs",
        pretrain_epochs,
        finetune_epochs,
    )
    try:
        from models.factory import build_encoder  # provides 'edge_mpnn' + fallbacks
    except Exception:
        # fallback to basic encoder if factory not present
        from models.encoder import GNNEncoder as _BasicEnc
    # Derive edge_dim (needed by gine/edge_mpnn/dmpnn/attentivefp). Fallback to 1 if absent.
    edge_dim = 0
    try:
        g0 = dataset.graphs[0]
        if getattr(g0, "edge_attr", None) is not None:
            edge_dim = int(g0.edge_attr.shape[1])
    except Exception:
        pass
    if gnn_type.lower() in ("gine","gin_edge","gin+edge","edge_mpnn","mpnn_edge","edge",
                            "dmpnn","chemprop","attentivefp","attnfp") and edge_dim <= 0:
        edge_dim = 1

    encoder = build_encoder(
        gnn_type=gnn_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
    )
    ema_encoder = build_encoder(
        gnn_type=gnn_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
    )
    
    ema_helper = EMA(encoder, decay=0.99)
    predictor = MLPPredictor(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)

    import importlib
    sup = importlib.import_module("training.supervised")
    unsup = importlib.import_module("training.unsupervised")
    
    train_fn = getattr(unsup, "train_contrastive", unsup.train_jepa) if contrastive else unsup.train_jepa
    train_fn(
        dataset=dataset,
        encoder=encoder,
        ema_encoder=ema_encoder,
        predictor=predictor,
        ema=ema_helper,
        epochs=pretrain_epochs,
        batch_size=64,
        mask_ratio=mask_ratio,
        contiguous=contiguous,
        lr=lr,
        device=device,
        reg_lambda=1e-4,
    )


    train_ds = subset_dataset(dataset, train_idx)
    val_ds = subset_dataset(dataset, val_idx)
    _ = subset_dataset(dataset, test_idx)

    combined_ds_graphs = train_ds.graphs + val_ds.graphs
    combined_ds_labels = np.concatenate([train_ds.labels, val_ds.labels])
    combined_ds = GraphDatasetCls(combined_ds_graphs, combined_ds_labels)

    # -------------------------------
    # Train classification head (Tox21 is binary classification)
    # -------------------------------
    # ---- Imbalance handling: compute pos_weight on TRAIN+VAL and pass if supported ----
    trainval_labels = combined_ds.labels
    mask = ~np.isnan(trainval_labels)
    n_all = int(mask.sum())
    n_pos = int(np.nansum(trainval_labels[mask]))
    n_neg = max(0, n_all - n_pos)
    extra_args = {}
    if use_pos_weight and n_pos > 0 and n_neg > 0:
        import torch
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)
        if "pos_weight" in inspect.signature(sup.train_linear_head).parameters:
            extra_args["pos_weight"] = pos_weight
        elif "class_weight" in inspect.signature(sup.train_linear_head).parameters:
            extra_args["class_weight"] = {0: 1.0, 1: float(n_neg / max(1, n_pos))}

    # Train classification head (Tox21 is binary classification)
    clf_metrics = sup.train_linear_head(
        dataset=combined_ds,
        encoder=encoder,
        task_type="classification",
        epochs=finetune_epochs,
        lr=lr,
        batch_size=32,
        device=device,
        patience=5,
        **extra_args,
        # selection_metric="val_auc",  # uncomment when supported
    )

    head = clf_metrics.get("head")
    encoder.eval()
    head.eval()
    head = head.to(device)
    encoder = encoder.to(device)
    
    def _predict_logits_probs_in_chunks(ds, idxs, encoder, head, device, batch_size=256):
        encoder.eval(); head.eval()
        import torch
        n = len(idxs)
        logits_out = np.empty(n, dtype=np.float32)
        probs_out  = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                slice_idxs = list(idxs[start:end])
                batch_x, batch_adj, batch_ptr, _ = ds.get_batch(slice_idxs)
                batch_x   = batch_x.to(device)
                batch_adj = batch_adj.to(device)
                batch_ptr = batch_ptr.to(device).long()
                node_emb  = encoder(batch_x, batch_adj)
                graph_emb = global_mean_pool(node_emb, batch_ptr)
                logits    = head(graph_emb).squeeze(1)
                logits_out[start:end] = logits.detach().cpu().numpy()
                probs_out[start:end]  = torch.sigmoid(logits).detach().cpu().numpy()
        return logits_out, probs_out

    # Predict on VAL (for calibration) and TEST
    import numpy as _np
    val_idx_arr  = np.asarray(val_idx)
    test_idx_arr = np.asarray(test_idx)
    val_logits,  val_probs  = _predict_logits_probs_in_chunks(dataset, val_idx_arr,  encoder, head, device, batch_size=256)
    test_logits, test_probs = _predict_logits_probs_in_chunks(dataset, test_idx_arr, encoder, head, device, batch_size=256)

    
    # ---- Optional VAL calibration (Platt scaling) → apply to TEST ----
    calibrated_probs = test_probs
    if calibrate:
        try:
            val_y = all_labels[val_idx_arr].astype(float)
            m = ~np.isnan(val_y)
            if m.sum() > 1 and np.unique(val_y[m]).size > 1:
                platt = LogisticRegression(solver="lbfgs", max_iter=1000)
                platt.fit(val_logits[m].reshape(-1, 1), val_y[m].astype(int))
                calibrated_probs = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
        except Exception as _e:
            logger.warning("Calibration skipped due to error: %s", _e)

    # --- Triage within TEST only (percentage, using calibrated probs if available) ---
    test_prob_rank = calibrated_probs  # use test_probs if you later disable calibration entirely
    k = max(1, int(triage_pct * test_idx_arr.size))
    k = min(k, test_idx_arr.size)

    top_k = np.argsort(-test_prob_rank)[:k]
    exclude_pred = test_idx_arr[top_k]
    remaining_pred = [i for i in test_idx_arr if i not in exclude_pred]
    mean_pred = float(np.mean(all_labels[remaining_pred])) if remaining_pred else 0.0

    # Random baseline triage (within TEST)
    rng = np.random.default_rng(seed)
    exclude_rand = rng.choice(test_idx_arr, size=k, replace=False)
    remaining_rand = [i for i in test_idx_arr if i not in exclude_rand]
    mean_rand = float(np.mean(all_labels[remaining_rand])) if remaining_rand else 0.0

    # Reference: TEST mean
    mean_true = float(np.mean(all_labels[test_idx_arr]))

    # Classification metrics on TEST: ROC-AUC, PR-AUC, Brier, ECE

    # Mask NaNs and guard degenerate splits
    y_true = all_labels[test_idx_arr].astype(float)
    y_pred = calibrated_probs
    m = ~np.isnan(y_true)
    y_true_m = y_true[m]
    y_pred_m = y_pred[m]
    if y_true_m.size > 0 and np.unique(y_true_m).size > 1:
        auc   = float(roc_auc_score(y_true_m.astype(int), y_pred_m))
        ap    = float(average_precision_score(y_true_m.astype(int), y_pred_m))
        brier = float(brier_score_loss(y_true_m.astype(int), y_pred_m))

        # compare mean predicted probability vs. empirical positive rate per bin
        def _ece(y, p, n_bins=10):
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            idxs = np.digitize(p, bins) - 1
            ece = 0.0
            N = len(p)
            for b in range(n_bins):
                mask = idxs == b
                if not np.any(mask):
                    continue
                conf = p[mask].mean()     # average predicted prob in bin
                freq = y[mask].mean()     # empirical positive rate in bin
                ece += (np.sum(mask) / N) * abs(freq - conf)
            return float(ece)
        ece = _ece(y_true_m.astype(int), y_pred_m, n_bins=10)
        logger.info(
            "Tox21 %s TEST metrics — AUC=%.4f, PR-AUC=%.4f, Brier=%.4f, ECE=%.4f",
            task_name, auc, ap, brier, ece
        )
    else:
        logger.warning("TEST split degenerate (one class/empty). Skipping AUC/Brier/ECE.")

    baseline_means: dict[str, float] = {}
    if baseline_embeddings:
        from sklearn.linear_model import Ridge

        train_val_idx = train_idx + val_idx
        y_train_val = all_labels[train_val_idx]

        for name, path in baseline_embeddings.items():
            logger.debug("Evaluating baseline %s from %s", name, path)
            if path.lower().endswith(".npy"):
                X = np.load(path)
            else:
                X = pd.read_csv(path).to_numpy()
            if X.shape[0] != num_total:
                raise ValueError(
                    f"Embeddings for {name} have {X.shape[0]} rows, expected {num_total}"
                )
            reg = Ridge(alpha=1.0, random_state=seed).fit(X[train_val_idx], y_train_val)
            pred = reg.predict(X)
            pred_test = pred[test_idx_arr]
            top = np.argsort(-pred_test)[:k]
            exclude = test_idx_arr[top]
            remain = [i for i in test_idx_arr if i not in exclude]
            baseline_means[name] = float(np.mean(all_labels[remain])) if len(remain) else 0.0

    return mean_true, mean_rand, mean_pred, baseline_means


if __name__ == "__main__":
    csv = "samples/tox21_mini.csv"
    if os.path.exists(csv):
        true_mean, rand_mean, pred_mean, baseline_means = run_tox21_case_study(
            csv_path=csv,
            task_name="NR-AR",
            pretrain_epochs=1,
            finetune_epochs=1,
            triage_pct=0.10,
        )
        logger.info("Mean true toxicity: %s", true_mean)
        logger.info("Mean toxicity after random exclusion: %s", rand_mean)
        logger.info("Mean toxicity after JEPA exclusion: %s", pred_mean)
        for name, val in baseline_means.items():
            logger.info("Mean toxicity after %s exclusion: %s", name, val)
    else:
        logger.error("Tox21 sample CSV not found: %s", csv)

