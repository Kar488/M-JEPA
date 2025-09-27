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

from utils.graph_ops import _ensure_edge_attr_np_or_torch as ensure_edge_attr
from utils.graph_ops import _encode_graph_flex
try:
    from utils.bond_feats import attach_bond_features_from_smiles
except Exception:  # pragma: no cover - optional dependency
    def attach_bond_features_from_smiles(graph, smiles):
        """Fallback when RDKit bond featurisation is unavailable."""
        return graph

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
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    bf16: bool = False,
    pretrain_time_budget_mins: int = 0,
    finetune_time_budget_mins: int = 0,
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
        num_workers: Number of worker processes for DataLoader construction.
        pin_memory: Whether to enable pinned host memory for CUDA pipelines.
        persistent_workers: Keep data-loader workers alive across epochs.
        prefetch_factor: Batches prefetched per worker when ``num_workers > 0``.
        bf16: Enable bfloat16 mixed precision during training when supported.
        pretrain_time_budget_mins: Optional wall-clock budget for pretraining.
            A value of ``0`` disables the budget.
        finetune_time_budget_mins: Optional wall-clock budget for the linear
            head training phase.  A value of ``0`` disables the budget.
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
    gnn_type_lower = (gnn_type or "").lower()
    requires_3d = gnn_type_lower in {"schnet3d", "schnet"}

    def subset_dataset(ds: 'GraphDatasetT', idxs: Iterable[int]) -> 'GraphDatasetT':
        sub_graphs = [ds.graphs[i] for i in idxs]
        sub_labels = ds.labels[idxs] if ds.labels is not None else None
        return GraphDatasetCls(sub_graphs, sub_labels)


    dataset = GraphDatasetCls.from_smiles_list(
        smiles_list,
        labels=labels_list,
        add_3d=requires_3d,
    )
    if len(dataset) == 0:
        raise ValueError("No valid molecules could be parsed from the dataset.")


    all_labels = dataset.labels.astype(float)
    num_total = len(dataset)

    for i, g in enumerate(dataset.graphs):
        # be extra-safe if dataset has no .smiles or a row is None
        smi = getattr(g, "smiles", None) or (getattr(dataset, "smiles", None)[i] if hasattr(dataset, "smiles") else None)
        if not smi:
            continue  # can’t featurize bonds without SMILES; leave edge_attr as-is
        g.smiles = smi

        # only (re)compute if missing or empty
        ea = getattr(g, "edge_attr", None)
        if ea is None or getattr(ea, "shape", (0, 0))[1] == 0:
            attach_bond_features_from_smiles(g, smi) # sets edge_attr to shape (E, 13)

    if requires_3d and all(getattr(g, "pos", None) is None for g in dataset.graphs):
        raise ValueError(
            "SchNet-style encoders require 3D coordinates, but none were generated. "
            "Ensure RDKit is installed with 3D conformer support."
        )

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
        edge_dim = 1 # safe fallback when graphs have no edge features

    # ---- pre-pad ALL graphs once so pretrain/finetune can't crash ---------------
    for i, g in enumerate(dataset.graphs):
        dataset.graphs[i] = ensure_edge_attr(g, edge_dim, device=device)

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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        bf16=bf16,
        time_budget_mins=pretrain_time_budget_mins,
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
        patience=10,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        bf16=bf16,
        time_budget_mins=finetune_time_budget_mins,
        **extra_args,
        # selection_metric="val_auc",  # uncomment when supported
    )

    head = clf_metrics.get("head")
    encoder.eval()
    head.eval()
    head = head.to(device)
    encoder = encoder.to(device)
    


    def _predict_logits_probs_in_chunks(dataset, idcs, encoder, head, device, batch_size=256, edge_dim=1):
        encoder.eval(); head.eval()
        all_logits, all_probs = [], []
        with torch.no_grad():
            for start in range(0, len(idcs), batch_size):
                chunk = idcs[start:start+batch_size]
                graph_embs = []
                for i in chunk:
                    g = dataset.graphs[i]
                    g = ensure_edge_attr(g, edge_dim, device=device)
                    # single-arg forward: pass the graph object
                    node_emb = _encode_graph_flex(encoder, g, device)   # [N_i, D]
                    # guard against NaNs in node embeddings before pooling
                    node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    graph_embs.append(node_emb.mean(0, keepdim=True))  # mean-pool → [1, D]
                if not graph_embs:
                    continue
                batch = torch.cat(graph_embs, dim=0).to(device)        # [B, D]
                logits = head(batch)                                   # [B, 1] for binary
                # sanitize logits & derived probabilities
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                probs  = torch.sigmoid(logits) if logits.shape[-1] == 1 else torch.softmax(logits, dim=-1)
                probs  = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
                probs  = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                all_logits.append(logits.detach().cpu())
                all_probs.append(probs.detach().cpu())
        return torch.cat(all_logits, dim=0), torch.cat(all_probs, dim=0)


    # Predict on VAL (for calibration) and TEST
    import numpy as _np
    val_idx_arr  = np.asarray(val_idx)
    test_idx_arr = np.asarray(test_idx)
    val_logits,  val_probs  = _predict_logits_probs_in_chunks(dataset, val_idx_arr,  encoder, head, device, batch_size=256, edge_dim=edge_dim)
    test_logits, test_probs = _predict_logits_probs_in_chunks(dataset, test_idx_arr, encoder, head, device, batch_size=256, edge_dim=edge_dim)

    
    # ---- Optional VAL calibration (Platt scaling) → apply to TEST ----
    calibrated_probs = test_probs
    if calibrate:
        try:
            val_y = all_labels[val_idx_arr].astype(float)
            Xv = np.asarray(val_logits).reshape(-1, 1)
            # keep only finite rows and both classes present
            m = (~np.isnan(val_y)) & np.isfinite(Xv[:, 0])
            yv = val_y[m].astype(int)
            Xv = np.nan_to_num(Xv[m], copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
            if yv.size > 1 and np.unique(yv).size > 1:
                platt = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
                platt.fit(Xv, yv)
                Xt = np.asarray(test_logits).reshape(-1, 1)
                Xt = np.nan_to_num(Xt, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
                calibrated_probs = platt.predict_proba(Xt)[:, 1]
        except Exception as _e:
            logger.warning("Calibration skipped due to error: %s", _e)

    # --- Triage within TEST only (percentage, using calibrated probs if available) ---
    test_prob_rank = calibrated_probs  # use test_probs if you later disable calibration entirely
    k = max(1, int(triage_pct * test_idx_arr.size))
    k = min(k, test_idx_arr.size)

    top_k = np.asarray(np.argsort(-test_prob_rank)[:k], dtype=int).reshape(-1)
    exclude_pred = test_idx_arr[top_k]
    mask = np.ones(test_idx_arr.shape[0], dtype=bool)
    mask[top_k] = False
    remaining_pred = test_idx_arr[mask].tolist()
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

        # --- Metrics on TEST (NaN/Inf safe) ---
        yy = y_true_m.astype(int)
        pp = np.asarray(y_pred_m).astype(float)
        keep = (~np.isnan(yy)) & np.isfinite(pp)
        yy, pp = yy[keep], np.nan_to_num(pp[keep], copy=False, nan=0.5, posinf=1.0, neginf=0.0)
        # if not enough signal/classes, return neutral scores instead of crashing
        if yy.size < 2 or np.unique(yy).size < 2:
            auc, ap, brier = 0.5, float('nan'), float('nan')
        else:
            auc   = float(roc_auc_score(yy, pp))
            ap    = float(average_precision_score(yy, pp))
            brier = float(brier_score_loss(yy, pp))
        

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
        
        from utils.metrics import expected_calibration_error

        ece = expected_calibration_error(yy, pp, n_bins=10)
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

