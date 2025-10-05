"""Case study using real Tox21 toxicity labels.

This module demonstrates how JEPA embeddings can prioritise molecules by
ranking predictions on the Tox21 dataset. A small encoder is pretrained on
unlabelled molecules, a classification head is fitted on a chosen toxicity task
and the most toxic predictions are compared against a random exclusion
baseline.
"""

from __future__ import annotations

import importlib
import logging
import os
import random
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_RDKIT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_RDKIT = False

try:  # pragma: no cover - optional dependency
    from data.scaffold_split import scaffold_split_indices
except Exception:  # pragma: no cover - fallback when RDKit is absent
    scaffold_split_indices = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from scripts.bench import BenchmarkRule, resolve_metric_threshold
except Exception:  # pragma: no cover - fallback when scripts package unavailable
    BenchmarkRule = None  # type: ignore[assignment]
    resolve_metric_threshold = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from utils.checkpoint import load_state_dict_forgiving, safe_load_checkpoint
except Exception:  # pragma: no cover - fallback for environments without checkpoint helpers
    load_state_dict_forgiving = None  # type: ignore[assignment]
    safe_load_checkpoint = None  # type: ignore[assignment]

from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import stratified_split, train_linear_head
from training.unsupervised import train_jepa
from utils.seed import set_seed
from utils.metrics import expected_calibration_error

import inspect

if TYPE_CHECKING:  # pragma: no cover - typing only
    from data.mdataset import GraphDataset as GraphDatasetT

logger = logging.getLogger(__name__)

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


HIGHER_IS_BETTER = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "acc",
    "ap",
}

_METRIC_ALIASES = {
    "auc": "roc_auc",
    "auroc": "roc_auc",
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
    "ap": "pr_auc",
}


@dataclass
class EvaluationResult:
    """Summary of a single evaluation pass for the Tox21 case study."""

    name: str
    encoder_source: str
    mean_true: float
    mean_random: float
    mean_pred: float
    baseline_means: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_metric: Optional[str] = None
    benchmark_threshold: Optional[float] = None
    met_benchmark: Optional[bool] = None
    manifest_path: Optional[str] = None


@dataclass
class CaseStudyResult:
    """Container returned by :func:`run_tox21_case_study`."""

    evaluations: List[EvaluationResult]
    threshold_rule: Optional["BenchmarkRule"] = None


def _canonical_metric_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return _METRIC_ALIASES.get(str(name).lower(), str(name).lower())


def _metric_is_higher_better(name: Optional[str]) -> bool:
    canonical = _canonical_metric_name(name)
    return canonical in HIGHER_IS_BETTER


def _compute_met_benchmark(metric_name: Optional[str], value: Optional[float], threshold: Optional[float]) -> Optional[bool]:
    if metric_name is None or value is None or threshold is None:
        return None
    if np.isnan(value):  # type: ignore[arg-type]
        return None
    if _metric_is_higher_better(metric_name):
        return bool(value >= threshold)
    return bool(value <= threshold)


def _load_manifest_config(manifest_path: Optional[str]) -> Dict[str, Any]:
    if not manifest_path:
        return {}
    try:
        import json

        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        hyper = payload.get("hyperparameters") if isinstance(payload, dict) else None
        return hyper if isinstance(hyper, dict) else {}
    except Exception:
        logger.warning("Failed to read encoder manifest from %s", manifest_path, exc_info=True)
        return {}


def _extract_state_config(state: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    cfg = state.get("encoder_cfg")
    return cfg if isinstance(cfg, dict) else {}


def _resolve_threshold_rule(dataset_name: Optional[str], task_name: Optional[str]) -> Optional["BenchmarkRule"]:
    if resolve_metric_threshold is None or dataset_name is None:
        return None
    try:
        return resolve_metric_threshold(dataset_name, task_name)
    except KeyError:
        return None


def _predict_logits_probs_in_chunks(
    dataset,
    indices: List[int],
    encoder,
    head,
    device: str,
    edge_dim: int,
    batch_size: int = 256,
):
    encoder.eval()
    head.eval()
    device_t = torch.device(device) if not isinstance(device, torch.device) else device
    logits_list: List[torch.Tensor] = []
    probs_list: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            if not chunk:
                continue
            graph_embs: List[torch.Tensor] = []
            for idx in chunk:
                graph = dataset.graphs[idx]
                graph = ensure_edge_attr(graph, edge_dim, device=device)
                node_emb = _encode_graph_flex(encoder, graph, device_t)
                node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
                graph_embs.append(node_emb.mean(0, keepdim=True))
            if not graph_embs:
                continue
            batch = torch.cat(graph_embs, dim=0).to(device_t)
            logits = head(batch)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            logits_list.append(logits.detach().cpu())
            probs_list.append(probs.detach().cpu())
    if not logits_list:
        return torch.empty((0, 1)), torch.empty((0, 1))
    return torch.cat(logits_list, dim=0), torch.cat(probs_list, dim=0)


def _evaluate_case_study(
    dataset,
    encoder,
    head,
    all_labels: np.ndarray,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    triage_pct: float,
    calibrate: bool,
    device: str,
    edge_dim: int,
    seed: int,
    baseline_embeddings: Optional[dict[str, str]] = None,
):
    val_idx_arr = np.asarray(val_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int)

    if not isinstance(val_idx_arr, np.ndarray):
        val_idx_arr = np.asarray(list(val_idx_arr), dtype=int)
    if not isinstance(test_idx_arr, np.ndarray):
        test_idx_arr = np.asarray(list(test_idx_arr), dtype=int)

    val_indices = val_idx_arr.reshape(-1).tolist()
    test_indices = test_idx_arr.reshape(-1).tolist()

    val_logits, val_probs = _predict_logits_probs_in_chunks(
        dataset, val_indices, encoder, head, device, edge_dim
    )
    test_logits, test_probs = _predict_logits_probs_in_chunks(
        dataset, test_indices, encoder, head, device, edge_dim
    )

    val_logits_np = val_logits.cpu().numpy().reshape(-1, 1) if val_logits.numel() else np.zeros((0, 1))
    test_logits_np = test_logits.cpu().numpy().reshape(-1, 1) if test_logits.numel() else np.zeros((0, 1))

    val_probs_np = val_probs.cpu().numpy()
    test_probs_np = test_probs.cpu().numpy()
    if val_probs_np.ndim > 1:
        val_probs_np = val_probs_np[:, 0]
    if test_probs_np.ndim > 1:
        test_probs_np = test_probs_np[:, 0]

    calibrated_probs = test_probs_np
    if calibrate:
        try:
            val_y = all_labels[val_idx_arr].astype(float)
            mask = (~np.isnan(val_y)) & np.isfinite(val_logits_np[:, 0])
            yv = val_y[mask].astype(int)
            Xv = np.nan_to_num(val_logits_np[mask], nan=0.0, posinf=1e6, neginf=-1e6)
            if yv.size > 1 and np.unique(yv).size > 1:
                platt = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
                platt.fit(Xv, yv)
                Xt = np.nan_to_num(test_logits_np, nan=0.0, posinf=1e6, neginf=-1e6)
                calibrated_probs = platt.predict_proba(Xt)[:, 1]
        except Exception as exc:  # pragma: no cover - calibration optional
            logger.warning("Calibration skipped due to error: %s", exc)
            calibrated_probs = test_probs_np

    calibrated_probs = np.asarray(calibrated_probs).reshape(-1)
    if calibrated_probs.size != test_idx_arr.size:
        calibrated_probs = np.resize(calibrated_probs, test_idx_arr.size)

    k = max(1, int(triage_pct * test_idx_arr.size))
    k = min(k, test_idx_arr.size) if test_idx_arr.size else 0

    order = np.argsort(-calibrated_probs)[:k] if k > 0 else np.array([], dtype=int)
    mask_pred = np.ones(test_idx_arr.shape[0], dtype=bool)
    if order.size > 0:
        mask_pred[order] = False
    remaining_pred = test_idx_arr[mask_pred]
    mean_pred = float(np.mean(all_labels[remaining_pred])) if remaining_pred.size else 0.0

    rng = np.random.default_rng(seed)
    rand_choice = rng.choice(test_idx_arr, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    mask_rand = np.ones(test_idx_arr.shape[0], dtype=bool)
    if rand_choice.size > 0:
        mask_rand[np.isin(test_idx_arr, rand_choice)] = False
    remaining_rand = test_idx_arr[mask_rand]
    mean_rand = float(np.mean(all_labels[remaining_rand])) if remaining_rand.size else 0.0

    mean_true = float(np.mean(all_labels[test_idx_arr])) if test_idx_arr.size else 0.0

    metrics: Dict[str, float] = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
    }

    y_true = all_labels[test_idx_arr].astype(float)
    mask_valid = ~np.isnan(y_true)
    y_true_m = y_true[mask_valid]
    y_pred_m = calibrated_probs[mask_valid]

    if y_true_m.size > 0 and np.unique(y_true_m).size > 1:
        yy = y_true_m.astype(int)
        pp = np.nan_to_num(y_pred_m, nan=0.5, posinf=1.0, neginf=0.0)
        try:
            metrics["roc_auc"] = float(roc_auc_score(yy, pp))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(yy, pp))
        except Exception:
            metrics["pr_auc"] = float("nan")
        try:
            metrics["brier"] = float(brier_score_loss(yy, pp))
        except Exception:
            metrics["brier"] = float("nan")
        try:
            metrics["ece"] = float(expected_calibration_error(yy, pp, n_bins=10))
        except Exception:
            metrics["ece"] = float("nan")
    else:
        logger.warning("TEST split degenerate (one class/empty). Skipping AUC/Brier/ECE.")

    baseline_means: Dict[str, float] = {}
    if baseline_embeddings:
        try:
            from sklearn.linear_model import Ridge
        except Exception:  # pragma: no cover - optional dependency
            Ridge = None  # type: ignore[assignment]
        if Ridge is not None:
            train_idx_arr = np.asarray(train_idx, dtype=int)
            val_idx_arr = np.asarray(val_idx, dtype=int)
            train_val_idx = np.concatenate([train_idx_arr, val_idx_arr])
            train_val_idx = np.unique(train_val_idx)
            y_train_val = all_labels[train_val_idx]
            for name, path in baseline_embeddings.items():
                try:
                    if path.lower().endswith(".npy"):
                        X = np.load(path)
                    else:
                        X = pd.read_csv(path).to_numpy()
                except Exception as exc:
                    logger.warning("Failed to load baseline embeddings %s from %s: %s", name, path, exc)
                    continue
                if X.shape[0] != all_labels.shape[0]:
                    raise ValueError(
                        f"Embeddings for {name} have {X.shape[0]} rows, expected {all_labels.shape[0]}"
                    )
                reg = Ridge(alpha=1.0, random_state=seed).fit(X[train_val_idx], y_train_val)
                pred = reg.predict(X)
                pred_test = pred[test_idx_arr]
                top = np.argsort(-pred_test)[:k]
                mask = np.ones(test_idx_arr.shape[0], dtype=bool)
                if top.size > 0:
                    mask[top] = False
                remain = test_idx_arr[mask]
                baseline_means[name] = float(np.mean(all_labels[remain])) if remain.size else 0.0

    return mean_true, mean_rand, mean_pred, baseline_means, metrics


def _import_graphdataset():
    from data.mdataset import GraphDataset
    return GraphDataset

def run_tox21_case_study(
    csv_path: str,
    task_name: str,
    smiles_col: str = "smiles",
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
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
    triage_pct: float = 0.10,
    calibrate: bool = True,
    use_pos_weight: bool = True,
    device: str = "cpu",
    baseline_embeddings: dict[str, str] | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    bf16: bool = False,
    pretrain_time_budget_mins: int = 0,
    finetune_time_budget_mins: int = 0,
    *,
    dataset_name: str = "tox21",
    encoder_checkpoint: Optional[str] = None,
    encoder_manifest: Optional[str] = None,
    strict_encoder_config: bool = False,
    bf16_head: Optional[bool] = None,
    encoder_source_override: Optional[str] = None,
) -> CaseStudyResult:
    """Run the Tox21 case study and return structured evaluation results."""

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

    GraphDatasetCls = _load_real_graphdataset()
    gnn_type_lower = (gnn_type or "").lower()
    requires_3d = gnn_type_lower in {"schnet3d", "schnet"}

    dataset = GraphDatasetCls.from_smiles_list(
        smiles_list,
        labels=labels_list,
        add_3d=requires_3d,
    )
    if len(dataset) == 0:
        raise ValueError("No valid molecules could be parsed from the dataset.")

    if requires_3d and all(getattr(g, "pos", None) is None for g in dataset.graphs):
        raise ValueError(
            "SchNet-style encoders require 3D coordinates, but none were generated. "
            "Ensure RDKit is installed with 3D conformer support."
        )

    for i, graph in enumerate(dataset.graphs):
        smi = getattr(graph, "smiles", None) or (
            getattr(dataset, "smiles", None)[i] if hasattr(dataset, "smiles") else None
        )
        if not smi:
            continue
        graph.smiles = smi
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is None or getattr(edge_attr, "shape", (0, 0))[1] == 0:
            attach_bond_features_from_smiles(graph, smi)

    all_labels = dataset.labels.astype(float)
    num_total = len(dataset)

    threshold_rule = _resolve_threshold_rule(dataset_name, task_name)

    if scaffold_split_indices and _HAS_RDKIT:
        train_split, val_split, test_split = scaffold_split_indices(
            smiles_list,
            train_frac=train_fraction,
            val_frac=val_fraction,
            seed=seed,
        )
        train_idx = np.asarray(train_split, dtype=int).tolist()
        val_idx = np.asarray(val_split, dtype=int).tolist()
        test_idx = np.asarray(test_split, dtype=int).tolist()
        logger.info(
            "Scaffold split: train=%d val=%d test=%d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )
    else:
        logger.warning("RDKit scaffold split unavailable; using stratified random split.")
        indices = list(range(num_total))
        rand_state = random.getstate()
        np_state = np.random.get_state()
        train_idx, val_idx, test_idx = stratified_split(
            indices,
            dataset.labels,
            train_frac=train_fraction,
            val_frac=val_fraction,
        )
        random.setstate(rand_state)
        np.random.set_state(np_state)

    input_dim = dataset.graphs[0].x.shape[1]
    edge_dim = 0
    try:
        g0 = dataset.graphs[0]
        if getattr(g0, "edge_attr", None) is not None:
            edge_dim = int(g0.edge_attr.shape[1])
    except Exception:
        edge_dim = 0

    manifest_cfg = _load_manifest_config(encoder_manifest)
    state_cfg: Dict[str, Any] = {}
    enc_state: Optional[Dict[str, Any]] = None
    if encoder_checkpoint:
        if safe_load_checkpoint is None:
            raise ImportError("Checkpoint loading utilities are unavailable")
        state, _ = safe_load_checkpoint(
            primary=encoder_checkpoint,
            ckpt_dir=None,
            default_name="encoder.pt",
            map_location=device,
            allow_missing=False,
        )
        state_cfg = _extract_state_config(state)
        enc_state = state.get("encoder", state) if isinstance(state, dict) else state

    final_cfg: Dict[str, Any] = {
        "gnn_type": gnn_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    cli_cfg = final_cfg.copy()
    for source in (state_cfg, manifest_cfg):
        for key in ("gnn_type", "hidden_dim", "num_layers"):
            value = source.get(key)
            if value is None:
                continue
            cli_value = cli_cfg.get(key)
            if (
                strict_encoder_config
                and cli_value is not None
                and str(cli_value) != str(value)
            ):
                raise ValueError(
                    f"Encoder configuration mismatch for {key}: CLI={cli_value} checkpoint={value}"
                )
            if cli_value is not None and str(cli_value) != str(value) and not strict_encoder_config:
                logger.warning(
                    "Overriding encoder %s from %s to %s based on checkpoint metadata",
                    key,
                    cli_value,
                    value,
                )
            final_cfg[key] = value

    final_hidden_dim = int(final_cfg.get("hidden_dim", hidden_dim))
    final_num_layers = int(final_cfg.get("num_layers", num_layers))
    final_gnn_type = str(final_cfg.get("gnn_type", gnn_type))

    gnn_type_lower = final_gnn_type.lower()
    if gnn_type_lower in {
        "gine",
        "gin_edge",
        "gin+edge",
        "edge_mpnn",
        "mpnn_edge",
        "edge",
        "dmpnn",
        "chemprop",
        "attentivefp",
        "attnfp",
    } and edge_dim <= 0:
        edge_dim = 1

    for i, graph in enumerate(dataset.graphs):
        dataset.graphs[i] = ensure_edge_attr(graph, edge_dim, device=device)

    try:
        from models.factory import build_encoder  # type: ignore[import-not-found]
    except Exception:
        from models.encoder import GNNEncoder as _BasicEnc

        def build_encoder(
            gnn_type: str,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            edge_dim: Optional[int] = None,
        ):
            return _BasicEnc(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                gnn_type=gnn_type,
            )

    encoder = build_encoder(
        gnn_type=final_gnn_type,
        input_dim=input_dim,
        hidden_dim=final_hidden_dim,
        num_layers=final_num_layers,
        edge_dim=edge_dim,
    )

    encoder_source = "scratch"
    eval_name = "fine_tuned"
    if encoder_checkpoint:
        encoder_source = "checkpoint"
        eval_name = "frozen"
        if enc_state:
            if load_state_dict_forgiving is not None:
                load_state_dict_forgiving(encoder, enc_state)
            else:
                encoder.load_state_dict(enc_state, strict=False)
        else:
            logger.warning(
                "Encoder checkpoint %s contained no weights; using random initialisation",
                encoder_checkpoint,
            )
    if encoder_source_override:
        encoder_source = str(encoder_source_override)
        eval_name = str(encoder_source_override)
    else:
        ema_encoder = build_encoder(
            gnn_type=final_gnn_type,
            input_dim=input_dim,
            hidden_dim=final_hidden_dim,
            num_layers=final_num_layers,
            edge_dim=edge_dim,
        )
        ema_helper = EMA(encoder, decay=0.99)
        predictor = MLPPredictor(embed_dim=final_hidden_dim, hidden_dim=final_hidden_dim * 2)

        train_fn = train_jepa
        if contrastive:
            try:
                from training.unsupervised import train_contrastive

                train_fn = train_contrastive
            except Exception:
                logger.warning(
                    "Contrastive pretraining requested but unavailable; falling back to JEPA",
                )

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

    encoder = encoder.to(device)

    bf16_linear = bf16_head if bf16_head is not None else bf16

    train_idx_arr = np.asarray(train_idx, dtype=int)
    val_idx_arr = np.asarray(val_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int)

    extra_args: Dict[str, Any] = {}
    if use_pos_weight and train_idx_arr.size > 0:
        train_labels = all_labels[train_idx_arr]
        mask = ~np.isnan(train_labels)
        n_pos = int(np.nansum(train_labels[mask]))
        n_all = int(mask.sum())
        n_neg = max(0, n_all - n_pos)
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)
            if "pos_weight" in inspect.signature(train_linear_head).parameters:
                extra_args["pos_weight"] = pos_weight
            elif "class_weight" in inspect.signature(train_linear_head).parameters:
                extra_args["class_weight"] = {0: 1.0, 1: float(n_neg / max(1, n_pos))}

    clf_metrics = train_linear_head(
        dataset=dataset,
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
        bf16=bf16_linear,
        time_budget_mins=finetune_time_budget_mins,
        use_scaffold=bool(scaffold_split_indices and _HAS_RDKIT),
        **extra_args,
    )

    head = clf_metrics.get("head")
    if head is None:
        raise RuntimeError("train_linear_head did not return a head module")
    head = head.to(device)
    encoder.eval()
    head.eval()

    mean_true, mean_rand, mean_pred, baseline_means, metrics = _evaluate_case_study(
        dataset=dataset,
        encoder=encoder,
        head=head,
        all_labels=all_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        triage_pct=triage_pct,
        calibrate=calibrate,
        device=device,
        edge_dim=edge_dim,
        seed=seed,
        baseline_embeddings=baseline_embeddings,
    )

    benchmark_metric = threshold_rule.metric if threshold_rule is not None else None
    canonical_metric = _canonical_metric_name(benchmark_metric)
    metric_value = metrics.get(canonical_metric) if canonical_metric else None
    met_benchmark = _compute_met_benchmark(
        benchmark_metric,
        metric_value,
        threshold_rule.threshold if threshold_rule is not None else None,
    )

    evaluation = EvaluationResult(
        name=eval_name,
        encoder_source=encoder_source,
        mean_true=float(mean_true),
        mean_random=float(mean_rand),
        mean_pred=float(mean_pred),
        baseline_means={k: float(v) for k, v in baseline_means.items()},
        metrics={k: float(v) for k, v in metrics.items()},
        benchmark_metric=benchmark_metric,
        benchmark_threshold=(
            float(threshold_rule.threshold) if threshold_rule is not None else None
        ),
        met_benchmark=met_benchmark,
        manifest_path=encoder_manifest if encoder_checkpoint else None,
    )

    return CaseStudyResult(evaluations=[evaluation], threshold_rule=threshold_rule)


if __name__ == "__main__":
    csv = "samples/tox21_mini.csv"
    if os.path.exists(csv):
        result = run_tox21_case_study(
            csv_path=csv,
            task_name="NR-AR",
            pretrain_epochs=1,
            finetune_epochs=1,
            triage_pct=0.10,
        )
        primary = result.evaluations[0]
        logger.info("Mean true toxicity: %s", primary.mean_true)
        logger.info("Mean toxicity after random exclusion: %s", primary.mean_random)
        logger.info("Mean toxicity after model exclusion: %s", primary.mean_pred)
        for name, val in primary.baseline_means.items():
            logger.info("Mean toxicity after %s exclusion: %s", name, val)
    else:
        logger.error("Tox21 sample CSV not found: %s", csv)

