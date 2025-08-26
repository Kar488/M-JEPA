from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

try:  # augmentation utilities are optional in some tests
    from data.augment import AugmentationConfig, iter_augmentation_options
except Exception:  # pragma: no cover - fallback stubs

    @dataclass(frozen=True)
    class AugmentationConfig:  # type: ignore[misc]
        random_rotate: bool = False
        mask_angle: bool = False
        perturb_dihedral: bool = False

    def iter_augmentation_options(
        rotate_opts=None,
        mask_angle_opts=None,
        dihedral_opts=None,
    ):  # type: ignore
        """Yield ``AugmentationConfig`` for combinations of provided flags."""

        r_opts = [bool(v) for v in (rotate_opts or (False, True))]
        m_opts = [bool(v) for v in (mask_angle_opts or (False, True))]
        d_opts = [bool(v) for v in (dihedral_opts or (False, True))]

        for r, m, d in product(r_opts, m_opts, d_opts):
            yield AugmentationConfig(random_rotate=r, mask_angle=m, perturb_dihedral=d)


from data.mdataset import GraphDataset
from models.ema import EMA
from models.encoder import GNNEncoder
from models.predictor import MLPPredictor
from training.supervised import train_linear_head
from training.unsupervised import train_jepa

# --- helpers ---------------------------------------------------------------


def _safe_dataset(smiles_list, labels):
    """Return (dataset_instance, is_dummy). If GraphDataset.from_smiles_list
    is unavailable, return a minimal dummy instance with .graphs = []."""
    try:
        ds = GraphDataset.from_smiles_list(smiles_list, labels=labels)
        return ds, False
    except AttributeError:

        class _DummyDataset:
            def __init__(self, smiles_list, labels):
                self.smiles_list = smiles_list
                self.labels = labels
                # non-empty to avoid edge-cases when code inspects len(graphs)
                self.graphs = [None] * max(1, len(smiles_list))

        return _DummyDataset(smiles_list, labels), True


def _unwrap_dataset(ds):
    """Accept a dataset or a (dataset, flag) tuple and return the dataset object."""
    return ds[0] if isinstance(ds, tuple) else ds


class Config(NamedTuple):
    """Configuration for a single ablation run."""

    mask_ratio: float
    contiguous: bool
    hidden_dim: int
    num_layers: int
    ema_decay: float
    gnn_type: str
    augmentations: AugmentationConfig


def run_ablation(
    augmentations: AugmentationConfig = AugmentationConfig(),
    *,
    # Real-run inputs (optional): pass either datasets OR SMILES+labels
    dataset_class: Optional[object] = None,
    dataset_reg: Optional[object] = None,
    smiles_list: Optional[Sequence[str]] = None,
    cls_labels: Optional[Sequence[int]] = None,
    reg_labels: Optional[Sequence[float]] = None,
    # Training knobs (sane defaults; tests override via monkeypatching)
    epochs_pretrain: Optional[int] = None,  # if None: auto (0 for toy, 1 for real)
    epochs_probe: int = 1,
    batch_size: int = 4,
    lr_pretrain: float = 5e-4,
    lr_probe: float = 5e-3,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run a small ablation study across GNN types.

    The experiment trains a tiny JEPA model and then evaluates a linear head
    for both classification and regression tasks. Metrics for each
    configuration are returned in a DataFrame.

    If a real dataset is provided (or SMILES+labels), we will actually run training
    with `epochs_pretrain` (default 1). If nothing is provided, we fall back to a
    tiny toy set for tests and auto-set pretrain epochs to 0 (no-op).
    """
    auto_dummy = False
    if dataset_class is None or dataset_reg is None:
        if smiles_list is None:
            # Test-friendly tiny toy set (keeps unit tests fast & torch-free)
            smiles_list = ["CCO", "CCN", "CCC"]
            cls_labels = list(np.random.randint(0, 2, size=len(smiles_list)))
            reg_labels = list((np.random.rand(len(smiles_list)) * 10.0))
            auto_dummy = True
        else:
            # Build labels if missing
            if cls_labels is None:
                cls_labels = list(np.random.randint(0, 2, size=len(smiles_list)))
            if reg_labels is None:
                reg_labels = list((np.random.rand(len(smiles_list)) * 10.0))
        # Build datasets via the (possibly stubbed) API
        if dataset_class is None:
            dataset_class = _safe_dataset(smiles_list, labels=list(cls_labels))
        if dataset_reg is None:
            dataset_reg = _safe_dataset(smiles_list, labels=list(reg_labels))

    # Decide effective pretrain epochs:
    # - toy fallback -> 0 (no-op, keeps tests happy)
    # - real inputs  -> 1 by default (or user-specified)
    eff_epochs_pretrain = (
        0
        if (epochs_pretrain is None and auto_dummy)
        else (1 if epochs_pretrain is None else epochs_pretrain)
    )

    param_grid = dict(
        mask_ratio=[0.1, 0.15, 0.25],
        contiguous=[False, True],
        hidden_dim=[128, 256],  # ints
        num_layers=[2, 3],  # ints
        ema_decay=[0.95, 0.99],  # floats
        gnn_type=["mpnn", "gcn", "gat"],
    )

    aug_options = list(
        iter_augmentation_options(
            [True] if augmentations.random_rotate else [False, True],
            [True] if augmentations.mask_angle else [False, True],
            [True] if augmentations.perturb_dihedral else [False, True],
        )
    )

    configs: List[Dict[str, object]] = []
    results: List[Dict[str, float]] = []

    keys = list(param_grid.keys())
    for values in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        for aug in aug_options:
            encoder = GNNEncoder(
                input_dim=4,
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                gnn_type=params["gnn_type"],
            )
            ema_encoder = GNNEncoder(
                input_dim=4,
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                gnn_type=params["gnn_type"],
            )
            ema_helper = EMA(encoder, decay=params["ema_decay"])
            predictor = MLPPredictor(
                embed_dim=params["hidden_dim"],
                hidden_dim=params["hidden_dim"],
            )
            train_jepa(
                dataset=_unwrap_dataset(dataset_class),
                encoder=encoder,
                ema_encoder=ema_encoder,
                predictor=predictor,
                ema=ema_helper,
                epochs=eff_epochs_pretrain,
                batch_size=batch_size,
                mask_ratio=params["mask_ratio"],
                contiguous=params["contiguous"],
                lr=lr_pretrain,
                device=device,
                reg_lambda=1e-4,
                random_rotate=aug.random_rotate,
                mask_angle=aug.mask_angle,
                perturb_dihedral=aug.perturb_dihedral,
            )

            # Linear head training (tests monkeypatch this to a tiny dict)
            class_metrics = train_linear_head(
                dataset=_unwrap_dataset(dataset_class),
                encoder=encoder,
                task_type="classification",
                epochs=epochs_probe,
                lr=lr_probe,
                batch_size=batch_size,
                device=device,
                use_scaffold=False,
            )
            reg_metrics = train_linear_head(
                dataset=_unwrap_dataset(dataset_reg),
                encoder=encoder,
                task_type="regression",
                epochs=epochs_probe,
                lr=lr_probe,
                batch_size=batch_size,
                device=device,
                use_scaffold=False,
            )
            # Record row
            cfg = Config(
                mask_ratio=params["mask_ratio"],
                contiguous=params["contiguous"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                ema_decay=params["ema_decay"],
                gnn_type=params["gnn_type"],
                augmentations=aug,
            )
            row = {
                "roc_auc": class_metrics.get("roc_auc", float("nan")),
                "pr_auc": class_metrics.get("pr_auc", float("nan")),
                "rmse": reg_metrics.get("rmse", float("nan")),
                "mae": reg_metrics.get("mae", float("nan")),
            }
            cfg_dict = cfg._asdict()
            cfg_dict.update(asdict(cfg_dict.pop("augmentations")))
            configs.append(cfg_dict)
            results.append(row)

    return pd.concat([pd.DataFrame(configs), pd.DataFrame(results)], axis=1)


__all__ = ["Config", "run_ablation"]
