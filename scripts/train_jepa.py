# flake8: noqa

"""End-to-end JEPA training and evaluation pipeline.

This script orchestrates self‑supervised JEPA pretraining, fine‑tuning/evaluation,
benchmarking and optional case study on Tox21.  Rather than duplicating
core logic, it delegates to reusable modules defined in the repository.  All
hyper‑parameters live in `default.yaml` and can be overridden via CLI.

Stages:
    - `pretrain`: run JEPA on unlabelled data, optionally contrastive baseline.
    - `finetune`: train a linear head on labelled data, averaging metrics over multiple seeds.
    - `evaluate`: same as finetune but without saving the head.
    - `benchmark`: compare JEPA vs contrastive encoders on the same dataset, reporting the better.
    - `tox21`: run a real case study on a Tox21 CSV.

If available, grid search and case study helpers from `experiments` are used.

Each major step is logged to Weights & Biases when enabled. Distinct exit
codes are used so that GitHub Actions can determine which stage failed.
"""


from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
# Attempt to import real PyTorch.  Some unit tests only need this module to be
# importable and will skip themselves if ``torch`` is absent.  To keep those
# tests lightweight we provide a tiny stub when the import fails; attempting to
# use it will still raise ``ModuleNotFoundError``.
try:  # pragma: no cover - exercised only when torch is missing
    import torch  # type: ignore
except Exception:  # noqa: BLE001 - broad to catch import errors
    class _MissingTorch:
        def __getattr__(self, name: str) -> None:
            raise ModuleNotFoundError("torch is required for training")

    torch = _MissingTorch()  # type: ignore[assignment]
    sys.modules.setdefault("torch", torch)  # ensure subsequent imports see the stub

import yaml

from utils.dataset import (
    load_dataset,
    load_directory_dataset as _utils_load_directory_dataset,
    load_parquet_dataset,
    _GRAPH_DATASET_IMPORT_ERROR as _UTILS_GRAPH_DATASET_IMPORT_ERROR,
)

try:
    from data.augment import iter_augmentation_options  # type: ignore
except Exception:

    def iter_augmentation_options(rot_flags, ang_flags, dih_flags):
        """Fallback generator producing AugmentationConfig objects from 0/1 flags."""
        rot_flags = tuple(bool(int(x)) for x in rot_flags)
        ang_flags = tuple(bool(int(x)) for x in ang_flags)
        dih_flags = tuple(bool(int(x)) for x in dih_flags)
        import inspect

        params = set()
        try:
            params = set(inspect.signature(AugmentationConfig).parameters.keys()) - {
                "self"
            }
        except Exception:
            pass
        for r in rot_flags:
            for a in ang_flags:
                for d in dih_flags:
                    kwargs = {}
                    if "random_rotate" in params:
                        kwargs["random_rotate"] = r
                    elif "rotate" in params:
                        kwargs["rotate"] = r
                    if "mask_angle" in params:
                        kwargs["mask_angle"] = a
                    if "perturb_dihedral" in params:
                        kwargs["perturb_dihedral"] = d
                    elif "dihedral" in params:
                        kwargs["dihedral"] = d
                    try:
                        yield AugmentationConfig(**kwargs)  # type: ignore
                    except Exception:
                        # fallback duck type
                        from types import SimpleNamespace

                        yield SimpleNamespace(
                            random_rotate=r,
                            rotate=r,
                            mask_angle=a,
                            perturb_dihedral=d,
                            dihedral=d,
                        )


try:
    from data.augment import AugmentationConfig
    from data.mdataset import GraphData, GraphDataset
    _GRAPH_DATASET_IMPORT_ERROR = _UTILS_GRAPH_DATASET_IMPORT_ERROR
except Exception as e:
    GraphData = None  # type: ignore[assignment]
    GraphDataset = None  # type: ignore[assignment]
    _GRAPH_DATASET_IMPORT_ERROR = e

    @dataclass(frozen=True)
    class AugmentationConfig:
        rotate: bool = False
        mask_angle: bool = False
        dihedral: bool = False


        def __init__(self, rotate: bool = False, mask_angle: bool = False,
                     dihedral: bool = False, **kw):
            # map common aliases if sweeps pass them; ignore the rest
            if "random_rotate" in kw:
                rv = kw["random_rotate"]
                rotate = rotate or (bool(int(rv)) if isinstance(rv, (int, str)) else bool(rv))
            if "perturb_dihedral" in kw:
                dv = kw["perturb_dihedral"]
                dihedral = dihedral or (bool(int(dv)) if isinstance(dv, (int, str)) else bool(dv))
            object.__setattr__(self, "rotate", bool(rotate))
            object.__setattr__(self, "mask_angle", bool(mask_angle))
            object.__setattr__(self, "dihedral", bool(dihedral))

        @classmethod
        def from_dict(cls, cfg: Optional[dict] = None) -> "AugmentationConfig":
            cfg = cfg or {}
            return cls(
                rotate=bool(cfg.get("rotate", False)),
                mask_angle=bool(cfg.get("mask_angle", False)),
                dihedral=bool(cfg.get("dihedral", False)),
            )


# Re-export dataset loader with guard for optional dependency
def load_directory_dataset(dirpath: str, **kwargs):  # type: ignore[override]
    if GraphDataset is None:
        raise ImportError(
            "GraphDataset is unavailable. Ensure `data.mdataset.GraphDataset` can be imported."
        ) from _GRAPH_DATASET_IMPORT_ERROR
    return _utils_load_directory_dataset(dirpath, **kwargs)


# Models
try:
    from models.factory import build_encoder  # provides 'edge_mpnn' + fallbacks
except Exception:
    # fallback to basic encoder if factory not present
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


try:
    from models.ema import EMA  # type: ignore[assignment]
    from models.predictor import MLPPredictor  # type: ignore[assignment]
except Exception:
    EMA = None  # type: ignore[assignment]
    MLPPredictor = None  # type: ignore[assignment]

# --- Minimal linear head builder (works for classification & regression) ---
try:
    # If you later add a proper head somewhere, import it here:
    from models.heads import build_linear_head  # type: ignore
except Exception:
    def build_linear_head(*args, **kwargs):  # pragma: no cover - used only without torch
        raise ModuleNotFoundError("torch is required for build_linear_head")


try:
    from training.unsupervised import (  # type: ignore[assignment]
        train_contrastive,
        train_jepa,
    )
except Exception:
    train_jepa = None  # type: ignore[assignment]
    train_contrastive = None  # type: ignore[assignment]

try:
    from training.supervised import train_linear_head  # type: ignore[assignment]
except Exception:
    train_linear_head = None  # type: ignore[assignment]

try:
    from experiments.case_study import run_tox21_case_study  # type: ignore[assignment]
except Exception:
    run_tox21_case_study = None  # type: ignore[assignment]

try:
    from experiments.grid_search import run_grid_search  # type: ignore[assignment]
except Exception:
    run_grid_search = None  # type: ignore[assignment]

try:
    from utils.logging import maybe_init_wandb  # type: ignore[assignment]
except Exception:
    # Provide a stub if W&B logging isn't available
    def maybe_init_wandb(*args, **kwargs):  # type: ignore[assignment]
        class DummyWB:
            def log(self, *a, **k):
                pass

            def finish(self) -> None:
                pass

        return DummyWB()


try:
    from utils.plotting import plot_training_curves  # type: ignore[assignment]
except Exception:

    def plot_training_curves(*args, **kwargs):  # type: ignore[assignment]
        class _DummyFig:
            def savefig(self, *a, **k):
                pass

        return _DummyFig()


try:
    from ..utils.checkpoint  import safe_load_checkpoint as _safe_load_checkpoint        # type: ignore[import-not-found]
    from ..utils.checkpoint  import load_state_dict_forgiving as _load_state_dict_forgiving      # type: ignore[import-not-found]
except ImportError:
    # Fallback: absolute imports when run from repo root with PYTHONPATH set
    from utils.checkpoint import safe_load_checkpoint  as _safe_load_checkpoint        # type: ignore[import-not-found]
    from utils.checkpoint import load_state_dict_forgiving as _load_state_dict_forgiving        # type: ignore[import-not-found]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Mock test support utils
# ---------------------------------------------------------------------------


def _maybe_to(module, device):
    """Call .to(device) if present (tests use dummy encoders without .to)."""
    if hasattr(module, "to"):
        module.to(device)
    return module


def _maybe_labels(ds):
    """Best-effort extraction of labels from various dataset shapes.
    Returns a NumPy array or None if not available."""
    import numpy as _np

    for attr in ("y", "labels", "targets"):
        if hasattr(ds, attr):
            try:
                return _np.asarray(getattr(ds, attr))
            except Exception:
                return None
    return None

def _infer_num_classes(labeled) -> int:
    """Best-effort class count. Falls back to 2 if we can't see labels."""
    # 1) explicit attributes
    for attr in ("num_classes", "n_classes", "classes"):
        if hasattr(labeled, attr):
            try:
                n = int(getattr(labeled, attr))
                if n > 0:
                    return n
            except Exception:
                pass
    # 2) try labels array
    import numpy as np

    y = _maybe_labels(labeled)  # your helper from the previous step
    if y is None:
        return 2
    try:
        y = np.asarray(y)
        if y.size == 0:
            return 2
        if y.ndim > 1:
            y = y[:, 0]
        # robust to non-integer labels
        uniq = np.unique(y[~np.isnan(y)]) if y.dtype.kind in "fc" else np.unique(y)
        if uniq.dtype.kind in "iu":
            return int(uniq.max() + 1)
        return int(len(uniq)) or 2
    except Exception:
        return 2


def _iter_params(m):
    ps = getattr(m, "parameters", None)
    if callable(ps):
        try:
            return list(ps())
        except Exception:
            return []
    return []


def _maybe_state_dict(obj):
    if obj is None:
        return None
    """Return obj.state_dict() if available, else None (for test dummies)."""
    sd = getattr(obj, "state_dict", None)
    if callable(sd):
        try:
            return sd()
        except Exception:
            return None
    return None
def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Load defaults eagerly.  These are used as defaults for CLI arguments.
CONFIG = load_config(Path(__file__).with_name("default.yaml"))
_aug_raw = CONFIG.get("pretrain", {}).get("augmentations", {}) or {}
_aug_raw = {
    # accept either style from YAML
    "rotate": bool(_aug_raw.get("rotate", _aug_raw.get("random_rotate", False))),
    "mask_angle": bool(_aug_raw.get("mask_angle", False)),
    "dihedral": bool(_aug_raw.get("dihedral", _aug_raw.get("perturb_dihedral", False))),
}

# Build DEFAULT_AUG robustly against differing constructor names
try:
    import inspect

    params = set(inspect.signature(AugmentationConfig).parameters.keys()) - {"self"}
except Exception:
    params = set()

_mapped = {}
if "random_rotate" in params:
    _mapped["random_rotate"] = _aug_raw["rotate"]
elif "rotate" in params:
    _mapped["rotate"] = _aug_raw["rotate"]

if "mask_angle" in params:
    _mapped["mask_angle"] = _aug_raw["mask_angle"]

if "perturb_dihedral" in params:
    _mapped["perturb_dihedral"] = _aug_raw["dihedral"]
elif "dihedral" in params:
    _mapped["dihedral"] = _aug_raw["dihedral"]

try:
    # Prefer keyword construction with mapped names
    DEFAULT_AUG = AugmentationConfig(**_mapped)  # type: ignore[arg-type]
except Exception:
    # Last resort — provide a duck-typed object with expected attrs
    from types import SimpleNamespace

    DEFAULT_AUG = SimpleNamespace(
        rotate=_aug_raw["rotate"],
        mask_angle=_aug_raw["mask_angle"],
        dihedral=_aug_raw["dihedral"],
        random_rotate=_aug_raw["rotate"],
        perturb_dihedral=_aug_raw["dihedral"],
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and std for each metric across runs.

    Excludes the key 'head' if present.
    """
    if not metrics_list:
        return {}
    out: Dict[str, float] = {}
    keys = sorted({k for d in metrics_list for k in d.keys() if k != "head"})
    for k in keys:
        vals = np.array([d[k] for d in metrics_list if k in d], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def resolve_device(preferred: str) -> str:
    """Return a valid PyTorch device string."""
    try:
        if (
            preferred
            and preferred != "cpu"
            and getattr(torch, "cuda", None) is not None
            and torch.cuda.is_available()  # type: ignore[union-attr]
        ):
            return preferred
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

# --- make repo root importable in tests/agents ---
# import os, sys
# _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if _REPO_ROOT not in sys.path:
#     sys.path.insert(0, _REPO_ROOT)

# --- command implementations loaded lazily to avoid heavy imports at module load ---
def _load_cmd(name: str):  # pragma: no cover - small helper
    try:
        mod = __import__(f"scripts.commands.{name}", fromlist=[name])
    except ModuleNotFoundError:
        mod = __import__(f"commands.{name}", fromlist=[name])
    return mod


def evaluate_finetuned_head(*a, **k):
    _finetune = _load_cmd("finetune")
    return _finetune.evaluate_finetuned_head(*a, **k)


def _inject_shared(m):
    """Inject shared utilities into a command module."""
    import sys as _sys
    this = _sys.modules[__name__]
    for name in CMD_CONTEXT.__dict__:
        setattr(m, name, getattr(this, name, getattr(CMD_CONTEXT, name)))


def cmd_sweep_run(args: argparse.Namespace) -> None:
    _sweep_run = _load_cmd("sweep_run")
    _inject_shared(_sweep_run)
    _sweep_run.cmd_sweep_run(args)


def cmd_grid_search(args: argparse.Namespace) -> None:
    _inject_shared(_grid_search)
    # Propagate a monkeypatched run_grid_search into the command module.
    # CMD_CONTEXT captures run_grid_search at import time, so tests that
    # patch ``train_jepa.run_grid_search`` would otherwise have no effect.
    # Reassign here to ensure the latest reference is used.
    _grid_search.run_grid_search = run_grid_search
    _grid_search.cmd_grid_search(args)


def cmd_pretrain(args: argparse.Namespace) -> None:
    _inject_shared(_pretrain)
    _pretrain.cmd_pretrain(args)


def cmd_finetune(args: argparse.Namespace) -> None:
    _inject_shared(_finetune)
    _finetune.cmd_finetune(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    # The evaluate command shares the finetune implementation.
    # Calling our wrapper ensures any monkeypatched dependencies are used.
    cmd_finetune(args)


def cmd_benchmark(args: argparse.Namespace) -> None:
    _inject_shared(_benchmark)
    _benchmark.cmd_benchmark(args)


def cmd_tox21(args: argparse.Namespace) -> None:
    _inject_shared(_tox21)
    _tox21.cmd_tox21(args)

# --- W&B helpers (same pattern) ---
try:
    from scripts.wandb_safety import (
        wb_get_or_init as _wb_get_or_init,
        wb_summary_update as _wb_summary_update,
        wb_finish_safely as _wb_finish_safely,
    )
except ModuleNotFoundError:
    from wandb_safety import (
        wb_get_or_init as _wb_get_or_init,
        wb_summary_update as _wb_summary_update,
        wb_finish_safely as _wb_finish_safely,
    )

# ---------------------------------------------------------------------------
# Dependency context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandContext:
    logger: logging.Logger
    load_dataset: Any
    load_directory_dataset: Any
    build_encoder: Any
    MLPPredictor: Any
    EMA: Any
    train_jepa: Any
    train_contrastive: Any
    train_linear_head: Any
    run_tox21_case_study: Any
    run_grid_search: Any
    maybe_init_wandb: Any
    plot_training_curves: Any
    resolve_device: Any
    aggregate_metrics: Any
    CONFIG: Dict[str, Any]
    DEFAULT_AUG: AugmentationConfig
    _maybe_to: Any
    _iter_params: Any
    _safe_load_checkpoint: Any
    _load_state_dict_forgiving: Any
    _maybe_labels: Any
    _infer_num_classes: Any
    _maybe_state_dict: Any
    evaluate_finetuned_head: Any
    iter_augmentation_options: Any
    AugmentationConfig: Any
    build_linear_head: Any


CMD_CONTEXT = CommandContext(
   logger=logger,
    load_dataset=load_dataset,
    load_directory_dataset=load_directory_dataset,
    build_encoder=build_encoder,
    MLPPredictor=MLPPredictor,
    EMA=EMA,
    train_jepa=train_jepa,
    train_contrastive=train_contrastive,
    train_linear_head=train_linear_head,
    run_tox21_case_study=run_tox21_case_study,
    run_grid_search=run_grid_search,
    maybe_init_wandb=maybe_init_wandb,
    plot_training_curves=plot_training_curves,
    resolve_device=resolve_device,
    aggregate_metrics=aggregate_metrics,
    CONFIG=CONFIG,
    DEFAULT_AUG=DEFAULT_AUG,
    _maybe_to=_maybe_to,
    _iter_params=_iter_params,
    _safe_load_checkpoint=_safe_load_checkpoint,
    _load_state_dict_forgiving=_load_state_dict_forgiving,
    _maybe_labels=_maybe_labels,
    _infer_num_classes=_infer_num_classes,
    _maybe_state_dict=_maybe_state_dict,
    evaluate_finetuned_head=evaluate_finetuned_head,
    iter_augmentation_options=iter_augmentation_options,
    AugmentationConfig=AugmentationConfig,
    build_linear_head=build_linear_head,
)

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


@dataclass
class CommonArgDefaults:
    gnn_type: str
    hidden_dim: int
    num_layers: int
    ema_decay: float
    add_3d: bool
    num_workers: int
    cache_dir: Optional[str]
    contiguous: bool
    aug_rotate: bool
    aug_mask_angle: bool
    aug_dihedral: bool
    epochs: int
    batch_size: int
    lr: float
    seeds: Optional[List[int]]
    device: str
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    bf16: bool
    devices: int
    use_wandb: bool
    wandb_project: str
    wandb_tags: List[str]

    @classmethod
    def from_config(cls, section: str) -> "CommonArgDefaults":
        model_cfg = CONFIG.get("model", {})
        sec_cfg = CONFIG.get(section, {})
        wandb_cfg = CONFIG.get("wandb", {})
        return cls(
            gnn_type=model_cfg.get("gnn_type", "mpnn"),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            ema_decay=model_cfg.get("ema_decay", 0.99),
            add_3d=False,
            num_workers=0,
            cache_dir=None,
            contiguous=False,
            aug_rotate=DEFAULT_AUG.rotate,
            aug_mask_angle=DEFAULT_AUG.mask_angle,
            aug_dihedral=DEFAULT_AUG.dihedral,
            epochs=sec_cfg.get("epochs", 1),
            batch_size=sec_cfg.get("batch_size", 32),
            lr=sec_cfg.get("lr", 1e-3),
            seeds=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            bf16=False,
            devices=1,
            use_wandb=False,
            wandb_project=wandb_cfg.get("project", "m-jepa"),
            wandb_tags=wandb_cfg.get("tags", []),
        )


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean (0/1/true/false), got '{v}'")

# Accept --flag, --flag=1, --flag 0 styles
class BoolFlag(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", "?")
        kwargs.setdefault("const", True)
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, _to_bool(values))

def _add_common_args(p: argparse.ArgumentParser, section: str) -> None:
    """Add arguments common to multiple commands."""

    d = CommonArgDefaults.from_config(section)
    p.add_argument("--add-3d", "--add_3d", dest="add_3d", action=BoolFlag, default=d.add_3d, help="Augment with 3D coordinate featurisation")
    p.add_argument("--cache-dir", type=str, default=d.cache_dir, help="Directory to cache processed graphs")
    p.add_argument("--contiguity", "--contiguous", dest="contiguity", action=BoolFlag, default=d.contiguous, help="Use contiguous subgraph masking (JEPA)")
    p.add_argument("--aug-rotate", "--aug_rotate",dest="aug_rotate", action=BoolFlag, default=d.aug_rotate, help="Randomly rotate coordinates during pretraining")
    p.add_argument("--aug-mask-angle", "--aug_mask_angle", dest="aug_mask_angle", action=BoolFlag, default=d.aug_mask_angle, help="Mask bond angles during pretraining")
    p.add_argument("--aug-dihedral", "--aug_dihedral", dest="aug_dihedral", action=BoolFlag, default=d.aug_dihedral, help="Perturb dihedral angles during pretraining")
    p.add_argument("--epochs", type=int, default=d.epochs, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=d.batch_size, help="Batch size")
    p.add_argument("--lr", type=float, default=d.lr, help="Learning rate")
    p.add_argument("--seeds", type=int, nargs="*", default=d.seeds, help="Random seeds for averaging results")
    
    p.add_argument("--prefetch-factor", type=int, default=d.prefetch_factor, help="Dataloader prefetch factor (workers>0 only).")
    p.add_argument("--pin-memory", "--pin_memory", dest="pin_memory", action=BoolFlag, default=d.pin_memory, help="Pin CUDA host memory in DataLoader.")
    p.add_argument("--persistent-workers", "--persistent_workers", dest="persistent_workers", action=BoolFlag, default=d.persistent_workers, help="Keep worker processes alive across epochs (workers>0).")
    p.add_argument("--bf16", action=BoolFlag, default=d.bf16, help="Enable bfloat16 autocast on GPU.")
    p.add_argument("--device", type=str, default=d.device, help="Device")
    p.add_argument("--devices", type=int, default=d.devices, help="Number of GPUs for DDP")
    p.add_argument("--num-workers", type=int, default=d.num_workers, help="Process pool workers for SMILES conversion (0=serial)")
    
    p.add_argument("--use-wandb", "--use_wandb", dest="use_wandb", action=BoolFlag, default=d.use_wandb, help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=d.wandb_project, help="W&B project name")
    p.add_argument("--wandb-tags", nargs="*", default=d.wandb_tags, help="Tags for W&B run")

def _add_model_args(p: argparse.ArgumentParser) -> None:
    md = CONFIG.get("model", {})
    p.add_argument("--gnn-type", "--gnn_type", dest="gnn_type",
                   choices=["gcn","gat","mpnn","edge_mpnn","graphsage","gin", "gine", "dmpnn", "attentivefp", "schnet3d"],
                    default=md.get("gnn_type", "edge_mpnn"),
                    help=(
                        "Backbone GNN. Use 'gine' (GIN+edge) or 'dmpnn' (Chemprop-style directed MPNN) "
                        "for 2D/bond-aware runs; 'schnet3d' for 3D geometry (requires pos); "
                        "'attentivefp' for attention readout over atoms/bonds."
                    ),)
    p.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=md.get("hidden_dim", 128))
    p.add_argument("--num-layers", "--num_layers", dest="num_layers", type=int, default=md.get("num_layers", 2))
    p.add_argument("--mask-ratio", "--mask_ratio", dest="mask_ratio", type=float, default=md.get("mask_ratio", 0.15))
    p.add_argument("--ema-decay", "--ema_decay", dest="ema_decay", type=float, default=md.get("ema_decay", 0.996))
    p.add_argument("--temperature", dest="temperature", type=float, default=md.get("temperature", 0.1))

def build_parser() -> argparse.ArgumentParser:

    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="JEPA training and evaluation pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Pretrain subcommand
    pre = sub.add_parser("pretrain", help="Self‑supervised pretraining")
    pre.add_argument("--unlabeled-dir", required=True, help="Directory of unlabeled graphs (.parquet or .csv)")
    pre.add_argument("--output", type=str, default="encoder.pt", help="Where to save the JEPA encoder checkpoint")
    pre.add_argument("--contrastive", action="store_true", help="Also run a contrastive baseline")
    pre.add_argument("--ckpt-dir", type=str, default="ckpts/pretrain", help="Directory to save pretrain checkpoints")
    pre.add_argument("--resume-ckpt", type=str, default="", help="Resume pretraining from a checkpoint")
    pre.add_argument("--save-every", type=int, default=1, help="Save a pretrain checkpoint every N epochs")
    pre.add_argument("--plot-dir", type=str, default=CONFIG.get("plot_dir", "plots"), help="Directory to save training plots")
    pre.add_argument("--force-tqdm", action="store_true", help="Force-enable tqdm progress bars even when not attached to a TTY")
    pre.add_argument("--sample-unlabeled", type=int, default=0, help="If >0, load at most N graphs from the unlabeled dataset.")
    pre.add_argument("--n-rows-per-file", type=int, default=None, help="If set, limit rows read per file when loading datasets.")
    
    _add_common_args(pre, "pretrain")
    _add_model_args(pre)
    pre.set_defaults(func=cmd_pretrain)

    # Fine‑tune subcommand
    ft = sub.add_parser("finetune", help="Fine‑tune a linear head on labelled data")
    ft.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs (.parquet or .csv)")
    ft.add_argument("--label-col", type=str, default="label", help="Label column name in input files")
    ft.add_argument("--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)")
    ft.add_argument("--ckpt-dir", type=str, default="ckpts/finetune", help="dir to write fine-tune checkpoints")
    ft.add_argument("--resume-ckpt", type=str, default="", help="resume fine-tune from this checkpoint")
    ft.add_argument("--save-every", type=int, default=1, help="save checkpoint every N epochs")
    ft.add_argument("--save-final", action="store_true", help="also save ft_last.pt at the end")
    ft.add_argument("--metric", type=str, default="val_loss", choices=["val_loss", "acc", "auroc","val_rmse"])
    ft.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ft.add_argument("--patience", type=int, default=CONFIG.get("finetune", {}).get("patience", 10), help="Early stopping patience")
    _add_common_args(ft, "finetune")
    _add_model_args(ft)
    ft.set_defaults(func=cmd_finetune)

    # Evaluate subcommand (alias for finetune)
    ev = sub.add_parser(
        "evaluate", help="Evaluate a pretrained encoder via a linear probe"
    )
    ev.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs")
    ev.add_argument("--label-col", type=str, default="label", help="Label column name")
    ev.add_argument(
        "--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)"
    )
    ev.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ev.add_argument("--patience", type=int, default=CONFIG.get("evaluate", {}).get("patience", 10), help="Early stopping patience")
    _add_common_args(ev, "evaluate")
    _add_model_args(ev)
    ev.set_defaults(func=cmd_evaluate)
    # Benchmark subcommand
    bench = sub.add_parser("benchmark", help="Compare JEPA and contrastive encoders on labelled data"); 
    bench.add_argument("--labeled-dir", required=True, help="Directory of labelled graphs"); 
    bench.add_argument("--test-dir", required=False, default=None, help="Optional directory of test graphs for eval-only benchmarking"); 
    bench.add_argument("--label-col", type=str, default="label", help="Label column name"); 
    bench.add_argument("--jepa-encoder", required=True, help="Path to a JEPA encoder checkpoint (.pt)"); 
    bench.add_argument("--contrastive-encoder", required=False, help="Path to a contrastive encoder checkpoint (.pt)"); 
    bench.add_argument("--task-type", choices=["classification", "regression"], default="classification"); 
    bench.add_argument("--patience", type=int, default=CONFIG.get("benchmark", {}).get("patience", 10), help="Early stopping patience"); 
    bench.add_argument("--ft-ckpt", type=str, default="", help="fine-tuned checkpoint (expects encoder and optionally head)"); 
    bench.add_argument("--report-dir", type=str, default="reports", help="where to write JSON/CSV"); 
    bench.add_argument("--report-stem", type=str, default="", help="filename stem; defaults to timestamped benchmark_*")
    _add_common_args(bench, "benchmark")
    _add_model_args(bench)
    bench.set_defaults(func=cmd_benchmark)
    # Tox21 case study
    tox = sub.add_parser("tox21", help="Run the Tox21 case study experiment")
    tox.add_argument("--csv", required=True, help="Path to the Tox21 CSV containing SMILES and labels"); 
    tox.add_argument("--task", required=True, help="Name of the toxicity column to predict"); 
    case_cfg = CONFIG.get("case_study", {}); 
    tox.add_argument("--pretrain-epochs", type=int, default=case_cfg.get("pretrain_epochs", 5), help="JEPA pretrain epochs for case study"); 
    tox.add_argument("--finetune-epochs", type=int, default=case_cfg.get("finetune_epochs", 20), help="Epochs to train regression head in case study"); 
    tox.add_argument("--tox21-dir", dest="tox21_dir", type=str, required=False, default=None, help="Directory of Tox21 outputs"); 
    tox.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
    tox.add_argument("--triage-pct", type=float, default=0.10, help="Fraction of TEST to exclude (e.g., 0.10 = 10%%)")
    tox.add_argument("--no-calibrate", action="store_true", help="Disable Platt scaling on VAL")
    tox.add_argument("--contrastive", action="store_true",
                     help="Use contrastive pretraining instead of JEPA during the case study")
    _add_common_args(tox, "case_study")
    _add_model_args(tox)
    tox.set_defaults(func=cmd_tox21)

    # pointing to wandb hyberband
    sweep = sub.add_parser("sweep-run", help="Single trial run from W&B sweep")

    # paths + task core
    sweep.add_argument("--labeled-dir", "--labeled_dir", dest="labeled_dir", type=str, required=True)
    sweep.add_argument("--unlabeled-dir", "--unlabeled_dir", dest="unlabeled_dir", type=str, required=True)
    sweep.add_argument("--training-method", "--training_method", dest="training_method",
                    choices=["jepa","contrastive"], default="jepa")
    sweep.add_argument("--task-type", "--task_type", dest="task_type",
                    choices=["classification","regression"], default="regression")
    sweep.add_argument("--label-col", "--label_col", dest="label_col", type=str, default="label")

    # hparams unique to sweep-run
    sweep.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-3)
    sweep.add_argument("--seed", dest="seed", type=int, default=0)

    # training lengths / caps
    sweep.add_argument("--pretrain-batch-size", "--pretrain_batch_size", dest="pretrain_batch_size", type=int, default=64)
    sweep.add_argument("--finetune-batch-size", "--finetune_batch_size", dest="finetune_batch_size", type=int, default=64)
    sweep.add_argument("--pretrain-epochs", "--pretrain_epochs", dest="pretrain_epochs", type=int, default=10)
    sweep.add_argument("--finetune-epochs", "--finetune_epochs", dest="finetune_epochs", type=int, default=1)
    sweep.add_argument("--max-pretrain-batches", "--max_pretrain_batches", dest="max_pretrain_batches", type=int, default=0)
    sweep.add_argument("--max-finetune-batches", "--max_finetune_batches", dest="max_finetune_batches", type=int, default=0)
    sweep.add_argument("--sample-unlabeled", "--sample_unlabeled", dest="sample_unlabeled", type=int, default=0)
    sweep.add_argument("--sample-labeled", "--sample_labeled", dest="sample_labeled", type=int, default=0) 
    sweep.add_argument("--time-budget-mins", type=int, default=0)

    # aug flags not covered by common (structural ones)
    sweep.add_argument("--aug-bond-deletion", "--aug_bond_deletion", dest="aug_bond_deletion", type=int, choices=[0,1], default=0)
    sweep.add_argument("--aug-atom-masking",  "--aug_atom_masking",  dest="aug_atom_masking",  type=int, choices=[0,1], default=0)
    sweep.add_argument("--aug-subgraph-removal", "--aug_subgraph_removal", dest="aug_subgraph_removal", type=int, choices=[0,1], default=0)

    # pull in perf/common knobs once (devices, workers, pin/prefetch/bf16/use-wandb, etc.)
    _add_common_args(sweep, "sweep")
    _add_model_args(sweep)

    sweep.set_defaults(func=cmd_sweep_run)
    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    # This subcommand exposes the hyper‑parameter sweep functionality from
    # ``experiments.grid_search``.  It allows a user to optimise JEPA
    # pretraining and downstream evaluation parameters across a user‑defined
    # search space.  The defaults mirror those in ``run_grid_search`` but
    # can be overridden on the CLI.  The dataset is specified via
    # ``--dataset-dir`` and will be loaded with the same loader used in
    # other stages.  Seeds and search ranges can also be customised.
    grid = sub.add_parser("grid-search", help="Perform hyper‑parameter grid search for JEPA using run_grid_search")
    grid.add_argument("--smiles-col", type=str, default="smiles", help="Column name that contains molecule SMILES.")
    grid.add_argument("--dataset-dir", required=False, default=None, help="Path to a graph dataset used for both pretraining and evaluation. If omitted, you must specify --unlabeled-dir and/or --labeled-dir.")
    grid.add_argument("--unlabeled-dir", type=str, default=None, help="Directory of an unlabeled graph dataset for JEPA pretraining (e.g. ZINC/PubChem).")
    grid.add_argument("--labeled-dir", type=str, default=None, help="Directory of a labeled graph dataset for downstream evaluation (e.g. MoleculeNet).")
    grid.add_argument("--label-col", type=str, default="label", help="Name of the label column in the dataset (ignored for unlabeled data)")
    grid.add_argument("--cache-dir", type=str, default=None, help="Directory for cached graph data (defaults to cache/graphs)")
    grid.add_argument("--no-cache", action="store_true", help="Disable graph caching during grid search")
    grid.add_argument("--num-workers", type=int, default=0, help="Number of worker processes for SMILES featurization")
    grid.add_argument("--task-type", choices=["classification", "regression"], default="classification", help="Task type for downstream evaluation")
    grid.add_argument("--methods", nargs="+", default=["jepa"], help="Names of methods to include in the sweep (e.g. jepa contrastive)")
    grid.add_argument("--mask-ratios", type=float, nargs="+", default=[0.10, 0.15, 0.25], help="List of mask ratios to sweep over")
    grid.add_argument("--contiguities", type=int, nargs="+", default=[0, 1], help="Contiguity flags (0 for False, 1 for True) to sweep over")
    grid.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256], help="Hidden dimensions to sweep over")
    grid.add_argument("--num-layers-list", type=int, nargs="+", default=[2, 3], help="Number of GNN layers to sweep over")
    grid.add_argument("--gnn-types", nargs="+", default=["mpnn", "gcn", "gat", "edge_mpnn"], help="GNN architectures to sweep over")
    grid.add_argument("--ema-decays", type=float, nargs="+", default=[0.95, 0.99], help="EMA decay rates to sweep over")
    grid.add_argument("--add-3d-options", type=int, nargs="+", default=[0, 1], help="Whether to include 3D features (0 for False, 1 for True)")
    grid.add_argument("--aug-rotate-options", type=int, nargs="+", default=[int(DEFAULT_AUG.rotate)], help="Apply random rotation augmentation (0 for False, 1 for True)")
    grid.add_argument("--aug-mask-angle-options", type=int, nargs="+", default=[int(DEFAULT_AUG.mask_angle)], help="Apply angle masking augmentation (0 for False, 1 for True)")
    grid.add_argument("--aug-dihedral-options", type=int, nargs="+", default=[int(DEFAULT_AUG.dihedral)], help="Apply dihedral perturbation augmentation (0 for False, 1 for True)")
    grid.add_argument("--pretrain-batch-sizes", type=int, nargs="+", default=[256], help="Batch sizes for JEPA pretraining")
    grid.add_argument("--finetune-batch-sizes", type=int, nargs="+", default=[64], help="Batch sizes for downstream fine‑tuning")
    grid.add_argument("--pretrain-epochs-options", type=int, nargs="+", default=[50], help="Number of epochs for JEPA pretraining")
    grid.add_argument("--finetune-epochs-options", type=int, nargs="+", default=[30], help="Number of epochs for downstream training")
    grid.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4], help="Learning rates to sweep over")
    grid.add_argument("--seeds", type=int, nargs="*", default=None, help="Random seeds for averaging results (overrides config)")
    grid.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu)")
    grid.add_argument("--out-csv", type=str, default=None, help="Path to output CSV file for grid search results")
    grid.add_argument("--ckpt-dir", type=str, default="outputs/grid_ckpts", help="Directory in which to save intermediate checkpoints during the sweep")
    grid.add_argument("--ckpt-every", type=int, default=25, help="Checkpoint every N epochs during pretraining in the sweep")
    grid.add_argument("--use-scheduler", action="store_true", help="Enable learning‑rate warmup and cosine scheduler during grid search")
    grid.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps for the scheduler during grid search")
    grid.add_argument("--best-config-out", type=str, default=None, help="Optional path to write the best hyper‑parameter configuration as a JSON file. This file can be parsed later to drive a production pretraining run.")
    grid.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging for the grid search")
    grid.add_argument("--wandb-project", type=str, default=CONFIG.get("wandb", {}).get("project", "m-jepa"), help="W&B project name for grid search runs")
    grid.add_argument("--wandb-tags", nargs="*", default=CONFIG.get("wandb", {}).get("tags", []), help="W&B tags for grid search runs")
    grid.add_argument("--force-tqdm", action="store_true", help="Force-enable tqdm progress bars even when not attached to a TTY")
    grid.add_argument("--sample-unlabeled", type=int, default=0, help="If >0, load at most N graphs from the unlabeled dataset.")
    grid.add_argument("--sample-labeled", type=int, default=0, help="If >0, load at most N graphs from the labeled dataset.")
    grid.add_argument("--n-rows-per-file", type=int, default=None, help="If set, limit rows read per file when loading datasets.")
    grid.add_argument("--max-pretrain-batches", type=int, default=0, help="If >0, stop each pretrain epoch after this many batches.")
    grid.add_argument("--target-pretrain-samples", type=int, default=0, help="If >0, cap each trial to roughly this many pretrain samples")
    grid.add_argument("--max-finetune-batches", type=int, default=0, help="If >0, stop each finetune epoch after this many batches.")
    grid.add_argument("--time-budget-mins", type=int, default=0, help="Optional wallclock budget; stop early when exceeded.")
    grid.add_argument("--force-refresh", action="store_true", default=False, help="Ignore cached grid search outputs and recompute")
    grid.add_argument("--temperatures", type=float, nargs="+", default=[0.1], help="List of InfoNCE temperatures to try (contrastive only)")
    grid.add_argument("--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor (workers>0 only).")
    grid.add_argument("--pin-memory", action="store_true", default=True, help="Pin CUDA host memory in DataLoader.")
    grid.add_argument("--persistent-workers", action="store_true", default=True, help="Keep worker processes alive across epochs (workers>0).")
    grid.add_argument("--bf16", action="store_true", help="Enable bfloat16 autocast on GPU.")
    grid.add_argument("--devices", type=int, default=1, help="Number of GPUs for DDP")
    grid.set_defaults(func=cmd_grid_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.error("No subcommand provided")
    logger.info("Invoking subcommand %s", getattr(args, "func", None))
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(2)
