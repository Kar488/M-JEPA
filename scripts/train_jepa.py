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
import math
import os
import sys
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - exercised when numpy missing
    class _MissingNumpy:
        """Minimal ``numpy`` shim that preserves import-time behaviour."""

        def __getattr__(self, name: str) -> "_MissingNumpy":
            raise ModuleNotFoundError("numpy is required for training")

        def __call__(self, *args, **kwargs):  # type: ignore[override]
            raise ModuleNotFoundError("numpy is required for training")

    np = _MissingNumpy()  # type: ignore[assignment]

# Attempt to import real PyTorch.  Some unit tests only need this module to be
# importable and will skip themselves if ``torch`` is absent.  To keep those
# tests lightweight we provide a tiny stub when the import fails; attempting to
# use it will still raise ``ModuleNotFoundError``.
try:  # pragma: no cover - exercised only when torch is missing
    import torch  # type: ignore
except Exception:  # noqa: BLE001 - broad to catch import errors
    class _MissingTorch:
        """Lightweight ``torch`` shim used when the real package is unavailable."""

        class _MissingCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_count() -> int:
                return 0

        cuda = _MissingCuda()

        def __getattr__(self, name: str) -> None:
            raise ModuleNotFoundError("torch is required for training")

    torch = _MissingTorch()  # type: ignore[assignment]
    sys.modules.setdefault("torch", torch)  # ensure subsequent imports see the stub


def _torch_cuda_available() -> bool:
    """Best-effort ``torch.cuda.is_available`` that tolerates missing torch."""

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False

    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False

    try:
        return bool(is_available())
    except Exception:
        return False


def _detect_visible_devices() -> int:
    """Infer the visible CUDA device count, falling back to a single slot.

    Prefers an explicit ``CUDA_VISIBLE_DEVICES`` mask when present; otherwise
    relies on ``torch.cuda.device_count``.  Returns ``1`` when CUDA is
    unavailable so that downstream consumers continue to run on CPU instead of
    erroring on ``0`` devices.
    """

    mask = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if mask:
        entries = [token.strip() for token in mask.split(",") if token.strip()]
        if entries:
            return len(entries)

    if _torch_cuda_available():
        try:
            count = int(torch.cuda.device_count())  # type: ignore[attr-defined]
            if count > 0:
                return count
        except Exception:
            pass

    return 1

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
        bond_deletion: bool = False
        atom_masking: bool = False
        subgraph_removal: bool = False


        def __init__(
            self,
            rotate: bool = False,
            mask_angle: bool = False,
            dihedral: bool = False,
            *,
            bond_deletion: bool = False,
            atom_masking: bool = False,
            subgraph_removal: bool = False,
            **kw,
        ):
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
            object.__setattr__(self, "random_rotate", bool(rotate))
            object.__setattr__(self, "perturb_dihedral", bool(dihedral))
            object.__setattr__(self, "bond_deletion", bool(bond_deletion))
            object.__setattr__(self, "atom_masking", bool(atom_masking))
            object.__setattr__(self, "subgraph_removal", bool(subgraph_removal))

        @classmethod
        def from_dict(cls, cfg: Optional[dict] = None) -> "AugmentationConfig":
            cfg = cfg or {}
            return cls(
                rotate=bool(cfg.get("rotate", False)),
                mask_angle=bool(cfg.get("mask_angle", False)),
                dihedral=bool(cfg.get("dihedral", False)),
                bond_deletion=bool(cfg.get("bond_deletion", False)),
                atom_masking=bool(cfg.get("atom_masking", False)),
                subgraph_removal=bool(cfg.get("subgraph_removal", False)),
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
    try:  # pragma: no cover - exercised when torch is installed
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover - torch missing in tests
        # When PyTorch is unavailable we still want this module to be importable
        # so that unit tests can skip themselves cleanly.  Delaying the error
        # until the fallback head is *used* keeps the import side‑effect free.

        def build_linear_head(*args, **kwargs):  # type: ignore[return-type]
            raise ModuleNotFoundError("torch is required for build_linear_head") from exc

    else:

        def build_linear_head(*args, **kwargs):
            """Fallback linear head using a single ``nn.Linear`` layer.

            This keeps unit tests lightweight while still exercising the
            orchestration code when the optional ``models.heads`` module is
            absent.  Requires PyTorch to be installed.
            """

            in_dim = kwargs.get("in_dim", args[0] if args else None)
            num_classes = kwargs.get("num_classes", args[1] if len(args) > 1 else None)
            if in_dim is None or num_classes is None:
                raise TypeError("in_dim and num_classes are required")
            return nn.Linear(in_dim, num_classes)


try:
    from training.unsupervised import (  # type: ignore[assignment]
        train_contrastive,
        train_jepa,
    )
except Exception:
    train_jepa = None  # type: ignore[assignment]
    train_contrastive = None  # type: ignore[assignment]

try:
    from training.supervised import (  # type: ignore[assignment]
        train_linear_head,
        set_stage_config as _set_linear_stage_config,
    )
except Exception:
    train_linear_head = None  # type: ignore[assignment]
    _set_linear_stage_config = None

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
    for attr in ("encoder", "module", "backbone"):
        sub = getattr(m, attr, None)
        if sub is None:
            continue
        sub_params = getattr(sub, "parameters", None)
        if callable(sub_params):
            try:
                return list(sub_params())
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
SOFT_TIMEOUT_EXIT_CODE = int(os.environ.get("LINEAR_HEAD_SOFT_TIMEOUT_EXIT", "86"))


def _coerce_stage_seconds(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            value = float(raw)
        else:
            text = str(raw).strip()
            if not text:
                return None
            multiplier = 1.0
            lowered = text.lower()
            if lowered.endswith("m") and lowered[:-1]:
                multiplier = 60.0
                lowered = lowered[:-1]
            elif lowered.endswith("s"):
                lowered = lowered[:-1]
            value = float(lowered)
            value *= multiplier
    except Exception:
        return None
    if value <= 0 or not math.isfinite(value):
        return None
    return value


def _collect_stage_config() -> Dict[str, Any]:
    env = os.environ

    def _first_seconds(*names: str) -> Optional[float]:
        for name in names:
            raw = env.get(name)
            if raw is None or str(raw).strip() == "":
                continue
            value = _coerce_stage_seconds(raw)
            if value is not None:
                return value
        return None

    cfg: Dict[str, Any] = {}

    timeout_secs = _first_seconds("STAGE_TIMEOUT_SECS", "STAGE_WALL_SECS", "ORCHESTRATOR_TIMEOUT_SECS")
    if timeout_secs is None:
        timeout_mins = _first_seconds("STAGE_TIMEOUT_MINS", "HARD_WALL_MINS")
        if timeout_mins is not None:
            timeout_secs = timeout_mins * 60.0
    if timeout_secs is not None:
        cfg["timeout_secs"] = timeout_secs

    grace_secs = _first_seconds("STAGE_GRACE_SECS", "ORCHESTRATOR_GRACE_SECS", "KILL_AFTER_SECS")
    if grace_secs is not None:
        cfg["grace_secs"] = grace_secs

    heartbeat_secs = _first_seconds("STAGE_HEARTBEAT_SECS", "ORCHESTRATOR_HEARTBEAT_SECS", "PIPELINE_HEARTBEAT_SECS")
    if heartbeat_secs is not None:
        cfg["heartbeat_secs"] = heartbeat_secs

    heartbeat_path = next(
        (env.get(name) for name in ("STAGE_HEARTBEAT_PATH", "PIPELINE_HEARTBEAT_PATH") if env.get(name)),
        None,
    )
    if heartbeat_path:
        cfg["heartbeat_path"] = heartbeat_path

    hard_wall_mins = env.get("HARD_WALL_MINS")
    if hard_wall_mins is not None and str(hard_wall_mins).strip() != "":
        try:
            cfg["hard_wall_mins"] = float(hard_wall_mins)
        except Exception:
            pass

    return cfg


STAGE_CONFIG = _collect_stage_config()
if _set_linear_stage_config is not None:
    try:
        _set_linear_stage_config(STAGE_CONFIG)
    except Exception:
        logger.debug("Failed to propagate stage config to training.supervised", exc_info=True)

_aug_raw = CONFIG.get("pretrain", {}).get("augmentations", {}) or {}
_aug_raw = {
    # accept either style from YAML
    "rotate": bool(_aug_raw.get("rotate", _aug_raw.get("random_rotate", False))),
    "mask_angle": bool(_aug_raw.get("mask_angle", False)),
    "dihedral": bool(_aug_raw.get("dihedral", _aug_raw.get("perturb_dihedral", False))),
    "bond_deletion": bool(_aug_raw.get("bond_deletion", False)),
    "atom_masking": bool(_aug_raw.get("atom_masking", False)),
    "subgraph_removal": bool(_aug_raw.get("subgraph_removal", False)),
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

if "bond_deletion" in params:
    _mapped["bond_deletion"] = _aug_raw["bond_deletion"]
if "atom_masking" in params:
    _mapped["atom_masking"] = _aug_raw["atom_masking"]
if "subgraph_removal" in params:
    _mapped["subgraph_removal"] = _aug_raw["subgraph_removal"]

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
        bond_deletion=_aug_raw["bond_deletion"],
        atom_masking=_aug_raw["atom_masking"],
        subgraph_removal=_aug_raw["subgraph_removal"],
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean and std for each numeric metric across runs.

    Excludes the key 'head' if present and skips non-numeric values.
    """
    if not metrics_list:
        return {}
    out: Dict[str, float] = {}
    keys = sorted({k for d in metrics_list for k in d.keys() if k != "head"})
    for k in keys:
        raw_vals: List[float] = []
        for d in metrics_list:
            if k not in d:
                continue
            try:
                raw_vals.append(float(d[k]))
            except Exception:
                continue
        if not raw_vals:
            continue
        vals = np.array(raw_vals, dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def resolve_device(preferred: str) -> str:
    """Return a valid PyTorch device string."""

    try:
        if not preferred:
            return "cpu"

        # ``torch.device`` performs basic validation (e.g. catches typos such as
        # ``cudda``).  When the real torch package is unavailable our stub raises
        # ``ModuleNotFoundError`` here, which we treat as a signal to fall back to
        # CPU.
        device = torch.device(preferred)
    except Exception:
        return "cpu"

    if device.type != "cuda":
        return str(device)

    try:
        cuda = getattr(torch, "cuda")
    except Exception:
        return "cpu"

    try:
        if not cuda.is_available():  # type: ignore[attr-defined]
            return "cpu"

        index = getattr(device, "index", None)
        if index is not None and index >= 0:
            try:
                if index >= cuda.device_count():  # type: ignore[attr-defined]
                    return "cpu"
            except Exception:
                return "cpu"

        # Some environments ship CUDA builds of PyTorch but do not provide a
        # functioning GPU runtime (common on CI machines).  Attempting to touch
        # CUDA later would surface an opaque ``AcceleratorError``.  Proactively
        # allocate a tiny tensor so we can fall back to CPU with a clear error
        # message instead.
        try:
            torch.empty(0, device=device)
        except Exception:
            return "cpu"

        return str(device)
    except Exception:
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
        context_value = getattr(CMD_CONTEXT, name)
        override = getattr(this, name, context_value)
        if override is None and hasattr(m, name):
            existing = getattr(m, name)
            if existing is not None and context_value is None:
                continue
        setattr(m, name, override)


def cmd_sweep_run(args: argparse.Namespace) -> None:
    _sweep_run = _load_cmd("sweep_run")
    _inject_shared(_sweep_run)
    _sweep_run.cmd_sweep_run(args)


def cmd_grid_search(args: argparse.Namespace) -> None:
    _grid_search = _load_cmd("grid_search")
    _inject_shared(_grid_search)
    # Propagate a monkeypatched run_grid_search into the command module.
    # CMD_CONTEXT captures run_grid_search at import time, so tests that
    # patch ``train_jepa.run_grid_search`` would otherwise have no effect.
    # Reassign here to ensure the latest reference is used.
    _grid_search.run_grid_search = run_grid_search
    _grid_search.cmd_grid_search(args)


def cmd_pretrain(args: argparse.Namespace) -> None:
    _pretrain = _load_cmd("pretrain")
    _inject_shared(_pretrain)
    _pretrain.cmd_pretrain(args)


def cmd_finetune(args: argparse.Namespace) -> None:
    _finetune = _load_cmd("finetune")
    _inject_shared(_finetune)
    _finetune.cmd_finetune(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    # The evaluate command shares the finetune implementation.
    # Calling our wrapper ensures any monkeypatched dependencies are used.
    cmd_finetune(args)


def cmd_benchmark(args: argparse.Namespace) -> None:
    _benchmark = _load_cmd("benchmark")
    _inject_shared(_benchmark)
    _benchmark.cmd_benchmark(args)


def cmd_tox21(args: argparse.Namespace) -> None:
    _tox21 = _load_cmd("tox21")
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
    stage_config: Dict[str, Any]
    soft_timeout_exit_code: int


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
    stage_config=STAGE_CONFIG,
    soft_timeout_exit_code=SOFT_TIMEOUT_EXIT_CODE,
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
    aug_bond_deletion: bool
    aug_atom_masking: bool
    aug_subgraph_removal: bool
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
            num_workers=-1,
            cache_dir=None,
            contiguous=False,
            aug_rotate=DEFAULT_AUG.rotate,
            aug_mask_angle=DEFAULT_AUG.mask_angle,
            aug_dihedral=DEFAULT_AUG.dihedral,
            aug_bond_deletion=DEFAULT_AUG.bond_deletion,
            aug_atom_masking=DEFAULT_AUG.atom_masking,
            aug_subgraph_removal=DEFAULT_AUG.subgraph_removal,
            epochs=sec_cfg.get("epochs", 1),
            batch_size=sec_cfg.get("batch_size", 32),
            lr=sec_cfg.get("lr", 1e-3),
            seeds=None,
            device="cuda" if _torch_cuda_available() else "cpu",
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            bf16=False,
            devices=_detect_visible_devices(),
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
        try:
            setattr(namespace, f"_{self.dest}_provided", True)
        except Exception:
            pass

def _add_common_args(p: argparse.ArgumentParser, section: str) -> None:
    """Add arguments common to multiple commands."""

    d = CommonArgDefaults.from_config(section)
    p.set_defaults(_epochs_provided=False)
    p.add_argument("--add-3d", "--add_3d", dest="add_3d", action=BoolFlag, default=d.add_3d, help="Augment with 3D coordinate featurisation")
    p.add_argument("--cache-dir", type=str, default=d.cache_dir, help="Directory to cache processed graphs")
    p.add_argument("--contiguity", "--contiguous", dest="contiguity", action=BoolFlag, default=d.contiguous, help="Use contiguous subgraph masking (JEPA)")
    p.add_argument("--aug-rotate", "--aug_rotate",dest="aug_rotate", action=BoolFlag, default=d.aug_rotate, help="Randomly rotate coordinates during pretraining")
    p.add_argument("--aug-mask-angle", "--aug_mask_angle", dest="aug_mask_angle", action=BoolFlag, default=d.aug_mask_angle, help="Mask bond angles during pretraining")
    p.add_argument("--aug-dihedral", "--aug_dihedral", dest="aug_dihedral", action=BoolFlag, default=d.aug_dihedral, help="Perturb dihedral angles during pretraining")
    p.add_argument("--aug-bond-deletion", "--aug_bond_deletion", dest="aug_bond_deletion", action=BoolFlag, default=d.aug_bond_deletion, help="Randomly delete bonds during pretraining")
    p.add_argument("--aug-atom-masking", "--aug_atom_masking", dest="aug_atom_masking", action=BoolFlag, default=d.aug_atom_masking, help="Mask random atom features during pretraining")
    p.add_argument("--aug-subgraph-removal", "--aug_subgraph_removal", dest="aug_subgraph_removal", action=BoolFlag, default=d.aug_subgraph_removal, help="Remove small subgraphs during pretraining")
    p.add_argument(
        "--epochs",
        type=int,
        default=d.epochs,
        action=_RecordProvided,
        help="Number of training epochs",
    )
    p.add_argument("--batch-size", type=int, default=d.batch_size, help="Batch size")
    p.add_argument("--lr", type=float, default=d.lr, help="Learning rate")
    p.add_argument("--seeds", type=int, nargs="*", default=d.seeds, help="Random seeds for averaging results")
    
    p.add_argument("--prefetch-factor", type=int, default=d.prefetch_factor, help="Dataloader prefetch factor (workers>0 only).")
    p.add_argument("--pin-memory", "--pin_memory", dest="pin_memory", action=BoolFlag, default=d.pin_memory, help="Pin CUDA host memory in DataLoader.")
    p.add_argument("--persistent-workers", "--persistent_workers", dest="persistent_workers", action=BoolFlag, default=d.persistent_workers, help="Keep worker processes alive across epochs (workers>0).")
    p.add_argument("--bf16", action=BoolFlag, default=d.bf16, help="Enable bfloat16 autocast on GPU.")
    p.add_argument("--device", type=str, default=d.device, help="Device")
    p.add_argument("--devices", type=int, default=d.devices, help="Number of GPUs for DDP")
    p.add_argument(
        "--num-workers",
        type=int,
        default=d.num_workers,
        help="Process pool workers for SMILES conversion (-1=auto, 0=serial)",
    )
    
    p.add_argument("--use-wandb", "--use_wandb", dest="use_wandb", action=BoolFlag, default=d.use_wandb, help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=d.wandb_project, help="W&B project name")
    p.add_argument("--wandb-tags", nargs="*", default=d.wandb_tags, help="Tags for W&B run")

class _RecordProvided(argparse.Action):
    """Argparse action that tracks whether a flag was explicitly provided."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_provided", True)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    md = CONFIG.get("model", {})
    # Track whether structural knobs were supplied explicitly. CI can then
    # distinguish between defaults vs. best-config overrides when routing
    # encoder metadata into downstream stages such as Tox21.
    p.set_defaults(
        _gnn_type_provided=False,
        _hidden_dim_provided=False,
        _num_layers_provided=False,
        _dropout_provided=False,
    )
    gnn_default = md.get("gnn_type", "edge_mpnn")
    hidden_default = md.get("hidden_dim", 128)
    layers_default = md.get("num_layers", 2)
    p.add_argument(
        "--gnn-type",
        "--gnn_type",
        dest="gnn_type",
        choices=[
            "gcn",
            "gat",
            "mpnn",
            "edge_mpnn",
            "graphsage",
            "gin",
            "gine",
            "dmpnn",
            "attentivefp",
            "schnet3d",
        ],
        default=gnn_default,
        action=_RecordProvided,
        help=(
            "Backbone GNN. Use 'gine' (GIN+edge) or 'dmpnn' (Chemprop-style directed MPNN) "
            "for 2D/bond-aware runs; 'schnet3d' for 3D geometry (requires pos); "
            "'attentivefp' for attention readout over atoms/bonds."
        ),
    )
    p.add_argument(
        "--hidden-dim",
        "--hidden_dim",
        dest="hidden_dim",
        type=int,
        default=hidden_default,
        action=_RecordProvided,
    )
    p.add_argument(
        "--num-layers",
        "--num_layers",
        dest="num_layers",
        type=int,
        default=layers_default,
        action=_RecordProvided,
    )
    dropout_default = md.get("dropout", 0.1)
    p.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=dropout_default,
        action=_RecordProvided,
        help="Dropout probability applied within GNN layers (where supported)",
    )
    p.add_argument("--mask-ratio", "--mask_ratio", dest="mask_ratio", type=float, default=md.get("mask_ratio", 0.15))
    p.add_argument("--ema-decay", "--ema_decay", dest="ema_decay", type=float, default=md.get("ema_decay", 0.996))
    p.add_argument("--temperature", dest="temperature", type=float, default=md.get("temperature", 0.1))

def build_parser() -> argparse.ArgumentParser:

    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="JEPA training and evaluation pipeline"
    )
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
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
    pre.add_argument(
        "--probe-dataset",
        dest="probe_dataset",
        type=str,
        default=CONFIG.get("pretrain", {}).get("probe_dataset"),
        help="Labelled dataset for periodic linear probing (pretraining only).",
    )
    pre.add_argument(
        "--probe-label-col",
        dest="probe_label_col",
        type=str,
        default=CONFIG.get("pretrain", {}).get("probe_label_col", "label"),
        help="Label column to use when loading the probe CSV dataset.",
    )
    pre.add_argument(
        "--probe-interval",
        dest="probe_interval",
        type=int,
        default=CONFIG.get("pretrain", {}).get("probe_interval", 0),
        help="Run the probe every N epochs (0 disables).",
    )
    pre.add_argument("--sample-unlabeled", type=int, default=0, help="If >0, load at most N graphs from the unlabeled dataset.")
    pre.add_argument("--n-rows-per-file", type=int, default=None, help="If set, limit rows read per file when loading datasets.")
    pre.add_argument(
        "--stream-chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, stream unlabeled Parquet files in chunks of this many rows per file during pretraining "
            "instead of loading the full dataset into memory."
        ),
    )
    pre.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile wrappers for the encoder and predictor.",
    )
    pre.add_argument(
        "--pretrain-epochs",
        "--pretrain_epochs",
        dest="pretrain_epochs",
        type=int,
        default=None,
        action=_RecordProvided,
        help="Number of epochs for JEPA pretraining (defaults to --epochs when unset)",
    )

    _add_common_args(pre, "pretrain")
    _add_model_args(pre)
    pre.set_defaults(func=cmd_pretrain)

    # Fine‑tune subcommand
    ft = sub.add_parser("finetune", help="Fine‑tune a linear head on labelled data")
    ft.add_argument("--labeled-dir", required=True, help="Directory containing labelled graphs")
    ft.add_argument(
        "--labeled-csv",
        dest="labeled_csv",
        default=None,
        help="Optional CSV file containing labelled graphs when the directory holds multiple assets",
    )
    ft.add_argument("--label-col", type=str, default="label", help="Label column name in input files")
    ft.add_argument("--encoder", required=True, help="Path to a pretrained encoder checkpoint (.pt)")
    ft.add_argument("--ckpt-dir", type=str, default="ckpts/finetune", help="dir to write fine-tune checkpoints")
    ft.add_argument("--resume-ckpt", type=str, default="", help="resume fine-tune from this checkpoint")
    ft.add_argument("--save-every", type=int, default=1, help="save checkpoint every N epochs")
    ft.add_argument("--save-final", action="store_true", help="also save ft_last.pt at the end")
    ft.add_argument(
        "--metric",
        "--early-stop-metric",
        dest="metric",
        type=str,
        default=None,
        choices=["val_loss", "val_auc", "auc", "auroc", "roc_auc", "val_rmse", "val_mae", "val_r2"],
        help="Validation metric used for early stopping and checkpoint selection",
    )
    ft.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    ft.add_argument("--patience", type=int, default=CONFIG.get("finetune", {}).get("patience", 10), help="Early stopping patience")
    ft.add_argument(
        "--use-scaffold",
        dest="use_scaffold",
        action=BoolFlag,
        default=False,
        help="Enable Murcko scaffold splits when SMILES are available during fine-tune",
    )
    ft.add_argument(
        "--load-encoder-checkpoint",
        dest="load_encoder_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint containing encoder weights to load before fine-tuning",
    )
    if bool_action is not None:
        ft.add_argument(
            "--freeze-encoder",
            action=bool_action,
            default=None,
            help="Freeze encoder weights during fine-tuning (use --no-freeze-encoder to train them)",
        )
    else:
        ft.add_argument(
            "--freeze-encoder",
            dest="freeze_encoder",
            action="store_true",
            default=None,
            help="Freeze encoder weights during fine-tuning",
        )
        ft.add_argument(
            "--no-freeze-encoder",
            dest="freeze_encoder",
            action="store_false",
            help="Allow encoder weights to update during fine-tuning",
        )
    ft.add_argument(
        "--unfreeze-top-layers",
        type=int,
        default=0,
        help="When freezing the encoder, unfreeze the top-N child modules",
    )
    ft.add_argument(
        "--unfreeze",
        dest="unfreeze_mode",
        choices=["none", "partial", "full"],
        default=CONFIG.get("finetune", {}).get("unfreeze", "none"),
        help=(
            "Encoder update policy: none=frozen probe, partial=top layers trainable, "
            "full=end-to-end. When enabling updates start with head_lr≈1e-4 and encoder_lr≈1e-4 "
            "for partial unfreeze; full unfreeze typically prefers encoder_lr in the 3e-6–1e-5 range."
        ),
    )
    ft.add_argument(
        "--encoder-lr",
        dest="encoder_lr",
        type=float,
        default=1e-4,
        help="Learning rate for encoder parameters when trainable (defaults to 1e-4)",
    )
    ft.add_argument(
        "--head-lr",
        dest="head_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the fine-tuning head (defaults to 1e-4)",
    )
    ft.add_argument(
        "--layerwise-decay",
        dest="layerwise_decay",
        type=float,
        default=None,
        help="Optional multiplicative decay applied to deeper encoder layers (e.g., 0.9)",
    )
    ft.add_argument(
        "--use-focal-loss",
        dest="use_focal_loss",
        action=BoolFlag,
        default=False,
        help="Wrap BCEWithLogits in focal loss for imbalanced labels",
    )
    ft.add_argument(
        "--focal-gamma",
        dest="focal_gamma",
        type=float,
        default=2.0,
        help="Gamma exponent for focal loss (ignored when disabled)",
    )
    ft.add_argument(
        "--dynamic-pos-weight",
        dest="dynamic_pos_weight",
        action=BoolFlag,
        default=False,
        help="Recompute pos_weight from the current training split each epoch",
    )
    ft.add_argument(
        "--oversample-minority",
        dest="oversample_minority",
        action=BoolFlag,
        default=False,
        help="Use a WeightedRandomSampler to upsample positive labels during fine-tune",
    )
    ft.add_argument(
        "--calibrate-probabilities",
        dest="calibrate_probabilities",
        action=BoolFlag,
        default=False,
        help="Apply post-hoc probability calibration (temperature scaling or isotonic)",
    )
    ft.add_argument(
        "--calibration-method",
        dest="calibration_method",
        choices=["temperature", "isotonic"],
        default="temperature",
        help="Calibration routine to pair with --calibrate-probabilities",
    )
    ft.add_argument(
        "--threshold-metric",
        dest="threshold_metric",
        choices=["f1", "roc_auc"],
        default="f1",
        help="Validation metric to optimise when tuning the decision threshold",
    )
    ft.add_argument(
        "--pos-class-weight",
        dest="pos_weight",
        action="append",
        default=None,
        help=(
            "Positive class weight (float) or per-task override TASK=weight; repeatable for multiple tasks"
        ),
    )
    ft.add_argument(
        "--per-task-hparams",
        dest="per_task_hparams",
        type=str,
        default=None,
        help="JSON or YAML mapping of task → hyperparameters (e.g., pos_weight overrides)",
    )
    ft.add_argument(
        "--max-finetune-batches",
        dest="max_finetune_batches",
        type=int,
        default=0,
        help=(
            "Optional cap on batches per fine-tune epoch (0 = uncapped). "
            "This is independent from --max-pretrain-batches."
        ),
    )
    ft.add_argument(
        "--best-config",
        dest="best_config",
        type=str,
        default=None,
        help="Optional path to a best_config JSON file to inherit downstream overrides.",
    )
    ft.add_argument(
        "--best-config-json",
        dest="best_config_json",
        type=str,
        default=None,
        help="Alias for --best-config when providing a JSON file with tuned hyper-parameters.",
    )
    ft.add_argument(
        "--best-config-path",
        dest="best_config_path",
        type=str,
        default=None,
        help="Alias for --best-config to ease integration with existing CI tooling.",
    )
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
    ev.add_argument(
        "--best-config",
        dest="best_config",
        type=str,
        default=None,
        help="Optional path to a best_config JSON file to inherit downstream overrides.",
    )
    ev.add_argument(
        "--best-config-json",
        dest="best_config_json",
        type=str,
        default=None,
        help="Alias for --best-config when providing a JSON file with tuned hyper-parameters.",
    )
    ev.add_argument(
        "--best-config-path",
        dest="best_config_path",
        type=str,
        default=None,
        help="Alias for --best-config to ease integration with existing CI tooling.",
    )
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
    bench.add_argument("--dataset", type=str, default=None, help="Dataset name for benchmark threshold lookup");
    bench.add_argument("--task", type=str, default=None, help="Optional task identifier for benchmark threshold lookup");
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
    tox.add_argument(
        "--task",
        required=False,
        default=None,
        help="Name of the toxicity column to predict (deprecated when --tasks is used)",
    );
    tox.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of toxicity columns to evaluate (defaults to all Tox21 assays when omitted)",
    );
    tox.add_argument("--dataset", type=str, default="tox21", help="Dataset name for threshold lookup");
    case_cfg = CONFIG.get("case_study", {});
    tox.add_argument("--pretrain-epochs", type=int, default=case_cfg.get("pretrain_epochs", 5), help="JEPA pretrain epochs for case study"); 
    tox.add_argument("--finetune-epochs", type=int, default=case_cfg.get("finetune_epochs", 20), help="Epochs to train regression head in case study"); 
    tox.add_argument("--tox21-dir", dest="tox21_dir", type=str, required=False, default=None, help="Directory of Tox21 outputs"); 
    tox.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)
    tox.add_argument(
        "--pretrain-lr",
        dest="pretrain_lr",
        type=float,
        default=case_cfg.get("pretrain_lr", 1e-4),
        help="Learning rate used during the JEPA pretraining phase of the case study",
    )
    tox.add_argument(
        "--head-lr",
        dest="head_lr",
        type=float,
        default=case_cfg.get("head_lr"),
        help="Learning rate for the linear head during Tox21 fine-tuning (defaults to --learning-rate)",
    )
    tox.add_argument(
        "--encoder-lr",
        dest="encoder_lr",
        type=float,
        default=case_cfg.get("encoder_lr"),
        help="Learning rate for encoder parameters when they are trainable during Tox21 runs",
    )
    tox.add_argument(
        "--full-finetune",
        dest="full_finetune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable full encoder fine-tuning on the Tox21 train split",
    )
    tox.add_argument(
        "--unfreeze-top-layers",
        dest="unfreeze_top_layers",
        type=int,
        default=case_cfg.get("unfreeze_top_layers", 0),
        help="When fine-tuning, number of top encoder layers to unfreeze (0 = all)",
    )
    tox.add_argument(
        "--tox21-head-batch-size",
        dest="tox21_head_batch_size",
        type=int,
        default=case_cfg.get("tox21_head_batch_size", 256),
        help="Batch size for the Tox21 head/encoder fine-tuning stage",
    )
    tox.add_argument(
        "--head-ensemble-size",
        dest="head_ensemble_size",
        type=int,
        default=case_cfg.get("head_ensemble_size", 1),
        help="Number of independent heads to train and ensemble during Tox21 evaluation",
    )
    tox.add_argument(
        "--head-scheduler",
        dest="head_scheduler",
        default=case_cfg.get("head_scheduler"),
        help="Optional learning-rate scheduler for the Tox21 head (e.g., cosine)",
    )
    tox.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=case_cfg.get("weight_decay"),
        help="Weight decay applied to the Tox21 linear head optimiser",
    )
    tox.add_argument(
        "--class-weights",
        dest="class_weights",
        default=case_cfg.get("class_weights"),
        help="Class weighting policy for Tox21 (auto, none, or JSON mapping)",
    )
    tox.add_argument(
        "--pos-class-weight",
        dest="pos_class_weight",
        action="append",
        default=None,
        help="Override the positive class weight (float or TASK=weight). Repeatable.",
    )
    tox.add_argument(
        "--triage-pct",
        type=float,
        default=case_cfg.get("triage_pct", 0.0),
        help="Fraction of TEST to exclude (set to 0.0 to keep all examples)",
    )
    tox.add_argument("--no-calibrate", action="store_true", help="Disable Platt scaling on VAL")
    tox.add_argument(
        "--freeze-encoder",
        dest="freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=case_cfg.get("freeze_encoder", False),
        help="Force the encoder to remain frozen during Tox21 fine-tuning",
    )
    tox.add_argument("--contrastive", action="store_true",
                     help="Use contrastive pretraining instead of JEPA during the case study")
    tox.add_argument(
        "--encoder-checkpoint",
        dest="encoder_checkpoint",
        type=str,
        default=None,
        help="Optional pretrained encoder checkpoint to evaluate without additional pretraining",
    )
    tox.add_argument(
        "--encoder-source",
        dest="encoder_source",
        type=str,
        default=None,
        help="Label describing the encoder variant being evaluated (e.g. pretrain_frozen, fine_tuned)",
    )
    tox.add_argument(
        "--evaluation-mode",
        dest="evaluation_mode",
        choices=["pretrain_frozen", "frozen_finetuned", "fine_tuned", "end_to_end"],
        default="pretrain_frozen",
        help="Tox21 evaluation policy: frozen pretrain baseline, frozen finetuned encoder, or end-to-end fine-tuned model",
    )
    tox.add_argument(
        "--explain-mode",
        dest="explain_mode",
        default=case_cfg.get("explain_mode"),
        help="Optional explanation mode (e.g., ig) to enable attribution logging during evaluation",
    )
    tox.add_argument(
        "--explain-steps",
        dest="explain_steps",
        type=int,
        default=case_cfg.get("explain_steps"),
        help="Number of interpolation steps to use for explanation methods",
    )
    tox.add_argument(
        "--strict-encoder-config",
        dest="strict_encoder_config",
        action="store_true",
        help="Require CLI model arguments to match the encoder checkpoint configuration",
    )
    tox.add_argument(
        "--allow-shape-coercion",
        dest="allow_shape_coercion",
        action="store_true",
        default=None,
        help=(
            "Permit resizing checkpoint tensors when loading encoders for Tox21 evaluation. "
            "If omitted, a best-effort fallback will retry with coercion when strictly loading fails."
        ),
    )
    tox.add_argument(
        "--allow-equal-hash",
        dest="allow_equal_hash",
        action="store_true",
        help="Allow fine-tuned evaluations to proceed even when encoder hash matches the baseline",
    )
    tox.add_argument(
        "--verify-match-threshold",
        dest="verify_match_threshold",
        type=float,
        default=0.98,
        help="Minimum fraction of encoder parameters that must match when loading checkpoints",
    )
    tox.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=12,
        help="Patience (epochs) for Tox21 head training early stopping",
    )
    tox.add_argument(
        "--bf16-head",
        dest="bf16_head",
        action="store_true",
        help="Enable bfloat16 mixed precision when training the linear head",
    )
    tox.add_argument(
        "--pretrain-time-budget-mins",
        type=int,
        default=0,
        help="Optional wall-clock budget (minutes) for the pretraining phase; 0 disables.",
    )
    tox.add_argument(
        "--finetune-time-budget-mins",
        type=int,
        default=0,
        help="Optional wall-clock budget (minutes) for the finetuning phase; 0 disables.",
    )
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
    sweep.add_argument(
        "--augmentation-profile",
        "--augmentation_profile",
        dest="augmentation_profile",
        type=str,
        default=None,
        help="Optional augmentation profile name supplied by sweep seeds",
    )

    # training lengths / caps
    sweep.add_argument("--pretrain-batch-size", "--pretrain_batch_size", dest="pretrain_batch_size", type=int, default=64)
    sweep.add_argument("--finetune-batch-size", "--finetune_batch_size", dest="finetune_batch_size", type=int, default=64)
    sweep.add_argument("--pretrain-epochs", "--pretrain_epochs", dest="pretrain_epochs", type=int, default=10)
    sweep.add_argument("--finetune-epochs", "--finetune_epochs", dest="finetune_epochs", type=int, default=1)
    sweep.add_argument("--max-pretrain-batches", "--max_pretrain_batches", dest="max_pretrain_batches", type=int, default=0)
    sweep.add_argument("--max-finetune-batches", "--max_finetune_batches", dest="max_finetune_batches", type=int, default=0)
    sweep.add_argument(
        "--max-graphs-per-run",
        "--max_graphs_per_run",
        dest="max_graphs_per_run",
        type=int,
        default=0,
    )
    sweep.add_argument("--sample-unlabeled", "--sample_unlabeled", dest="sample_unlabeled", type=int, default=0)
    sweep.add_argument(
        "--stream-chunk-size",
        "--stream_chunk_size",
        dest="stream_chunk_size",
        type=int,
        default=0,
    )
    sweep.add_argument("--sample-labeled", "--sample_labeled", dest="sample_labeled", type=int, default=0)
    sweep.add_argument("--time-budget-mins", type=int, default=0)
    sweep.add_argument(
        "--cache-datasets",
        "--cache_datasets",
        dest="cache_datasets",
        action=BoolFlag,
        default=False,
        help="Serialize GraphDataset objects for reuse across sweep trials",
    )
    sweep.add_argument(
        "--pair-id",
        "--pair_id",
        dest="pair_id",
        default=None,
        help="Optional pairing identifier supplied by sweep seeds; ignored when absent",
    )

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
    grid.add_argument(
        "--device",
        type=str,
        default="cuda" if _torch_cuda_available() else "cpu",
        help="Device for training (cuda or cpu)",
    )
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
    grid.add_argument(
        "--stream-chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, stream unlabeled Parquet files in chunks of this many rows per file during pretraining "
            "instead of loading the full dataset into memory."
        ),
    )
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
