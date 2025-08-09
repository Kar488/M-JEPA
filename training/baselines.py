from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Third-party baselines live under third_party/.  We add their directories to
# sys.path so they can be imported like regular modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
 
def _add_repo_to_path(repo: str) -> None:
    repo_path = THIRD_PARTY / repo
    if repo_path.exists() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def _load_molclr() -> Callable[..., Any]:
    _add_repo_to_path("MolCLR")
    from molclr import main as molclr_main  # type: ignore

    return molclr_main

def _load_geomgcl() -> Callable[..., Any]:
    _add_repo_to_path("GeomGCL")
    from train_gcl import main as geomgcl_main  # type: ignore

    return geomgcl_main

def _load_himol() -> Callable[..., Any]:
    _add_repo_to_path("HiMol")
    from pretrain import main as himol_main  # type: ignore

    return himol_main


BASELINES: Dict[str, Callable[[], Callable[..., Any]]] = {
    "molclr": _load_molclr,
    "geomgcl": _load_geomgcl,
    "himol": _load_himol,
}


def run_baseline(
    name: str,
    *,
    dataset: Any,
    device: str = "cuda",
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to a third-party baseline training routine."""
    name = name.lower()
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline '{name}'")
    fn = BASELINES[name]()
    cfg = dict(cfg or {})
    return fn(dataset=dataset, device=device, **cfg, **kwargs)


# Backwards-compatibility alias
pretrain_baseline = run_baseline
