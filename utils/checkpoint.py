from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def save_checkpoint(path: str, **states: Any) -> None:
    """Save arbitrary state dicts (e.g., encoder=..., optimizer=..., epoch=int)."""
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for checkpointing")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for checkpointing")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")
