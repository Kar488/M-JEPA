from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str, **states: Any) -> None:
    """Save arbitrary state dicts (e.g., encoder=..., optimizer=..., epoch=int)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")
