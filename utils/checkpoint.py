from __future__ import annotations
import torch
from typing import Any, Dict
from pathlib import Path


def save_checkpoint(path: str, **objects: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(objects, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
