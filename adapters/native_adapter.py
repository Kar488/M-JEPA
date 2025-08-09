from __future__ import annotations
import importlib.util, importlib, sys
from pathlib import Path
from typing import Optional, Dict, Any

class NativeBaseline:
    """
    Load baseline repos in‑process by adding their path to sys.path and calling
    configured train/export functions. Your repo must expose Python-callable
    entrypoints (no argparse parsing side‑effects).
    """
    def __init__(self, repo_path: str, train_spec: Dict[str, str], embed_spec: Dict[str, str]):
        self.repo_path = str(Path(repo_path).resolve())
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        self.t_mod = importlib.import_module(train_spec["module"])
        self.t_fn = getattr(self.t_mod, train_spec["function"])
        self.e_mod = importlib.import_module(embed_spec["module"])
        self.e_fn = getattr(self.e_mod, embed_spec["function"])

    def train(self, **kwargs):
        return self.t_fn(**kwargs)

    def embed(self, **kwargs):
        return self.e_fn(**kwargs)
