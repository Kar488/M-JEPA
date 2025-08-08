from __future__ import annotations
from typing import Optional, Dict, Any

class DummyWandb:
    def __init__(self): self._ok = False
    def log(self, *a, **k): pass
    def finish(self): pass

def maybe_init_wandb(enable: bool, project: str = "m-jepa", config: Optional[Dict[str, Any]] = None):
    if not enable:
        return DummyWandb()
    try:
        import wandb
        wandb.init(project=project, config=config or {})
        return wandb
    except Exception:
        return DummyWandb()
