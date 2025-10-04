from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

__all__ = ["log_effective_gnn"]


def _argv_flags(argv: Iterable[str]) -> Iterable[str]:
    for token in argv:
        if token.startswith("--"):
            yield token


def _extract_gnn_value(data: Dict[str, Any]) -> str | None:
    if not isinstance(data, dict):
        return None
    direct = data.get("gnn_type")
    if isinstance(direct, dict) and "value" in direct:
        return direct.get("value")
    if isinstance(direct, str):
        return direct
    if direct is not None and not isinstance(direct, (dict, str)):
        try:
            return str(direct)
        except Exception:
            return None
    for key in ("parameters", "config"):
        block = data.get(key)
        if isinstance(block, dict):
            val = block.get("gnn_type")
            if isinstance(val, dict) and "value" in val:
                return val.get("value")
            if isinstance(val, str):
                return val
            if val is not None:
                try:
                    return str(val)
                except Exception:
                    return None
    return None


def _determine_gnn_source(args, argv: Iterable[str]) -> Tuple[str | None, str]:
    gnn = getattr(args, "gnn_type", None)
    if gnn is None:
        return None, "missing"

    norm = str(gnn)
    lower_norm = norm.lower()

    for flag in _argv_flags(argv):
        if flag in {"--gnn-type", "--gnn_type"} or flag.startswith("--gnn-type=") or flag.startswith("--gnn_type="):
            return norm, "cli"

    grid_dir = os.environ.get("GRID_DIR")
    if grid_dir:
        try:
            cfg_path = Path(grid_dir) / "best_grid_config.json"
            if cfg_path.is_file():
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                candidate = _extract_gnn_value(raw)
                if candidate is not None and str(candidate).lower() == lower_norm:
                    return norm, "best_json"
        except Exception:
            pass

    return norm, "yaml"


def _update_wandb_config(wb: Any, payload: Dict[str, Any]) -> None:
    if wb is None:
        return
    configs = []
    cfg = getattr(wb, "config", None)
    if cfg is not None:
        configs.append(cfg)
    run = getattr(wb, "run", None)
    if run is not None:
        run_cfg = getattr(run, "config", None)
        if run_cfg is not None:
            configs.append(run_cfg)
    for cfg_obj in configs:
        try:
            cfg_obj.update(payload, allow_val_change=True)
            return
        except TypeError:
            try:
                cfg_obj.update(payload)
                return
            except Exception:
                continue
        except Exception:
            continue


def log_effective_gnn(args, logger, wb: Any | None = None) -> Tuple[str | None, str]:
    gnn, source = _determine_gnn_source(args, sys.argv[1:])
    if gnn is None:
        logger.info("effective_gnn: gnn_type='(missing)' source='%s'", source)
        return None, source

    logger.info("effective_gnn: gnn_type='%s' source='%s'", gnn, source)
    _update_wandb_config(wb, {"effective_gnn_type": gnn, "effective_gnn_source": source})
    return gnn, source
