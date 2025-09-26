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


# scripts/utils/checkpoint.py
import os, shutil, logging, torch
logger = logging.getLogger(__name__)
from collections.abc import Mapping


def _copy_within_shape(target, source):
    """Create a tensor shaped like ``target`` populated with overlapping data from ``source``."""

    if torch is None:
        return None
    if not torch.is_tensor(target) or not torch.is_tensor(source):
        return None
    if target.ndim != source.ndim:
        return None

    try:
        new_tensor = target.clone().zero_()
    except Exception:
        return None

    slices = tuple(slice(0, min(ts, ss)) for ts, ss in zip(target.shape, source.shape))
    try:
        new_tensor[slices] = source[slices]
    except Exception:
        return None
    return new_tensor

def load_state_dict_forgiving(module, state_dict):
    """
    Best-effort weight loader:
      - If module supports .load_state_dict, try non-strict load and log missing/unexpected keys.
      - On failure, if module exposes .state_dict(), retry with shape-filtered keys.
      - If module DOES NOT have .load_state_dict/.state_dict (e.g., Dummy in tests), skip and return None.
    Returns the LoadStateDictResult when available; otherwise None.
    """
    # If the "state_dict" isn't even a mapping, there's nothing sensible to load.
    if not isinstance(state_dict, Mapping):
        logger.warning("Forgiving load skipped: state_dict is not a mapping (%s).", type(state_dict).__name__)
        return None

    # If module can't load, just skip (test doubles / dummies)
    if not hasattr(module, "load_state_dict"):
        logger.warning("Module %s has no load_state_dict(); skipping weight load.", type(module).__name__)
        return None

    try:
        res = module.load_state_dict(state_dict, strict=False)
        miss = getattr(res, "missing_keys", [])
        unexp = getattr(res, "unexpected_keys", [])
        if miss or unexp:
            logger.warning("Non-strict load: missing=%s unexpected=%s", miss, unexp)
        return res
    except Exception:
        logger.exception("Forgiving load: non-strict load failed, attempting shape-filtered keys")

    # Fallback: only possible if module exposes a reference state to compare shapes
    if not hasattr(module, "state_dict"):
        logger.warning("Module %s has no state_dict(); cannot shape-filter. Skipping load.", type(module).__name__)
        return None

    try:
        current = module.state_dict()
        if not isinstance(current, Mapping):
            logger.warning("Module %s state_dict() did not return a Mapping; skipping load.", type(module).__name__)
            return None

        adapted = {}
        resized: list[str] = []
        for key, value in state_dict.items():
            if key not in current:
                continue
            curr_val = current[key]
            curr_shape = getattr(curr_val, "shape", None)
            value_shape = getattr(value, "shape", None)
            if curr_shape is None or value_shape is None:
                if curr_shape == value_shape:
                    adapted[key] = value
                continue

            if curr_shape == value_shape:
                adapted[key] = value
                continue

            replacement = _copy_within_shape(curr_val, value)
            if replacement is not None:
                adapted[key] = replacement
                resized.append(f"{key}: {tuple(value_shape)} -> {tuple(curr_shape)}")

        res = module.load_state_dict(adapted, strict=False)
        miss = getattr(res, "missing_keys", [])
        unexp = getattr(res, "unexpected_keys", [])
        if miss or unexp:
            logger.warning("Filtered load: missing=%s unexpected=%s", miss, unexp)
        if resized:
            logger.warning(
                "Resized checkpoint tensors to match module: %s",
                "; ".join(resized),
            )
        return res
    except Exception:
        logger.exception("Forgiving load: shape-filtered load also failed; skipping.")
        return None
    
def resolve_ckpt_path(primary: str | None,
                      ckpt_dir: str | None = None,
                      default_name: str = "head.pt") -> str:
    """
    Prefer `primary` if it exists; otherwise fall back to `<ckpt_dir>/<default_name>`.
    Raise FileNotFoundError with a clear message if neither exists.
    """
    cand = None
    if primary and os.path.isfile(primary):
        cand = primary
    elif ckpt_dir:
        fb = os.path.join(ckpt_dir, default_name)
        if os.path.isfile(fb):
            cand = fb
    if not cand:
        raise FileNotFoundError(f"Checkpoint not found: {primary!r} (fallback: {ckpt_dir}/{default_name})")
    return cand

def safe_load_checkpoint(primary: str | None,
                         ckpt_dir: str | None = None,
                         default_name: str = "head.pt",
                         map_location="cpu",
                         allow_missing: bool = True):
    """
    Try to load a checkpoint from a list of candidates:
      1) primary
      2) <ckpt_dir>/<default_name>
    Returns (state, path). If nothing is readable and allow_missing=True,
    returns ({}, None) and logs a warning (used by smoke/tests).
    """
    candidates = []
    if primary:
        candidates.append(primary)
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, default_name))

    errors = []
    for p in candidates:
        try:
            if p and os.path.isfile(p):
                state = torch.load(p, map_location=map_location)
                return state, p
        except Exception as e:
            errors.append((p, str(e)))
            continue

    # Nothing loaded:
    msg = " / ".join(repr(c) for c in candidates) if candidates else "<none>"
    if allow_missing:
        logger.warning("Could not load checkpoint from %s; proceeding with random init (test/smoke mode).", msg)
        return {"encoder": {}}, None
    raise FileNotFoundError(f"Checkpoint not found or unreadable. Tried: {msg}. Errors: {errors}")


def safe_link_or_copy(src: str, dst: str) -> str:
    """Create dst pointing at src. Prefer symlink; fall back to hardlink, then copy. Returns mode."""
    src = os.path.abspath(src); dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.lexists(dst):
            os.remove(dst)
    except FileNotFoundError:
        pass
    try:
        os.symlink(src, dst); return "symlink"
    except Exception:
        pass
    try:
        os.link(src, dst); return "hardlink"
    except Exception:
        shutil.copy2(src, dst); return "copy"
