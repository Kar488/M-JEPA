from __future__ import annotations

import hashlib
import logging
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


logger = logging.getLogger(__name__)


def _value_to_bytes(value: Any) -> bytes:
    """Convert ``value`` into a deterministic byte representation for hashing."""

    if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[arg-type]
        try:
            tensor = value.detach().cpu().contiguous()
            return tensor.numpy().tobytes()
        except Exception:  # pragma: no cover - defensive path
            return repr(value).encode("utf-8", errors="ignore")

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    if isinstance(value, str):
        return value.encode("utf-8")

    if isinstance(value, (int, float, bool)):
        return str(value).encode("utf-8")

    if isinstance(value, Mapping):
        chunks = []
        for key in sorted(value.keys(), key=str):
            chunks.append(str(key).encode("utf-8"))
            chunks.append(_value_to_bytes(value[key]))
        return b"".join(chunks)

    if isinstance(value, Iterable):
        return b"".join(_value_to_bytes(item) for item in value)

    return repr(value).encode("utf-8", errors="ignore")


def compute_state_dict_hash(state: Mapping[str, Any]) -> str:
    """Return a SHA-256 hash for ``state`` using sorted keys and tensor bytes."""

    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for checkpoint hashing")

    digest = hashlib.sha256()
    for key in sorted(state.keys(), key=str):
        digest.update(str(key).encode("utf-8"))
        try:
            digest.update(_value_to_bytes(state[key]))
        except Exception:  # pragma: no cover - defensive
            digest.update(repr(state[key]).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def extract_encoder_hash(state: Mapping[str, Any] | None) -> Optional[str]:
    """Extract an encoder hash from a checkpoint payload when available."""

    if not isinstance(state, Mapping):
        return None

    hash_value = state.get("encoder_hash")
    if isinstance(hash_value, str) and hash_value:
        return hash_value

    hashes = state.get("hashes") if isinstance(state, Mapping) else None
    if isinstance(hashes, Mapping):
        candidate = hashes.get("encoder")
        if isinstance(candidate, str) and candidate:
            return candidate

    return None


def save_checkpoint(path: str, **states: Any) -> None:
    """Save arbitrary state dicts (e.g., encoder=..., optimizer=..., epoch=int)."""

    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for checkpointing")

    payload = dict(states)
    encoder_state = payload.get("encoder")
    if isinstance(encoder_state, Mapping) and encoder_state:
        try:
            encoder_hash = compute_state_dict_hash(encoder_state)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to hash encoder state for %s", path)
        else:
            payload["encoder_hash"] = encoder_hash
            hashes_bucket = payload.get("hashes") if isinstance(payload.get("hashes"), dict) else {}
            hashes_bucket["encoder"] = encoder_hash
            payload["hashes"] = hashes_bucket
            logger.info(
                "[encoder_hash]=%s source=export path=%s keys=%d",
                encoder_hash,
                os.path.abspath(path),
                len(encoder_state),
            )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for checkpointing")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


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


def _prepare_state_dict_for_module(module, state_dict: Mapping[str, Any]):
    """Return a copy of ``state_dict`` aligned with ``module.state_dict()`` shapes."""

    if not hasattr(module, "state_dict"):
        return None, [], []

    try:
        current = module.state_dict()
    except Exception:
        logger.exception(
            "Could not inspect module %s state_dict(); falling back to raw load.",
            type(module).__name__,
        )
        return None, [], []

    if not isinstance(current, Mapping):
        logger.warning(
            "Module %s state_dict() did not return a Mapping; skipping shape alignment.",
            type(module).__name__,
        )
        return None, [], []

    try:
        prepared = state_dict.copy()
    except Exception:
        prepared = dict(state_dict)

    resized: list[str] = []
    dropped: list[str] = []

    for key, curr_val in current.items():
        if key not in state_dict:
            continue
        incoming_val = state_dict[key]
        curr_shape = getattr(curr_val, "shape", None)
        incoming_shape = getattr(incoming_val, "shape", None)

        if curr_shape is None or incoming_shape is None:
            if curr_shape != incoming_shape:
                dropped.append(f"{key}: {type(incoming_val).__name__} -> {type(curr_val).__name__}")
                prepared.pop(key, None)
            continue

        if curr_shape == incoming_shape:
            continue

        replacement = _copy_within_shape(curr_val, incoming_val)
        if replacement is not None:
            prepared[key] = replacement
            resized.append(f"{key}: {tuple(incoming_shape)} -> {tuple(curr_shape)}")
        else:
            prepared.pop(key, None)
            dropped.append(f"{key}: {tuple(incoming_shape)} -> {tuple(curr_shape)}")

    return prepared, resized, dropped


def load_state_dict_forgiving(module, state_dict):
    """
    Best-effort weight loader:
      - Align incoming tensors to module parameter shapes before loading.
      - Gracefully skip modules without load_state_dict/state_dict (e.g., Dummy in tests).
      - Log resized/dropped tensors but avoid surfacing benign size-mismatch tracebacks.
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

    prepared, resized, dropped = _prepare_state_dict_for_module(module, state_dict)

    load_attempts: list[tuple[str, Mapping[str, Any]]] = []
    if prepared is not None:
        load_attempts.append(("aligned", prepared))
    load_attempts.append(("raw", state_dict))

    res = None
    errors: list[str] = []
    for label, payload in load_attempts:
        try:
            res = module.load_state_dict(payload, strict=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            errors.append(f"{label}: {exc}")
            logger.exception(
                "Forgiving load: module %s rejected %s state_dict payload.",
                type(module).__name__,
                label,
            )
            continue

        if label == "raw" and prepared is not None:
            logger.warning(
                "Forgiving load: falling back to raw state_dict after aligned load failed."
            )
        break

    if res is None:
        logger.error(
            "Forgiving load failed for module %s after attempts: %s",
            type(module).__name__,
            ", ".join(errors) if errors else "<none>",
        )
        return None

    miss = getattr(res, "missing_keys", [])
    unexp = getattr(res, "unexpected_keys", [])
    if miss or unexp:
        logger.warning("Non-strict load: missing=%s unexpected=%s", miss, unexp)
    if resized:
        logger.warning("Resized checkpoint tensors to match module: %s", "; ".join(resized))
    if dropped:
        logger.warning("Dropped checkpoint tensors with incompatible shapes: %s", "; ".join(dropped))
    return res


def resolve_ckpt_path(
    primary: str | None,
    ckpt_dir: str | None = None,
    default_name: str = "head.pt",
) -> str:
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
        raise FileNotFoundError(
            f"Checkpoint not found: {primary!r} (fallback: {ckpt_dir}/{default_name})"
        )
    return cand


def safe_load_checkpoint(
    primary: str | None,
    ckpt_dir: str | None = None,
    default_name: str = "head.pt",
    map_location="cpu",
    allow_missing: bool = True,
    weights_only: bool = True,
):
    """
    Try to load a checkpoint from a list of candidates:
      1) primary
      2) <ckpt_dir>/<default_name>
    Returns (state, path). If nothing is readable and allow_missing=True,
    returns ({}, None) and logs a warning (used by smoke/tests).
    """

    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for checkpointing")

    candidates = []
    if primary:
        candidates.append(primary)
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, default_name))

    errors = []
    for p in candidates:
        try:
            if p and os.path.isfile(p):
                state = torch.load(p, map_location=map_location, weights_only=weights_only)
                return state, p
        except Exception as e:
            if (
                weights_only
                and p
                and os.path.isfile(p)
                and "weights_only" in str(e).lower()
            ):
                logger.warning(
                    "Retrying checkpoint load with weights_only=False for trusted artifact: %s",
                    p,
                )
                try:
                    state = torch.load(p, map_location=map_location, weights_only=False)
                    return state, p
                except Exception as e2:
                    errors.append((p, str(e2)))
                    continue
            errors.append((p, str(e)))
            continue

    # Nothing loaded:
    msg = " / ".join(repr(c) for c in candidates) if candidates else "<none>"
    if allow_missing:
        logger.warning(
            "Could not load checkpoint from %s; proceeding with random init (test/smoke mode).",
            msg,
        )
        return {"encoder": {}}, None
    raise FileNotFoundError(f"Checkpoint not found or unreadable. Tried: {msg}. Errors: {errors}")


def safe_link_or_copy(src: str, dst: str) -> str:
    """Create dst pointing at src. Prefer symlink; fall back to hardlink, then copy. Returns mode."""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.lexists(dst):
            os.remove(dst)
    except FileNotFoundError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except Exception:
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"
