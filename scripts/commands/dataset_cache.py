"""Shared helpers for materialising sweep dataset caches."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import re
from typing import Any, Callable, Dict, Optional

DATASET_CACHE_VERSION = "v1"

LogFn = Optional[Callable[[str], None]]


def resolve_env_path(path: str) -> str:
    """Expand ``${env:VAR}``, user (~) and env vars, returning an absolute path."""

    expanded = re.sub(r"\$\{env:([^}]+)\}", r"${\1}", str(path))
    return os.path.abspath(os.path.expanduser(os.path.expandvars(expanded)))


def prepare_cache_root(base_cache_dir: Optional[str], *, enabled: bool = True) -> Optional[str]:
    """Return the ``prebuilt_datasets`` directory for sweep caches."""

    if not enabled:
        return None
    base_cache = base_cache_dir or os.path.join("cache", "graphs")
    pathlib.Path(base_cache).mkdir(parents=True, exist_ok=True)
    cache_root = os.path.join(base_cache, "prebuilt_datasets")
    pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
    return cache_root


def dataset_cache_path(
    kind: str,
    payload: Dict[str, Any],
    cache_root: Optional[str],
    *,
    version: str = DATASET_CACHE_VERSION,
) -> Optional[str]:
    """Return the cache path for ``kind``+``payload`` under ``cache_root``."""

    if not cache_root:
        return None
    cache_key = {"version": version, **payload}
    digest = hashlib.sha1(
        json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_root, f"{kind}_{digest}.pkl")


def cache_exists(kind: str, payload: Dict[str, Any], cache_root: Optional[str]) -> bool:
    """Return ``True`` when the cache file for ``payload`` already exists."""

    path = dataset_cache_path(kind, payload, cache_root)
    return bool(path and os.path.exists(path))


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def _build_dataset_cache(
    kind: str,
    payload: Dict[str, Any],
    builder: Callable[[], Any],
    cache_root: Optional[str],
    *,
    force: bool = False,
    log: LogFn = None,
    load_existing: bool,
):
    log_fn = log or _default_log
    cache_path = dataset_cache_path(kind, payload, cache_root)
    if cache_path is None:
        return builder()

    if os.path.exists(cache_path) and not force:
        log_fn(f"cache hit for {kind} dataset → {cache_path}")
        if load_existing:
            try:
                with open(cache_path, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                log_fn(
                    f"failed to load {kind} cache {cache_path}: {exc}; rebuilding"
                )
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        else:
            return None
    else:
        if not os.path.exists(cache_path):
            log_fn(f"cache miss for {kind} dataset; will store at {cache_path}")
        else:
            log_fn(f"force rebuilding {kind} dataset cache at {cache_path}")

    dataset = builder()
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(dataset, fh)
        log_fn(f"cached {kind} dataset at {cache_path}")
    except Exception as exc:
        log_fn(f"failed to persist {kind} cache {cache_path}: {exc}")
    return dataset


def load_or_build_dataset(
    kind: str,
    payload: Dict[str, Any],
    builder: Callable[[], Any],
    cache_root: Optional[str],
    *,
    force: bool = False,
    log: LogFn = None,
):
    """Return the dataset, reading/writing the cache on demand."""

    return _build_dataset_cache(
        kind,
        payload,
        builder,
        cache_root,
        force=force,
        log=log,
        load_existing=True,
    )


def ensure_dataset_cache(
    kind: str,
    payload: Dict[str, Any],
    builder: Callable[[], Any],
    cache_root: Optional[str],
    *,
    force: bool = False,
    log: LogFn = None,
) -> None:
    """Materialise ``kind`` cache if needed (without loading it back)."""

    _build_dataset_cache(
        kind,
        payload,
        builder,
        cache_root,
        force=force,
        log=log,
        load_existing=False,
    )
