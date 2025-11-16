#!/usr/bin/env python3
"""Materialise sweep dataset caches outside of Phase-1 sweeps."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _maybe_inject_repo_root() -> None:
    """Ensure the repository root is available on ``sys.path``.

    GitHub-hosted runners source ``env.sh`` so ``PYTHONPATH`` already points at
    the repository root, but self-hosted runners may execute this script directly
    via an absolute path (e.g. ``/srv/mjepa/scripts/ci/...``). In that scenario
    Python tries to resolve ``scripts`` relative to ``scripts/ci`` and fails.
    Injecting the root path mirrors the CI setup without requiring callers to
    pre-seed ``PYTHONPATH``.
    """

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_maybe_inject_repo_root()

from scripts.commands import dataset_cache

try:  # pragma: no cover - exercised in tests via monkeypatch
    from scripts.train_jepa import load_directory_dataset
except Exception:  # pragma: no cover - import fallback when run as module
    try:
        from train_jepa import load_directory_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        load_directory_dataset = None  # type: ignore
        _IMPORT_ERROR = exc
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None


def _require_loader() -> Any:
    if load_directory_dataset is None:  # pragma: no cover - guarded in tests
        raise ImportError(
            "load_directory_dataset is unavailable; ensure scripts.train_jepa imports succeed"
        ) from _IMPORT_ERROR
    return load_directory_dataset


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unlabeled-dir", required=True, help="Path to the unlabeled corpus")
    parser.add_argument("--labeled-dir", required=True, help="Path to the labeled corpus")
    parser.add_argument("--cache-dir", required=True, help="Base cache directory (same as sweeps)")
    parser.add_argument("--label-col", default=None, help="Optional label column for the labeled set")
    parser.add_argument("--sample-unlabeled", type=int, default=0, help="Optional max graphs for unlabeled set")
    parser.add_argument("--sample-labeled", type=int, default=0, help="Optional max graphs for labeled set")
    parser.add_argument("--num-workers", type=int, default=-1, help="RDKit worker pool size")
    parser.add_argument("--add-3d", action="store_true", help="Enable 3D featurisation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even when files already exist",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    return parser


def _normalized_payload(dirpath: str, add_3d: bool, sample: int, label_col: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": dirpath,
        "add_3d": bool(add_3d),
        "sample": int(sample or 0),
    }
    if label_col:
        payload["label_col"] = label_col
    return payload


def _ensure_dir(path: str, name: str) -> str:
    resolved = dataset_cache.resolve_env_path(path)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{name} directory not found: {resolved}")
    return resolved


def _log(msg: str) -> None:
    print(f"[cache-warm] {msg}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    loader = _require_loader()

    unlabeled_dir = _ensure_dir(args.unlabeled_dir, "unlabeled")
    labeled_dir = _ensure_dir(args.labeled_dir, "labeled")
    cache_dir = dataset_cache.resolve_env_path(args.cache_dir)

    cache_root = dataset_cache.prepare_cache_root(cache_dir, enabled=True)
    if not cache_root:
        raise RuntimeError("cache directory could not be prepared")

    _log(
        f"warming dataset caches at {cache_root} (add_3d={int(bool(args.add_3d))}, "
        f"force={int(bool(args.force))})"
    )

    def _build_unlabeled():
        return loader(
            dirpath=unlabeled_dir,
            label_col=None,
            add_3d=bool(args.add_3d),
            max_graphs=args.sample_unlabeled,
            num_workers=args.num_workers,
            cache_dir=cache_dir,
        )

    def _build_labeled():
        return loader(
            dirpath=labeled_dir,
            label_col=args.label_col,
            add_3d=bool(args.add_3d),
            max_graphs=args.sample_labeled,
            num_workers=args.num_workers,
            cache_dir=cache_dir,
        )

    dataset_cache.ensure_dataset_cache(
        "unlabeled",
        _normalized_payload(unlabeled_dir, args.add_3d, args.sample_unlabeled, None),
        _build_unlabeled,
        cache_root,
        force=args.force,
        log=_log,
    )
    dataset_cache.ensure_dataset_cache(
        "labeled",
        _normalized_payload(labeled_dir, args.add_3d, args.sample_labeled, args.label_col),
        _build_labeled,
        cache_root,
        force=args.force,
        log=_log,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
