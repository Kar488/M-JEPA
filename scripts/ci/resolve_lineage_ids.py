#!/usr/bin/env python3
"""Resolve the active grid and pretrain experiment identifiers."""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Iterable, Optional, Tuple


def _read_json(path: pathlib.Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_pretrain_id(state: Optional[dict]) -> Optional[str]:
    if not isinstance(state, dict):
        return None
    for key in ("exp_id", "pretrain_exp_id", "id"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _latest_phase1_spec(roots: Iterable[pathlib.Path]) -> Tuple[Optional[str], Optional[pathlib.Path]]:
    latest: Tuple[float, pathlib.Path] | None = None
    for root in roots:
        for spec in root.glob("*/grid/grid_sweep_phase2.yaml"):
            try:
                mtime = spec.stat().st_mtime
            except FileNotFoundError:
                continue
            if latest is None or mtime > latest[0]:
                latest = (mtime, spec)
    if latest is None:
        return None, None
    spec_path = latest[1]
    grid_dir = spec_path.parent
    exp_id = grid_dir.parent.name
    return exp_id or None, grid_dir


def _latest_phase1_artifact(roots: Iterable[pathlib.Path]) -> tuple[Optional[str], Optional[pathlib.Path]]:
    """Return the most recent experiment containing phase1 export artifacts.

    We look for ``phase1_runs.csv`` because it is emitted after both JEPA and
    contrastive sweeps finish, so it is a reliable signal that the grid exists
    and produced metrics worth collecting.
    """

    latest: tuple[float, pathlib.Path] | None = None
    for root in roots:
        for candidate in root.glob("*/grid/phase1_export/stage-outputs/phase1_runs.csv"):
            try:
                mtime = candidate.stat().st_mtime
            except FileNotFoundError:
                continue
            if latest is None or mtime > latest[0]:
                latest = (mtime, candidate)

    if latest is None:
        return None, None

    csv_path = latest[1]
    grid_dir = csv_path.parent.parent.parent
    exp_id = grid_dir.parent.name
    return exp_id or None, grid_dir


def _discover_lineage(
    root: pathlib.Path,
    default_id: Optional[str],
    fallback_root: Optional[pathlib.Path] = None,
) -> dict:
    root = root.expanduser().resolve()
    fallback_root = fallback_root.expanduser().resolve() if fallback_root else None
    payload: dict[str, object] = {
        "root": str(root),
        "grid_exp_id": None,
        "grid_dir": None,
        "pretrain_exp_id": None,
        "pretrain_dir": None,
        "frozen": False,
    }

    roots: list[pathlib.Path] = []
    roots.append(root)
    if fallback_root and fallback_root != root:
        roots.append(fallback_root)

    global_state = None
    for candidate_root in roots:
        global_state = _read_json(candidate_root / "pretrain_state.json")
        if global_state:
            break

    pretrain_id = _extract_pretrain_id(global_state)

    if pretrain_id:
        payload["pretrain_exp_id"] = pretrain_id
        candidate_root = root if (root / pretrain_id).exists() else roots[0]
        if fallback_root and (fallback_root / pretrain_id).exists():
            candidate_root = fallback_root

        payload["pretrain_dir"] = str(candidate_root / pretrain_id)
        marker = pathlib.Path(payload["pretrain_dir"]) / "bench" / "encoder_frozen.ok"
        payload["frozen"] = marker.exists()

    grid_id: Optional[str] = None
    grid_dir: Optional[pathlib.Path] = None

    if default_id:
        for candidate_root in roots:
            candidate_dir = candidate_root / default_id / "grid"
            if candidate_dir.is_dir():
                grid_id = default_id
                grid_dir = candidate_dir
                break
        if grid_id is None:
            grid_id = default_id
            grid_dir = root / grid_id / "grid"

    if grid_id is None:
        grid_id, grid_dir = _latest_phase1_spec(roots)
        if grid_id is None and pretrain_id:
            for candidate_root in roots:
                candidate_dir = candidate_root / pretrain_id / "grid"
                if candidate_dir.is_dir():
                    grid_id = pretrain_id
                    grid_dir = candidate_dir
                    break

    if grid_id is None:
        artifact_id, artifact_dir = _latest_phase1_artifact(roots)
        if artifact_id:
            grid_id = artifact_id
            grid_dir = artifact_dir

    if grid_id:
        payload["grid_exp_id"] = grid_id
        payload["grid_dir"] = str(grid_dir) if grid_dir else None
        if payload.get("pretrain_exp_id") is None:
            payload["pretrain_exp_id"] = grid_id
            payload["pretrain_dir"] = str((grid_dir or root / grid_id).parent)
    elif default_id:
        payload["grid_exp_id"] = default_id
        payload["grid_dir"] = str(root / default_id / "grid")
        if payload.get("pretrain_exp_id") is None:
            payload["pretrain_exp_id"] = default_id
            payload["pretrain_dir"] = str(root / default_id)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("EXPERIMENTS_ROOT", "/data/mjepa/experiments"))
    parser.add_argument("--fallback-root", dest="fallback_root", default=os.environ.get("CACHE_ROOT"))
    parser.add_argument("--default-id", dest="default_id", default=os.environ.get("RUN_ID"))
    args = parser.parse_args()

    root = pathlib.Path(args.root).expanduser().resolve()
    fallback_root = args.fallback_root
    if fallback_root is None:
        fallback_root = root.parent / "cache"
    payload = _discover_lineage(root, args.default_id, pathlib.Path(fallback_root).expanduser().resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
