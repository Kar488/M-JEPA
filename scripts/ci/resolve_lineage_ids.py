#!/usr/bin/env python3
"""Resolve the active grid and pretrain experiment identifiers."""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Optional, Tuple


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


def _latest_phase1_spec(root: pathlib.Path) -> Tuple[Optional[str], Optional[pathlib.Path]]:
    latest: Tuple[float, pathlib.Path] | None = None
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


def _discover_lineage(root: pathlib.Path, default_id: Optional[str]) -> dict:
    payload: dict[str, object] = {
        "root": str(root),
        "grid_exp_id": None,
        "grid_dir": None,
        "pretrain_exp_id": None,
        "pretrain_dir": None,
        "frozen": False,
    }

    global_state = _read_json(root / "pretrain_state.json")
    pretrain_id = _extract_pretrain_id(global_state)

    if pretrain_id:
        payload["pretrain_exp_id"] = pretrain_id
        payload["pretrain_dir"] = str(root / pretrain_id)
        marker = root / pretrain_id / "bench" / "encoder_frozen.ok"
        payload["frozen"] = marker.exists()

    grid_id, grid_dir = _latest_phase1_spec(root)
    if grid_id is None and pretrain_id:
        candidate_dir = root / pretrain_id / "grid"
        if candidate_dir.is_dir():
            grid_id = pretrain_id
            grid_dir = candidate_dir

    if grid_id is None and default_id:
        candidate_dir = root / default_id / "grid"
        if candidate_dir.is_dir():
            grid_id = default_id
            grid_dir = candidate_dir

    if grid_id:
        payload["grid_exp_id"] = grid_id
        payload["grid_dir"] = str(grid_dir) if grid_dir else None
        if payload.get("pretrain_exp_id") is None:
            payload["pretrain_exp_id"] = grid_id
            payload["pretrain_dir"] = str(root / grid_id)
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
    parser.add_argument("--default-id", dest="default_id", default=os.environ.get("RUN_ID"))
    args = parser.parse_args()

    root = pathlib.Path(args.root).expanduser().resolve()
    payload = _discover_lineage(root, args.default_id)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
