"""Helpers for interpreting paired-effect results during Phase-1 selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

DEFAULT_WINNER = "jepa"
DEFAULT_TASK = "regression"


def _load_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, (str, Path)):
        return json.loads(Path(payload).read_text())
    return dict(payload)


def resolve_phase1_decision(
    payload: Any,
    *,
    default_winner: str = DEFAULT_WINNER,
    default_task: str = DEFAULT_TASK,
    tie_tol: float = 1e-9,
) -> Tuple[str, str, bool]:
    """Return (winner, task, tie_flag) for the paired-effect payload.

    The paired-effect tool already reports a ``winner`` string, but shell callers
    need a resilient interpretation that remains correct if the JSON artifact is
    modified or missing expected keys.  This helper mirrors the logic from
    ``paired_effect_from_wandb.py`` while guaranteeing that ties fall back to the
    JEPA method.
    """

    data = _load_payload(payload)
    direction = data.get("direction")
    winner = data.get("winner")
    task = data.get("task")
    mean_delta = data.get("mean_delta_contrastive_minus_jepa")

    tie = False
    delta = None
    if mean_delta is not None:
        try:
            delta = float(mean_delta)
        except (TypeError, ValueError):
            delta = None

    if delta is not None and abs(delta) <= tie_tol:
        tie = True
        winner = default_winner

    if winner not in {"jepa", "contrastive"}:
        if delta is not None and direction in {"min", "max"}:
            if abs(delta) <= tie_tol:
                winner = default_winner
                tie = True
            elif direction == "min":
                winner = "contrastive" if delta < 0 else "jepa"
            else:
                winner = "contrastive" if delta > 0 else "jepa"
        else:
            winner = default_winner

    if task not in {"regression", "classification"}:
        if direction == "max":
            task = "classification"
        else:
            task = default_task

    return winner, task, tie


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description="Parse Phase-1 paired-effect output")
    ap.add_argument("path", help="Path to paired_effect.json")
    ap.add_argument(
        "--tie-tol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for considering the mean delta a tie",
    )
    args = ap.parse_args(argv)

    winner, task, tie = resolve_phase1_decision(args.path, tie_tol=args.tie_tol)
    status = "tie" if tie else "clear"
    print(f"{winner} {task} {status}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI
    raise SystemExit(main())
