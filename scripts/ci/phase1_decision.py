"""Helpers for interpreting paired-effect results during Phase-1 selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_WINNER: Optional[str] = None
DEFAULT_TASK = "regression"

VALID_METHODS = {"jepa", "contrastive"}
VALID_TASKS = {"regression", "classification"}
VALID_DIRECTIONS = {"min", "max"}


def _normalize_choice(value: Any, choices: set[str]) -> Optional[str]:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in choices:
            return lowered
    return None


def _load_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, (str, Path)):
        return json.loads(Path(payload).read_text())
    return dict(payload)


def resolve_phase1_decision(
    payload: Any,
    *,
    default_winner: Optional[str] = DEFAULT_WINNER,
    default_task: str = DEFAULT_TASK,
    tie_tol: float = 1e-9,
) -> Tuple[str, str, bool]:
    """Return (winner, task, tie_flag) for the paired-effect payload.

    The paired-effect tool already reports a ``winner`` string, but shell callers
    need a resilient interpretation that remains correct if the JSON artifact is
    modified or missing expected keys.  This helper mirrors the core comparison
    logic while surfacing ties explicitly (``winner == "tie"``) so that callers
    can choose an appropriate follow-up policy instead of assuming a default
    method wins.
    """

    data = _load_payload(payload)
    direction = _normalize_choice(data.get("direction"), VALID_DIRECTIONS)
    winner = _normalize_choice(data.get("winner"), VALID_METHODS | {"tie"})
    task = _normalize_choice(data.get("task"), VALID_TASKS)
    mean_delta = data.get("mean_delta_contrastive_minus_jepa")

    tie = winner == "tie"
    delta = None
    if mean_delta is not None:
        try:
            delta = float(mean_delta)
        except (TypeError, ValueError):
            delta = None

    if delta is not None and abs(delta) <= tie_tol:
        tie = True
        winner = "tie"

    if winner not in VALID_METHODS:
        if delta is not None and direction in VALID_DIRECTIONS:
            if abs(delta) <= tie_tol:
                tie = True
                winner = "tie"
            elif direction == "min":
                winner = "contrastive" if delta < 0 else "jepa"
            else:
                winner = "contrastive" if delta > 0 else "jepa"
        elif default_winner in VALID_METHODS:
            winner = default_winner

    if winner is None:
        if tie:
            winner = "tie"
        else:
            raise ValueError("unable to determine phase-1 winner from payload")

    if task is None:
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
