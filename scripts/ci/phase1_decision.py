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


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
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
    tie_tol: float = 1e-2,
) -> Tuple[str, str, bool, bool, bool]:
    """Return (winner, task, tie_flag, tie_breaker_used, primary_tied) for the payload.

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

    primary_info = data.get("primary_metric")
    primary_tied: Optional[bool] = None
    payload_tol: Optional[float] = None
    if isinstance(primary_info, dict):
        tied_val = primary_info.get("tied")
        if isinstance(tied_val, bool):
            primary_tied = tied_val
        tol_val = primary_info.get("tolerance")
        try:
            payload_tol = float(tol_val)
        except (TypeError, ValueError):
            payload_tol = None

    tie_breaker_used = _coerce_bool(data.get("tie_breaker_used")) or False

    delta = None
    if mean_delta is not None:
        try:
            delta = float(mean_delta)
        except (TypeError, ValueError):
            delta = None

    effective_tol = payload_tol if payload_tol is not None else tie_tol
    delta_is_tie = bool(
        effective_tol is not None and delta is not None and abs(delta) <= effective_tol
    )
    tied_by_tolerance = bool(primary_tied) if isinstance(primary_tied, bool) else False
    if not tied_by_tolerance and primary_tied is None and delta_is_tie:
        tied_by_tolerance = True

    raw_winner = winner if winner in VALID_METHODS | {"tie"} else None
    tie = raw_winner == "tie"
    if raw_winner in VALID_METHODS:
        winner = raw_winner
    else:
        winner = None

    if winner is None:
        if delta is not None and direction in VALID_DIRECTIONS:
            if direction == "min":
                winner = "contrastive" if delta < 0 else "jepa"
            else:
                winner = "contrastive" if delta > 0 else "jepa"
        elif default_winner in VALID_METHODS:
            winner = default_winner
        elif tie:
            winner = "tie"
        else:
            raise ValueError("unable to determine phase-1 winner from payload")

    tie = winner == "tie"

    if task is None:
        if direction == "max":
            task = "classification"
        else:
            task = default_task

    return winner, task, tie, tie_breaker_used, tied_by_tolerance


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description="Parse Phase-1 paired-effect output")
    ap.add_argument("path", help="Path to paired_effect.json")
    ap.add_argument(
        "--tie-tol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for considering the mean delta a tie",
    )
    args = ap.parse_args(argv)

    winner, task, tie, tie_breaker_used, tied_by_tolerance = resolve_phase1_decision(
        args.path, tie_tol=args.tie_tol
    )
    if tie:
        status = "tie"
    elif tie_breaker_used:
        status = "tie-breaker"
    elif tied_by_tolerance:
        status = "tied-primary"
    else:
        status = "clear"
    print(f"{winner} {task} {status}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI
    raise SystemExit(main())
