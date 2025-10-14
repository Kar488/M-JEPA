"""Helper utilities for working with the W&B API."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from utils.logging import DummyWandb, maybe_init_wandb

try:
    from utils.wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - optional helper when utils unavailable
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return


LOGGER = logging.getLogger(__name__)

_MIN_HTTP_TIMEOUT = 30


def resolve_wandb_http_timeout(preferred: Optional[int] = None) -> int:
    """Public helper exposing the timeout resolution for other modules."""

    return _resolve_http_timeout(preferred)


def _resolve_http_timeout(preferred: Optional[int]) -> int:
    """Resolve the timeout to use for ``wandb.Api`` calls.

    The helper favours an explicit ``WANDB_HTTP_TIMEOUT`` environment override when
    present.  When the environment does not provide a value we fall back to the
    caller's preferred timeout (``None`` indicating the default of 60 seconds) and
    enforce a 30 second floor so W&B GraphQL requests do not inherit the legacy
    19 second deadline.  Existing overrides below the floor are automatically
    raised to the minimum and a warning is emitted so callers can update their
    configuration.
    """

    env_timeout: Optional[int] = None
    raw_env = os.environ.get("WANDB_HTTP_TIMEOUT")
    if raw_env:
        try:
            env_timeout = int(raw_env)
        except ValueError:
            LOGGER.debug(
                "Ignoring non-integer WANDB_HTTP_TIMEOUT override: %r", raw_env
            )
            env_timeout = None

    if env_timeout is not None:
        if env_timeout < _MIN_HTTP_TIMEOUT:
            LOGGER.warning(
                "WANDB_HTTP_TIMEOUT=%s is below the supported floor; raising to %s to avoid GraphQL timeouts",
                env_timeout,
                _MIN_HTTP_TIMEOUT,
            )
            os.environ["WANDB_HTTP_TIMEOUT"] = str(_MIN_HTTP_TIMEOUT)
            return _MIN_HTTP_TIMEOUT
        return env_timeout

    resolved = preferred if preferred is not None else 60
    resolved = max(_MIN_HTTP_TIMEOUT, resolved)

    if not raw_env:
        os.environ.setdefault("WANDB_HTTP_TIMEOUT", str(resolved))

    return resolved


@dataclass
class RunRecord:
    """Light-weight representation of a W&B run."""

    run_id: str
    name: str
    tags: Sequence[str]
    summary: Mapping[str, Any]
    config: Mapping[str, Any]
    history: Optional[pd.DataFrame]
    group: Optional[str]
    job_type: Optional[str]
    url: Optional[str]


def _coerce_mapping(value: Any, *, context: str) -> Dict[str, Any]:
    """Best-effort conversion of W&B payloads to a mapping."""

    if isinstance(value, MappingABC):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            LOGGER.debug("Ignoring non-mapping %s from W&B API: %r", context, value)
            return {}
        if isinstance(parsed, MappingABC):
            return dict(parsed)
        LOGGER.debug(
            "Parsed %s JSON is not a mapping (type=%s): %r", context, type(parsed).__name__, parsed
        )
        return {}
    try:
        return dict(value)
    except Exception:  # pragma: no cover - defensive fallback
        LOGGER.debug("Failed to coerce %s into mapping: %r", context, value)
        return {}


def get_wandb_api(
    *, allow_missing: bool = False, project: str = "m-jepa", timeout: Optional[int] = None
):
    has_env_credentials = any(
        os.getenv(env_var)
        for env_var in ("WANDB_API_KEY", "WANDB_API_KEY_FILE", "WANDB_ANONYMOUS")
    )
    credential_message = (
        "WANDB_API_KEY is required to access the W&B API in non-interactive environments"
    )
    try:
        silence_pydantic_field_warnings()
        wandb_module = maybe_init_wandb(
            enable=True,
            project=project,
            initialise_run=False,
        )
    except Exception as exc:  # pragma: no cover - defensive; helper already logs
        if allow_missing:
            LOGGER.warning("Failed to prepare W&B API client: %s", exc)
            return None
        raise

    if isinstance(wandb_module, DummyWandb) or not hasattr(wandb_module, "Api"):
        message = "wandb is required for report generation"
        if allow_missing:
            LOGGER.warning("%s; continuing without remote data", message)
            return None
        raise RuntimeError(message)

    resolved_timeout = resolve_wandb_http_timeout(timeout)

    api_error: Optional[Exception] = None
    try:
        return wandb_module.Api(timeout=resolved_timeout)
    except TypeError as type_error:
        LOGGER.debug(
            "wandb.Api does not accept a timeout argument; retrying without explicit timeout",
            exc_info=type_error,
        )
        try:
            return wandb_module.Api()
        except Exception as fallback_error:  # pragma: no cover - external API dependent
            api_error = fallback_error
    except Exception as exc:  # pragma: no cover - depends on external API
        api_error = exc

    if api_error is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to initialise W&B API client")

    if allow_missing:
        if has_env_credentials:
            LOGGER.warning("Failed to initialise W&B API client: %s", api_error)
        else:
            LOGGER.warning("%s; continuing without remote data", credential_message)
        return None
    if not has_env_credentials:
        raise RuntimeError(credential_message) from api_error
    raise api_error


def _load_history(run, keys: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    if keys is None:
        keys = []
    try:
        history = run.history(samples=10000, pandas=True, keys=list(keys))
    except Exception as exc:  # pragma: no cover - depends on external API
        LOGGER.debug("Failed to download history for %s: %s", run.id, exc)
        return None
    if history is None or history.empty:
        return None
    history = history.dropna(axis=1, how="all")
    return history


def fetch_runs(
    entity: Optional[str],
    project: str,
    *,
    filters: Optional[Mapping[str, Any]] = None,
    max_runs: int = 1000,
    history_keys: Optional[Sequence[str]] = None,
    api: Optional[Any] = None,
) -> List[RunRecord]:
    if api is None:
        api = get_wandb_api(project=project)
    if api is None:  # pragma: no cover - defensive; only when allow_missing=True
        return []
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path, filters=filters or {}, per_page=max_runs)

    records: List[RunRecord] = []
    for run in runs:
        history = _load_history(run, history_keys)
        record = RunRecord(
            run_id=run.id,
            name=run.name,
            tags=tuple(run.tags or []),
            summary=_coerce_mapping(getattr(run, "summary", None), context="summary"),
            config=_coerce_mapping(getattr(run, "config", None), context="config"),
            history=history,
            group=getattr(run, "group", None),
            job_type=getattr(run, "job_type", None),
            url=getattr(run, "url", None),
        )
        records.append(record)
    return records


def normalise_tag(tag: str) -> str:
    return tag.replace(" ", "_").lower()


def group_runs_by_seed(
    runs: Sequence[RunRecord], seed_keys: Sequence[str]
) -> Mapping[str, List[RunRecord]]:
    grouped: MutableMapping[str, List[RunRecord]] = {}
    for run in runs:
        seed = None
        for key in seed_keys:
            if key in run.config:
                seed = str(run.config[key])
                break
        if seed is None and "seed" in run.summary:
            seed = str(run.summary["seed"])
        seed = seed or "unknown"
        grouped.setdefault(seed, []).append(run)
    return grouped


def aggregate_metrics(
    runs: Sequence[RunRecord],
    metric_keys: Sequence[str],
    *,
    seed_keys: Sequence[str] = ("seed", "global_seed"),
    reducer: str = "mean",
) -> pd.DataFrame:
    grouped = group_runs_by_seed(runs, seed_keys)
    rows: List[Dict[str, Any]] = []
    for seed, group in grouped.items():
        row: Dict[str, Any] = {"seed": seed}
        for key in metric_keys:
            values: List[float] = []
            for run in group:
                value = run.summary.get(key)
                if isinstance(value, Mapping) and "value" in value:
                    value = value["value"]
                if value is not None:
                    with contextlib.suppress(TypeError):
                        values.append(float(value))
            if not values:
                row[key] = None
                continue
            series = pd.Series(values)
            if reducer == "mean":
                row[f"{key}_mean"] = float(series.mean())
                row[f"{key}_std"] = float(series.std(ddof=0))
            elif reducer == "median":
                row[f"{key}_median"] = float(series.median())
                row[f"{key}_iqr"] = float(series.quantile(0.75) - series.quantile(0.25))
            else:
                raise ValueError(f"Unsupported reducer: {reducer}")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["seed", *metric_keys])
    return pd.DataFrame(rows)


def runs_to_table(
    runs: Sequence[RunRecord], metric_keys: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row = {
            "run_id": run.run_id,
            "name": run.name,
            "group": run.group,
            "job_type": run.job_type,
            "url": run.url,
        }
        for key in metric_keys:
            value = run.summary.get(key)
            if isinstance(value, Mapping) and "value" in value:
                value = value["value"]
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)
