"""Helper utilities for working with the W&B API."""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import random
import time
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar

import pandas as pd

from utils.logging import DummyWandb, maybe_init_wandb

try:
    from utils.wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - optional helper when utils unavailable
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return


try:  # pragma: no cover - wandb is optional in some environments
    from wandb.errors import CommError as WandbCommError
except Exception:  # pragma: no cover - fallback when wandb is unavailable
    WandbCommError = Exception  # type: ignore[assignment]

try:
    from requests import exceptions as requests_exceptions
except Exception:  # pragma: no cover - requests may be optional in some contexts
    requests_exceptions = None  # type: ignore


T = TypeVar("T")


LOGGER = logging.getLogger(__name__)

_MIN_HTTP_TIMEOUT = 30

REPORT_UNAVAILABLE_SENTINEL = "report unavailable"


@dataclass(frozen=True)
class _RetrySettings:
    """Retry controls for WANDB API interactions."""

    max_attempts: int = 5
    base_backoff: float = 1.0
    max_backoff: float = 30.0
    max_total_backoff: float = 180.0
    jitter: float = 0.25


class WandbRetryError(RuntimeError):
    """Raised when a W&B request exhausts transient retries."""



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


def _classify_transient_error(exc: Exception) -> Tuple[bool, Optional[float]]:
    """Return whether ``exc`` is transient and an optional explicit backoff."""

    retry_after: Optional[float] = None

    if isinstance(exc, WandbCommError):
        return True, None

    if requests_exceptions is None:
        return False, None

    if isinstance(exc, requests_exceptions.Timeout):
        return True, None
    if isinstance(exc, requests_exceptions.ConnectionError):
        return True, None
    if isinstance(exc, requests_exceptions.HTTPError):
        status_code: Optional[int] = None
        if exc.response is not None:
            with contextlib.suppress(Exception):
                status_code = exc.response.status_code
            if status_code == 429:
                retry_header = exc.response.headers.get("Retry-After")
                if retry_header:
                    with contextlib.suppress(ValueError):
                        retry_after = float(retry_header)
        if status_code is None:
            return True, retry_after
        if status_code >= 500:
            return True, retry_after
        if status_code in {408}:
            return True, retry_after
        if status_code in {400, 401, 403, 404}:
            return False, None
        if status_code == 429:
            return True, retry_after
    return False, None


def _retry_wandb_call(
    operation: Callable[[], T],
    *,
    context: str,
    settings: _RetrySettings,
) -> T:
    """Execute ``operation`` with retries for recognised transient failures."""

    attempt = 1
    total_backoff = 0.0
    last_exc: Optional[Exception] = None

    while attempt <= settings.max_attempts:
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - depends on external API
            should_retry, explicit_backoff = _classify_transient_error(exc)
            if not should_retry:
                LOGGER.error("Non-retriable error while %s: %s", context, exc)
                raise

            last_exc = exc
            backoff = min(settings.max_backoff, settings.base_backoff * (2 ** (attempt - 1)))
            if explicit_backoff is not None:
                backoff = max(backoff, explicit_backoff)
            jitter = random.uniform(0, settings.jitter)
            sleep_for = backoff + jitter
            if total_backoff + sleep_for > settings.max_total_backoff:
                remaining = max(0.0, settings.max_total_backoff - total_backoff)
                sleep_for = remaining

            if attempt == settings.max_attempts or sleep_for <= 0:
                break

            LOGGER.warning(
                "Transient W&B error during %s (attempt %s/%s, %s): %s; retrying in %.1fs",
                context,
                attempt,
                settings.max_attempts,
                exc.__class__.__name__,
                exc,
                sleep_for,
            )

            time.sleep(sleep_for)
            total_backoff += sleep_for
            attempt += 1

    message = (
        f"Exhausted transient retries while {context}"
        if last_exc is None
        else f"Exhausted transient retries while {context}: {last_exc}"
    )
    raise WandbRetryError(message) from last_exc


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


def _load_history(
    run,
    keys: Optional[Sequence[str]] = None,
    *,
    settings: _RetrySettings,
) -> Optional[pd.DataFrame]:
    if keys is None:
        keys = []
    try:
        history = _retry_wandb_call(
            lambda: run.history(samples=10000, pandas=True, keys=list(keys)),
            context=f"loading history for {getattr(run, 'id', 'unknown run')}",
            settings=settings,
        )
    except WandbRetryError as exc:  # pragma: no cover - depends on external API
        LOGGER.warning(
            "Skipping history for %s after retries: %s",
            getattr(run, "id", "unknown run"),
            exc,
        )
        return None
    except Exception as exc:  # pragma: no cover - depends on external API
        LOGGER.debug("Failed to download history for %s: %s", getattr(run, "id", "unknown run"), exc)
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
    per_page: int = 75,
    soft_fail: bool = False,
    retry_settings: Optional[_RetrySettings] = None,
) -> List[RunRecord]:
    """Download runs for the given project with retry-aware pagination.

    Args:
        entity: Optional W&B entity (organisation or user).
        project: W&B project name.
        filters: Raw W&B run filters.
        max_runs: Maximum number of runs to materialise locally.
        history_keys: Optional keys to extract from ``run.history``.
        api: Pre-initialised ``wandb.Api`` instance.
        per_page: Number of runs fetched per W&B page (smaller values reduce
            payload sizes for flaky networks).
        soft_fail: Whether callers are prepared to handle transient failures via
            :class:`WandbRetryError` and continue without remote data.
        retry_settings: Optional override for retry configuration.

    Returns:
        A list of :class:`RunRecord` instances.  When retries are exhausted a
        :class:`WandbRetryError` is raised; callers can use the
        :data:`REPORT_UNAVAILABLE_SENTINEL` to signal soft-fail behaviour.
    """
    if api is None:
        api = get_wandb_api(project=project)
    if api is None:  # pragma: no cover - defensive; only when allow_missing=True
        return []

    if max_runs <= 0:
        return []

    settings = retry_settings or _RetrySettings()

    per_page = max(1, min(per_page, max_runs))
    LOGGER.info(
        "Fetching W&B runs with per_page=%s max_runs=%s max_attempts=%s soft_fail=%s",
        per_page,
        max_runs,
        settings.max_attempts,
        soft_fail,
    )

    project_path = f"{entity}/{project}" if entity else project
    applied_filters = dict(filters or {})

    records: List[RunRecord] = []
    fetched = 0

    while len(records) < max_runs:
        offset = fetched

        def load_page() -> List[Any]:  # pragma: no cover - depends on external API
            run_iterable = api.runs(
                project_path,
                filters=applied_filters,
                per_page=per_page,
            )
            return list(itertools.islice(run_iterable, offset, offset + per_page))

        try:
            page = _retry_wandb_call(
                load_page,
                context=f"fetching runs page offset={offset}",
                settings=settings,
            )
        except WandbRetryError as exc:
            LOGGER.error(
                "Failed to fetch W&B runs after retries (soft_fail=%s): %s",
                soft_fail,
                exc,
            )
            raise

        if not page:
            break

        fetched += len(page)
        for run in page:
            run_id = getattr(run, "id", None)
            if not run_id:
                continue
            history = _load_history(run, history_keys, settings=settings)
            record = RunRecord(
                run_id=run_id,
                name=getattr(run, "name", ""),
                tags=tuple(getattr(run, "tags", []) or []),
                summary=_coerce_mapping(getattr(run, "summary", None), context="summary"),
                config=_coerce_mapping(getattr(run, "config", None), context="config"),
                history=history,
                group=getattr(run, "group", None),
                job_type=getattr(run, "job_type", None),
                url=getattr(run, "url", None),
            )
            records.append(record)
            if len(records) >= max_runs:
                break

        if len(page) < per_page:
            break

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
