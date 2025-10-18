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
import inspect
import sys
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


def detect_wandb_capabilities() -> WandbCapabilities:
    """Return runtime capabilities for the W&B Reports v2 API."""

    wandb_version: Optional[str] = None
    workspaces_version: Optional[str] = None
    reports_module: Optional[Any] = None
    panels_module: Optional[Any] = None
    blocks_module: Optional[Any] = None
    has_v2_panels = False
    has_v2_blocks = False
    can_instantiate_panels = False
    can_instantiate_blocks = False

    try:  # pragma: no cover - diagnostic logging only
        import wandb  # type: ignore

        wandb_version = getattr(wandb, "__version__", None)
    except Exception as exc:  # pragma: no cover - wandb optional
        LOGGER.debug("Failed to import wandb while probing capabilities: %s", exc)
        wandb = None  # type: ignore[assignment]

    try:  # pragma: no cover - optional dependency
        import wandb_workspaces  # type: ignore

        workspaces_version = getattr(wandb_workspaces, "__version__", None)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.debug("Failed to import wandb_workspaces: %s", exc)
        wandb_workspaces = None  # type: ignore[assignment]

    try:  # pragma: no cover - optional dependency
        from wandb_workspaces.reports import v2 as reports_v2  # type: ignore

        reports_module = reports_v2
        panels_module = getattr(reports_v2, "panels", None)
        blocks_module = getattr(reports_v2, "blocks", None)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.debug("Failed to import wandb_workspaces.reports.v2: %s", exc)
        reports_module = None
        panels_module = None
        blocks_module = None

    panel_members: Optional[List[str]] = None
    block_members: Optional[List[str]] = None
    if panels_module is not None:
        try:
            panel_members = [name for name, _ in inspect.getmembers(panels_module)]
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            LOGGER.debug("Failed to inspect reports.v2.panels: %s", exc)
    if blocks_module is not None:
        try:
            block_members = [name for name, _ in inspect.getmembers(blocks_module)]
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            LOGGER.debug("Failed to inspect reports.v2.blocks: %s", exc)

    if panels_module is not None:
        has_v2_panels = all(
            hasattr(panels_module, attribute) for attribute in ("RunTable", "RunImage")
        )
        if has_v2_panels:
            try:
                panels_module.RunTable(runs=[], columns=[])  # type: ignore[call-arg]
                can_instantiate_panels = True
            except Exception as exc:  # pragma: no cover - depends on API
                LOGGER.info(
                    "[report_diag] RunTable instantiation failed: %s", exc
                )
            else:
                LOGGER.info("[report_diag] RunTable instantiation succeeded")

    block_constructor: Optional[Callable[..., Any]] = None
    if blocks_module is not None:
        for attribute in ("Markdown", "Text", "Paragraph"):
            if hasattr(blocks_module, attribute):
                has_v2_blocks = True
                block_constructor = getattr(blocks_module, attribute)
                break
        if block_constructor is not None:
            try:
                block_constructor("ok")  # type: ignore[misc]
                can_instantiate_blocks = True
            except Exception as exc:  # pragma: no cover - depends on API
                LOGGER.info("[report_diag] Markdown/Text instantiation failed: %s", exc)
            else:
                LOGGER.info("[report_diag] Markdown/Text instantiation succeeded")

    LOGGER.info(
        "[report_diag] wandb.__version__=%s wandb_workspaces.__version__=%s",
        wandb_version or "<unavailable>",
        workspaces_version or "<unavailable>",
    )
    LOGGER.info("[report_diag] sys.executable=%s", sys.executable)
    LOGGER.info("[report_diag] sys.path=%s", sys.path)
    LOGGER.info(
        "[report_diag] wandb_workspaces.reports.v2.panels members=%s",
        panel_members if panel_members is not None else "<unavailable>",
    )
    LOGGER.info(
        "[report_diag] wandb_workspaces.reports.v2.blocks members=%s",
        block_members if block_members is not None else "<unavailable>",
    )

    base_url = os.getenv("WANDB_BASE_URL", "<unset>")
    entity_env = os.getenv("WANDB_ENTITY", "<unset>")
    project_env = os.getenv("WANDB_PROJECT", "<unset>")
    LOGGER.info(
        "[report_caps] wandb=%s workspaces=%s has_panels=%s has_blocks=%s can_panels=%s can_blocks=%s base=%s entity=%s project=%s",
        wandb_version or "<unavailable>",
        workspaces_version or "<unavailable>",
        has_v2_panels,
        has_v2_blocks,
        can_instantiate_panels,
        can_instantiate_blocks,
        base_url,
        entity_env,
        project_env,
    )

    return WandbCapabilities(
        has_v2_panels=has_v2_panels,
        has_v2_blocks=has_v2_blocks,
        can_instantiate_panels=can_instantiate_panels,
        can_instantiate_blocks=can_instantiate_blocks,
        wandb_version=wandb_version,
        workspaces_version=workspaces_version,
        reports_module=reports_module,
        panels_module=panels_module,
        blocks_module=blocks_module,
    )


@dataclass(frozen=True)
class RetrySettings:
    """Retry controls for WANDB API interactions."""

    max_attempts: int = 5
    base_backoff: float = 1.0
    max_backoff: float = 30.0
    max_total_backoff: float = 180.0
    jitter: float = 0.25


@dataclass(frozen=True)
class WandbCapabilities:
    """Runtime capabilities for the Reports v2 API."""

    has_v2_panels: bool
    has_v2_blocks: bool
    can_instantiate_panels: bool
    can_instantiate_blocks: bool
    wandb_version: Optional[str]
    workspaces_version: Optional[str]
    reports_module: Optional[Any]
    panels_module: Optional[Any]
    blocks_module: Optional[Any]


@dataclass(frozen=True)
class RunFetchResult:
    """Container describing fetched runs and fetch status."""

    runs: List["RunRecord"]
    partial: bool = False


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
    settings: RetrySettings,
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
    settings: RetrySettings,
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
    retry_settings: Optional[RetrySettings] = None,
) -> RunFetchResult:
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
        A :class:`RunFetchResult` describing the downloaded runs and whether the
        pagination completed successfully. When retries are exhausted a
        :class:`WandbRetryError` is raised unless ``soft_fail`` is enabled.
    """
    if api is None:
        api = get_wandb_api(project=project)
    if api is None:  # pragma: no cover - defensive; only when allow_missing=True
        return RunFetchResult(runs=[], partial=False)

    if max_runs <= 0:
        return RunFetchResult(runs=[], partial=False)

    settings = retry_settings or RetrySettings()

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
    partial_fetch = False

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
            if soft_fail:
                partial_fetch = True
                LOGGER.warning(
                    "Failed to fetch all W&B runs after retries; continuing with %d runs (partial=%s): %s",
                    len(records),
                    partial_fetch,
                    exc,
                )
                break
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

    LOGGER.info(
        "Fetched %d runs for filters=%s (partial=%s)",
        len(records),
        applied_filters or {},
        partial_fetch,
    )
    return RunFetchResult(runs=records, partial=partial_fetch)


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
