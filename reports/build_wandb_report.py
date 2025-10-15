"""Build a W&B report summarising the M-JEPA project."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import discover_schema
from .wandb_utils import (
    REPORT_UNAVAILABLE_SENTINEL,
    WandbRetryError,
    fetch_runs,
    get_wandb_api,
)

SOFT_FAIL_ENV_VAR = "WANDB_SOFT_FAIL"


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

LOGGER = logging.getLogger(__name__)

REPORT_SECTIONS = [
    "Overview",
    "Sweeps & Ablations",
    "Pretraining Diagnostics",
    "Representation",
    "Finetuning — Regression",
    "Finetuning — Classification",
    "Tox21 Utility",
    "Method Comparison",
    "Interpretability",
    "Robustness & Reproducibility",
]


def _ensure_schema(
    root: Path, max_runs: int, schema_path: Optional[Path]
) -> discover_schema.Schema:
    default_path = root / "reports" / discover_schema.SCHEMA_FILENAME
    target_path = schema_path or default_path
    try:
        schema = discover_schema.load_schema_file(target_path)
        LOGGER.info("[ci][info] Loaded cached schema from %s", target_path)
        return schema
    except FileNotFoundError:
        LOGGER.info("[ci][info] Schema missing at %s; running discovery", target_path)
    except json.JSONDecodeError as exc:
        LOGGER.warning(
            "[ci][warn] Failed to parse schema at %s (%s); regenerating",
            target_path,
            exc,
        )
    except Exception as exc:
        LOGGER.warning(
            "[ci][warn] Unexpected error loading schema at %s: %s; regenerating",
            target_path,
            exc,
        )
    schema = discover_schema.discover_schema(root, max_runs=max_runs)
    LOGGER.info(
        "[ci][info] Discovered schema using root=%s max_runs=%s", root, max_runs
    )
    discover_schema.save_schema(schema, root)
    LOGGER.info("[ci][info] Cached schema at %s", default_path)
    if schema_path and schema_path != default_path:
        discover_schema.save_schema_to(schema, schema_path)
        LOGGER.info("[ci][info] Wrote schema copy to %s", schema_path)
    return schema


def _write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Figure Manifest",
        "",
        "This file records which artefacts populate each report panel.",
        "",
    ]
    for section, entries in manifest.items():
        lines.append(f"## {section}")
        if not entries:
            lines.append("- *(no figures yet)*")
        else:
            for entry in entries:
                lines.append(f"- {entry}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_report(
    entity: Optional[str],
    project: str,
    *,
    max_runs: int,
    refresh: bool,
    manifest_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    wandb_soft_fail: bool = False,
) -> Optional[str]:
    """Construct the report metadata and optionally upload it to W&B.

    When ``wandb_soft_fail`` is enabled and W&B remains unavailable after all
    retries, the function returns :data:`REPORT_UNAVAILABLE_SENTINEL` instead of
    raising so downstream automation can continue.
    """
    root = Path(__file__).resolve().parents[1]
    schema = _ensure_schema(root, max_runs=max_runs, schema_path=schema_path)

    try:
        api = get_wandb_api(project=project, allow_missing=True)
    except Exception as exc:  # pragma: no cover - defensive; helper logs already
        LOGGER.warning("W&B API initialisation failed: %s", exc)
        api = None

    if api is None:
        resolved_manifest = manifest_path or root / "reports" / "FIGURE_MANIFEST.md"
        _write_manifest({section: [] for section in REPORT_SECTIONS}, resolved_manifest)
        LOGGER.warning(
            "W&B API unavailable; generated empty manifest at %s instead of a report.",
            resolved_manifest,
        )
        return None

    LOGGER.info("Using project %s/%s", entity, project)
    LOGGER.info(
        "Soft-fail mode for W&B fetching is %s (env %s)",
        wandb_soft_fail,
        SOFT_FAIL_ENV_VAR,
    )

    filters: Dict[str, Any] = {}
    resolved_manifest = manifest_path or root / "reports" / "FIGURE_MANIFEST.md"
    try:
        runs = fetch_runs(
            entity,
            project,
            filters=filters,
            max_runs=max_runs,
            api=api,
            soft_fail=wandb_soft_fail,
        )
    except WandbRetryError as exc:
        if wandb_soft_fail:
            LOGGER.warning(
                "W&B runs unavailable after retries; returning %s sentinel: %s",
                REPORT_UNAVAILABLE_SENTINEL,
                exc,
            )
            _write_manifest(
                {section: [] for section in REPORT_SECTIONS}, resolved_manifest
            )
            return REPORT_UNAVAILABLE_SENTINEL
        raise
    LOGGER.info("Fetched %d runs", len(runs))

    manifest: Dict[str, Any] = {section: [] for section in REPORT_SECTIONS}

    _write_manifest(manifest, resolved_manifest)

    # The current implementation focuses on discovery.  Integrating with the W&B
    # Reports API is left for environments where credentials are available.
    LOGGER.warning("Report construction is a no-op without W&B credentials.")
    return None


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M-JEPA W&B report")
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--project", type=str, default=os.getenv("WANDB_PROJECT", "m-jepa")
    )
    parser.add_argument(
        "--refresh",
        dest="refresh",
        action="store_true",
        help="Force regeneration of report assets",
    )
    parser.add_argument(
        "--no-refresh",
        dest="refresh",
        action="store_false",
        help="Reuse previously generated artefacts when possible",
    )
    parser.set_defaults(refresh=True)
    parser.add_argument("--max-runs", type=int, default=500)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--schema-path", type=Path)
    parser.add_argument(
        "--wandb-soft-fail",
        dest="wandb_soft_fail",
        action="store_true",
        default=_env_flag(SOFT_FAIL_ENV_VAR),
        help="Return a sentinel when W&B fetching exhausts retries",
    )
    parser.add_argument(
        "--no-wandb-soft-fail",
        dest="wandb_soft_fail",
        action="store_false",
        help="Disable W&B soft-fail mode",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    url = build_report(
        args.entity,
        args.project,
        max_runs=args.max_runs,
        refresh=args.refresh,
        manifest_path=args.manifest_path,
        schema_path=args.schema_path,
        wandb_soft_fail=args.wandb_soft_fail,
    )
    if url == REPORT_UNAVAILABLE_SENTINEL:
        LOGGER.warning(
            "Report generation skipped because W&B data was unavailable and soft-fail mode is enabled."
        )
    elif url:
        print(url)
    else:
        LOGGER.info("Report generation completed without publishing a W&B URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
