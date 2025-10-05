"""Build a W&B report summarising the M-JEPA project."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import discover_schema
from .wandb_utils import fetch_runs, get_wandb_api

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
        return discover_schema.load_schema_file(target_path)
    except FileNotFoundError:
        LOGGER.info("Schema missing at %s; running discovery", target_path)
        schema = discover_schema.discover_schema(root, max_runs=max_runs)
        discover_schema.save_schema(schema, root)
        if schema_path and schema_path != default_path:
            discover_schema.save_schema_to(schema, schema_path)
        return schema


def _instantiate_api() -> Optional[Any]:
    try:
        return get_wandb_api()
    except RuntimeError as exc:
        LOGGER.warning("Could not initialise W&B API: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - depends on environment
        LOGGER.warning("W&B API initialisation failed: %s", exc)
        return None


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
) -> str:
    root = Path(__file__).resolve().parents[1]
    schema = _ensure_schema(root, max_runs=max_runs, schema_path=schema_path)

    api = _instantiate_api()
    if api is None:
        resolved_manifest = manifest_path or root / "reports" / "FIGURE_MANIFEST.md"
        _write_manifest({section: [] for section in REPORT_SECTIONS}, resolved_manifest)
        LOGGER.error(
            "W&B API unavailable; cannot build report. Please login with wandb.login()."
        )
        return ""

    LOGGER.info("Using project %s/%s", entity, project)

    filters: Dict[str, Any] = {}
    runs = fetch_runs(entity, project, filters=filters, max_runs=max_runs)
    LOGGER.info("Fetched %d runs", len(runs))

    manifest: Dict[str, Any] = {section: [] for section in REPORT_SECTIONS}

    resolved_manifest = manifest_path or root / "reports" / "FIGURE_MANIFEST.md"
    _write_manifest(manifest, resolved_manifest)

    # The current implementation focuses on discovery.  Integrating with the W&B
    # Reports API is left for environments where credentials are available.
    LOGGER.warning("Report construction is a no-op without W&B credentials.")
    return ""


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
    )
    if url:
        print(url)
        return 0
    LOGGER.error("Report URL unavailable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
