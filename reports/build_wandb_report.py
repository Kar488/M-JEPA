"""Build a W&B report summarising the M-JEPA project."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import os
import tempfile
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd

try:  # pragma: no cover - optional plotting dependencies
    from . import (
        plots_classification,
        plots_compare,
        plots_pretrain,
        plots_regression,
        plots_repr,
        plots_tox21,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    logging.getLogger(__name__).debug("Failed to import plotting helpers: %s", exc)
    plots_classification = plots_compare = plots_pretrain = plots_regression = plots_repr = plots_tox21 = None  # type: ignore[assignment]

from . import discover_schema
from .wandb_utils import (
    REPORT_UNAVAILABLE_SENTINEL,
    WandbRetryError,
    RunRecord,
    RunFetchResult,
    RetrySettings,
    WandbCapabilities,
    aggregate_metrics,
    detect_wandb_capabilities,
    fetch_runs,
    get_wandb_api,
    group_runs_by_seed,
    normalise_tag,
    runs_to_table,
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


SECTION_KEYWORDS: Mapping[str, Sequence[str]] = {
    "Sweeps & Ablations": ("sweep", "ablation", "grid"),
    "Pretraining Diagnostics": ("pretrain", "unsupervised", "self_supervised", "jepa"),
    "Representation": ("representation", "embedding", "umap", "tsne"),
    "Finetuning — Regression": (
        "regression",
        "esol",
        "freesolv",
        "lipo",
        "qm",
        "qm9",
    ),
    "Finetuning — Classification": (
        "classification",
        "tox",
        "roc",
        "auc",
    ),
    "Tox21 Utility": ("tox21", "tox_utility"),
    "Method Comparison": ("comparison", "baseline", "contrastive", "jepa"),
    "Interpretability": ("interpret", "explain", "attention", "saliency"),
    "Robustness & Reproducibility": (
        "robust",
        "repro",
        "seed",
        "variance",
        "stability",
    ),
}


@dataclass
class _LoggedAsset:
    """Metadata describing a figure or table logged to W&B."""

    section: str
    key: str
    run_path: str
    kind: str
    title: str
    caption: Optional[str] = None

    @property
    def manifest_entry(self) -> str:
        caption = f" – {self.caption}" if self.caption else ""
        return f"{self.kind}:{self.title} ({self.run_path}::{self.key}){caption}"


def _resolve_base_url() -> str:
    base = os.getenv("WANDB_BASE_URL")
    if base:
        return base.rstrip("/")
    return "https://wandb.ai"


def _format_run_url(run_path: str, base_url: str) -> str:
    if run_path.startswith("http://") or run_path.startswith("https://"):
        return run_path
    return f"{base_url.rstrip('/')}/{run_path.lstrip('/')}"


def _build_markdown_block(blocks_module: Any, text: str) -> Optional[Any]:
    if blocks_module is None:
        return None
    for attribute in ("Markdown", "Text", "Paragraph"):
        if not hasattr(blocks_module, attribute):
            continue
        constructor = getattr(blocks_module, attribute)
        try:
            return constructor(text)  # type: ignore[misc]
        except TypeError:
            for keyword in ("text", "content", "body", "markdown", "value"):
                try:
                    return constructor(**{keyword: text})  # type: ignore[misc]
                except TypeError:
                    continue
                except Exception as exc:  # pragma: no cover - depends on external API
                    LOGGER.debug(
                        "Failed to instantiate %s block with keyword %s: %s",
                        attribute,
                        keyword,
                        exc,
                    )
                    break
        except Exception as exc:  # pragma: no cover - depends on external API
            LOGGER.debug("Failed to instantiate %s block: %s", attribute, exc)
    LOGGER.debug("No Markdown-compatible block constructor available")
    return None


def _render_markdown_section(
    section: str,
    assets: Sequence[_LoggedAsset],
    base_url: str,
    *,
    heading_level: int = 3,
) -> str:
    heading = "#" * max(1, heading_level)
    lines = [f"{heading} {section}", ""]
    if not assets:
        lines.append("_No assets were generated for this section._")
    else:
        for asset in assets:
            run_url = _format_run_url(asset.run_path, base_url)
            title = asset.title or asset.key
            kind = asset.kind.replace("_", " ").title()
            line = (
                f"- **{kind}** `{title}` — key `{asset.key}` ([run link]({run_url}))"
            )
            lines.append(line)
            if asset.caption:
                lines.append(f"  - {asset.caption}")
    return "\n".join(lines)


def _build_header_block(
    blocks_module: Any,
    *,
    generated_at: datetime,
    partial_fetch: bool,
) -> Optional[Any]:
    lines = [
        "## M-JEPA Automation Report",
        "",
        f"Generated automatically on {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}.",
    ]
    if partial_fetch:
        lines.append(
            "\n> ⚠️ Partial data: run fetching was incomplete because WANDB_SOFT_FAIL=1."
        )
    text = "\n".join(lines)
    return _build_markdown_block(blocks_module, text)


def _instantiate_report(
    reports_module: Any,
    base_kwargs: Mapping[str, Any],
    api: Any,
) -> Optional[Any]:
    if not hasattr(reports_module, "Report"):
        LOGGER.warning("Reports module does not expose a Report class")
        return None

    attempts: List[Mapping[str, Any]] = []
    if api is not None:
        allowed_keywords: Set[str] = set()
        report_cls = reports_module.Report
        mro_candidates = getattr(report_cls, "__mro__", ())
        candidate_targets = [report_cls, getattr(report_cls, "__init__", None)]
        if mro_candidates:
            candidate_targets.extend(
                base for base in mro_candidates if base not in {None, report_cls, object}
            )

        for candidate in candidate_targets:
            if candidate is None:
                continue
            try:
                signature = inspect.signature(candidate)
            except (TypeError, ValueError):
                continue
            allowed_keywords.update(
                name
                for name in signature.parameters
                if name not in {"self", "args", "kwargs"}
            )

        keyword_targets = {report_cls, *(mro_candidates[1:] if mro_candidates else [])}
        explicit_allowlists: List[Set[str]] = []
        for target in keyword_targets:
            if target in {None, object}:
                continue
            for field_attr in (
                "model_fields",
                "fields",
                "allowed_kwargs",
                "_allowed",
            ):
                field_map = getattr(target, field_attr, None)
                if isinstance(field_map, Mapping):
                    entries = {str(entry) for entry in field_map.keys()}
                elif isinstance(field_map, Iterable) and not isinstance(field_map, (str, bytes)):
                    entries = {str(entry) for entry in field_map}
                else:
                    entries = set()

                if not entries:
                    continue

                if field_attr in {"allowed_kwargs", "_allowed"}:
                    explicit_allowlists.append(entries)
                else:
                    allowed_keywords.update(entries)

        if explicit_allowlists:
            combined_allowlist = set(explicit_allowlists[0])
            for allowlist in explicit_allowlists[1:]:
                combined_allowlist &= allowlist
            allowed_keywords = (
                allowed_keywords & combined_allowlist if allowed_keywords else combined_allowlist
            )

        candidate_kwargs: Sequence[Tuple[str, Mapping[str, Any]]] = (
            ("api", {"api": api}),
            ("client", {"client": api}),
            ("connection", {"connection": api}),
        )
        seen: Set[str] = set()
        for keyword, payload in candidate_kwargs:
            if keyword in allowed_keywords and keyword not in seen:
                attempts.append(payload)
                seen.add(keyword)

    attempts.append({})

    errors: List[str] = []
    for extra in attempts:
        try:
            return reports_module.Report(**{**base_kwargs, **extra})
        except Exception as exc:  # pragma: no cover - external API dependent
            errors.append(f"{sorted(extra.keys()) or ['<none>']}: {exc}")

    details = "; ".join(errors) if errors else "no attempts"
    LOGGER.warning("Failed to initialise W&B report object (%s)", details)
    return None


def _initialise_report_object(
    capabilities: WandbCapabilities,
    api: Any,
    entity: Optional[str],
    project: str,
    *,
    generated_at: datetime,
    partial_fetch: bool,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    reports_module = capabilities.reports_module
    if reports_module is None:
        try:
            from wandb_workspaces.reports import v2 as reports_v2  # type: ignore

            reports_module = reports_v2
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("W&B Reports API unavailable: %s", exc)
            return None, None, None

    base_kwargs: Dict[str, Any] = {
        "entity": entity,
        "project": project,
        "title": "M-JEPA Project Report",
        "description": (
            "Auto-generated summary built on "
            f"{generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ),
    }
    if partial_fetch:
        base_kwargs["description"] += (
            "\n\nPartial data: run fetching exhausted retries with WANDB_SOFT_FAIL=1."
        )

    report = _instantiate_report(reports_module, base_kwargs, api)
    if report is None:
        return None, None, None

    panels_module = capabilities.panels_module or getattr(reports_module, "panels", None)
    blocks_module = capabilities.blocks_module or getattr(reports_module, "blocks", None)
    return report, panels_module, blocks_module


def _build_report_with_panels(
    capabilities: WandbCapabilities,
    api: Any,
    entity: Optional[str],
    project: str,
    assets_by_section: Mapping[str, Sequence[_LoggedAsset]],
    *,
    generated_at: datetime,
    base_url: str,
    partial_fetch: bool,
) -> Optional[str]:
    report, panels_module, blocks_module = _initialise_report_object(
        capabilities,
        api,
        entity,
        project,
        generated_at=generated_at,
        partial_fetch=partial_fetch,
    )
    if report is None or blocks_module is None:
        return None

    blocks: List[Any] = []
    header_block = _build_header_block(
        blocks_module,
        generated_at=generated_at,
        partial_fetch=partial_fetch,
    )
    if header_block is not None:
        blocks.append(header_block)

    has_panel_grid = hasattr(blocks_module, "PanelGrid")
    has_run_table = bool(panels_module and hasattr(panels_module, "RunTable"))
    has_run_image = bool(panels_module and hasattr(panels_module, "RunImage"))

    for section in REPORT_SECTIONS:
        assets = list(assets_by_section.get(section, []))
        markdown_assets: List[_LoggedAsset] = []
        section_panels: List[Any] = []

        for asset in assets:
            if not has_panel_grid:
                markdown_assets.append(asset)
                continue
            try:
                if asset.kind == "table" and has_run_table:
                    section_panels.append(
                        panels_module.RunTable(  # type: ignore[union-attr]
                            run_path=asset.run_path,
                            table_key=asset.key,
                            title=asset.title,
                            caption=asset.caption,
                        )
                    )
                elif asset.kind == "image" and has_run_image:
                    section_panels.append(
                        panels_module.RunImage(  # type: ignore[union-attr]
                            run_path=asset.run_path,
                            image_key=asset.key,
                            title=asset.title,
                            caption=asset.caption,
                        )
                    )
                else:
                    markdown_assets.append(asset)
            except Exception as exc:  # pragma: no cover - depends on external API
                LOGGER.warning(
                    "Failed to create panel for %s: %s; falling back to Markdown",
                    asset.manifest_entry,
                    exc,
                )
                markdown_assets.append(asset)

        if section_panels and has_panel_grid:
            try:
                blocks.append(
                    blocks_module.PanelGrid(  # type: ignore[union-attr]
                        title=section,
                        panels=section_panels,
                    )
                )
            except Exception as exc:  # pragma: no cover - depends on external API
                LOGGER.warning(
                    "Failed to build PanelGrid for section %s: %s; using Markdown fallback",
                    section,
                    exc,
                )
                markdown_assets = list(assets)

        if markdown_assets or not section_panels:
            markdown_text = _render_markdown_section(
                section,
                markdown_assets or assets,
                base_url,
            )
            block = _build_markdown_block(blocks_module, markdown_text)
            if block is not None:
                blocks.append(block)

    if not blocks:
        placeholder = _build_markdown_block(
            blocks_module,
            "### Report Overview\n_No report content could be generated for the selected runs._",
        )
        if placeholder is not None:
            blocks.append(placeholder)

    try:
        report.blocks = blocks
        report.save()
    except Exception as exc:  # pragma: no cover - depends on external API
        LOGGER.warning("Failed to save W&B report: %s", exc)
        return None
    return getattr(report, "url", None)


def _build_markdown_only_report(
    capabilities: WandbCapabilities,
    api: Any,
    entity: Optional[str],
    project: str,
    assets_by_section: Mapping[str, Sequence[_LoggedAsset]],
    *,
    generated_at: datetime,
    base_url: str,
    partial_fetch: bool,
) -> Optional[str]:
    report, _, blocks_module = _initialise_report_object(
        capabilities,
        api,
        entity,
        project,
        generated_at=generated_at,
        partial_fetch=partial_fetch,
    )
    if report is None or blocks_module is None:
        return None

    blocks: List[Any] = []
    header_block = _build_header_block(
        blocks_module,
        generated_at=generated_at,
        partial_fetch=partial_fetch,
    )
    if header_block is not None:
        blocks.append(header_block)

    for section in REPORT_SECTIONS:
        markdown_text = _render_markdown_section(
            section,
            assets_by_section.get(section, []),
            base_url,
        )
        block = _build_markdown_block(blocks_module, markdown_text)
        if block is not None:
            blocks.append(block)

    if not blocks:
        return None

    try:
        report.blocks = blocks
        report.save()
    except Exception as exc:  # pragma: no cover - depends on external API
        LOGGER.warning("Failed to save Markdown-only W&B report: %s", exc)
        return None
    return getattr(report, "url", None)


def _render_static_markdown(
    assets_by_section: Mapping[str, Sequence[_LoggedAsset]],
    *,
    base_url: str,
    generated_at: datetime,
    partial_fetch: bool,
) -> str:
    lines = [
        "# M-JEPA Static Report",
        "",
        f"Generated automatically on {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}.",
        "",
    ]
    if partial_fetch:
        lines.append(
            "> ⚠️ Partial data: run fetching was incomplete because WANDB_SOFT_FAIL=1."
        )
        lines.append("")

    for section in REPORT_SECTIONS:
        markdown = _render_markdown_section(
            section,
            assets_by_section.get(section, []),
            base_url,
            heading_level=2,
        )
        lines.append(markdown)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _upload_static_report_artifact(
    api: Any,
    entity: Optional[str],
    project: str,
    assets_by_section: Mapping[str, Sequence[_LoggedAsset]],
    *,
    base_url: str,
    generated_at: datetime,
    partial_fetch: bool,
) -> Optional[str]:
    content = _render_static_markdown(
        assets_by_section,
        base_url=base_url,
        generated_at=generated_at,
        partial_fetch=partial_fetch,
    )

    timestamp = generated_at.strftime("%Y%m%d-%H%M%S")
    artifact_name = f"report-static-{timestamp}"
    file_name = f"{artifact_name}.md"

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / file_name
        file_path.write_text(content, encoding="utf-8")

        artifact = None
        artifact_kwargs = {
            "name": artifact_name,
            "type": "report-static",
            "description": "Static fallback report for M-JEPA",
            "metadata": {"partial": partial_fetch},
        }

        if hasattr(api, "artifact_type"):
            try:
                collection = api.artifact_type(  # type: ignore[call-arg]
                    type_name="report-static",
                    project=project,
                    entity=entity,
                )
                if collection is not None and hasattr(collection, "create_artifact"):
                    artifact = collection.create_artifact(**artifact_kwargs)
            except TypeError:
                try:
                    collection = api.artifact_type("report-static")  # type: ignore[call-arg]
                    if collection is not None and hasattr(collection, "create_artifact"):
                        artifact = collection.create_artifact(**artifact_kwargs)
                except Exception as exc:  # pragma: no cover - external API dependent
                    LOGGER.debug("artifact_type fallback failed: %s", exc)
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.debug("Failed to use artifact_type API: %s", exc)

        if artifact is None and hasattr(api, "create_artifact"):
            try:
                artifact = api.create_artifact(**artifact_kwargs)
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.warning("Failed to create artifact via Api.create_artifact: %s", exc)
                artifact = None

        if artifact is None:
            LOGGER.error("Unable to create W&B artifact for static report fallback")
            return None

        added = False
        if hasattr(artifact, "add_file"):
            try:
                artifact.add_file(str(file_path), name=file_name)
                added = True
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.warning("Failed to add file to artifact via add_file: %s", exc)
        if not added and hasattr(artifact, "new_file"):
            try:
                handle = artifact.new_file(file_name)  # type: ignore[call-arg]
                with open(file_path, "rb") as stream:
                    handle.write(stream.read())  # type: ignore[union-attr]
                if hasattr(handle, "close"):
                    handle.close()
                added = True
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.warning("Failed to add file to artifact via new_file: %s", exc)

        if not added:
            LOGGER.error("Static report artifact could not accept file contents")
            return None

        if hasattr(artifact, "save"):
            try:
                artifact.save()
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.debug("Artifact save reported error: %s", exc)

        target_path = f"{entity}/{project}" if entity else project
        for method_name in ("link", "attach", "use"):
            method = getattr(artifact, method_name, None)
            if method is None:
                continue
            try:
                method(target_path)  # type: ignore[misc]
                break
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.debug("Failed to %s artifact: %s", method_name, exc)

        artifact_url = (
            getattr(artifact, "url", None)
            or getattr(artifact, "versioned_url", None)
            or getattr(artifact, "download_url", None)
        )
        if not artifact_url:
            artifact_url = (
                f"{base_url.rstrip('/')}/{target_path.strip('/')}/artifacts/report-static/{artifact_name}"
            )

    return artifact_url


def _publish_report(
    capabilities: WandbCapabilities,
    api: Any,
    entity: Optional[str],
    project: str,
    assets_by_section: Mapping[str, Sequence[_LoggedAsset]],
    *,
    generated_at: datetime,
    base_url: str,
    partial_fetch: bool,
) -> Optional[str]:
    if capabilities.can_instantiate_panels and capabilities.can_instantiate_blocks:
        url = _build_report_with_panels(
            capabilities,
            api,
            entity,
            project,
            assets_by_section,
            generated_at=generated_at,
            base_url=base_url,
            partial_fetch=partial_fetch,
        )
        if url:
            return url

    if capabilities.can_instantiate_blocks:
        LOGGER.info(
            "[report_fallback] using markdown_only=True static_artifact_upload=False"
        )
        url = _build_markdown_only_report(
            capabilities,
            api,
            entity,
            project,
            assets_by_section,
            generated_at=generated_at,
            base_url=base_url,
            partial_fetch=partial_fetch,
        )
        if url:
            return url

    LOGGER.info(
        "[report_fallback] using markdown_only=False static_artifact_upload=True"
    )
    return _upload_static_report_artifact(
        api,
        entity,
        project,
        assets_by_section,
        base_url=base_url,
        generated_at=generated_at,
        partial_fetch=partial_fetch,
    )
def _flatten_schema_values(mapping: Mapping[str, Sequence[str]]) -> List[str]:
    """Return a sorted list of unique values from a schema mapping."""

    values: MutableMapping[str, None] = {}
    for entries in mapping.values():
        for entry in entries:
            values[str(entry)] = None
    return sorted(values.keys())


def _normalise_section(section: str) -> str:
    return section.lower().replace(" ", "-")


def _infer_sections_for_run(
    run: RunRecord, available_tags: Sequence[str]
) -> Sequence[str]:
    tags = {normalise_tag(tag) for tag in run.tags}
    tags.update(
        tag for tag in available_tags if tag in tags
    )  # normalise known schema tags
    job_type = (run.job_type or "").lower()
    sections: List[str] = ["Overview"]
    for section, keywords in SECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in job_type:
                sections.append(section)
                break
            if any(keyword in tag for tag in tags):
                sections.append(section)
                break
    return list(dict.fromkeys(sections))


def _group_runs_by_section(
    runs: Sequence[RunRecord], available_tags: Sequence[str]
) -> Mapping[str, List[RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = {section: [] for section in REPORT_SECTIONS}
    for run in runs:
        sections = _infer_sections_for_run(run, available_tags)
        for section in sections:
            grouped.setdefault(section, []).append(run)
    return grouped


def _select_metric_keys(
    metric_keys: Sequence[str], include_keywords: Sequence[str]
) -> List[str]:
    if not metric_keys:
        return []
    lowered = [(key, key.lower()) for key in metric_keys]
    selected = [
        original
        for original, lower in lowered
        if any(keyword in lower for keyword in include_keywords)
    ]
    if selected:
        return selected
    # Fallback to the first few metrics to avoid empty tables
    return list(metric_keys[: min(len(metric_keys), 10)])


def _extract_config_value(config: Mapping[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    if "." in key:
        head, tail = key.split(".", 1)
        nested = config.get(head)
        if isinstance(nested, MappingABC):
            return _extract_config_value(nested, tail)
    return None


def _build_config_table(
    runs: Sequence[RunRecord], config_keys: Sequence[str]
) -> pd.DataFrame:
    if not runs or not config_keys:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {"run_id": run.run_id, "name": run.name}
        for key in config_keys:
            value = _extract_config_value(run.config, key)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _is_numeric_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    try:
        [float(v) for v in value]
    except Exception:
        return False
    return True


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
    except Exception:
        return False
    return True


def _collect_summary_sequences(
    runs: Sequence[RunRecord], include_keywords: Sequence[str]
) -> Mapping[str, Sequence[float]]:
    results: Dict[str, Sequence[float]] = {}
    for run in runs:
        for key, value in run.summary.items():
            lower_key = str(key).lower()
            if any(keyword in lower_key for keyword in include_keywords) and _is_numeric_sequence(value):
                results[f"{run.run_id}:{key}"] = [float(v) for v in value]
    return results


def _collect_prediction_pairs(
    runs: Sequence[RunRecord],
    true_keywords: Sequence[str],
    pred_keywords: Sequence[str],
) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    for run in runs:
        y_true: Optional[Sequence[float]] = None
        y_pred: Optional[Sequence[float]] = None
        for key, value in run.summary.items():
            lower_key = str(key).lower()
            if y_true is None and any(keyword in lower_key for keyword in true_keywords):
                if _is_numeric_sequence(value):
                    y_true = [float(v) for v in value]
            if y_pred is None and any(keyword in lower_key for keyword in pred_keywords):
                if _is_numeric_sequence(value):
                    y_pred = [float(v) for v in value]
        if y_true and y_pred and len(y_true) == len(y_pred):
            return y_true, y_pred
    return None


def _log_assets_to_wandb(
    section: str,
    entity: Optional[str],
    project: str,
    tables: Sequence[Tuple[str, pd.DataFrame, Optional[str]]],
    figures: Sequence[Tuple[str, Any, Optional[str]]],
) -> List[_LoggedAsset]:
    if not tables and not figures:
        return []
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - wandb optional
        LOGGER.warning("Unable to log %s assets because wandb is unavailable: %s", section, exc)
        return []

    run_name = f"report-{_normalise_section(section)}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="report-assets",
        group="reports",
        reinit=True,
        allow_val_change=True,
    )

    if run is None:  # pragma: no cover - defensive guard when wandb returns None
        LOGGER.warning("wandb.init returned None for section %s", section)
        return []

    run_path = "/".join(run.path) if hasattr(run, "path") else run.id
    logged: List[_LoggedAsset] = []
    try:
        for table_name, df, caption in tables:
            if df is None or df.empty:
                continue
            table = wandb.Table(dataframe=df)
            run.log({table_name: table})
            logged.append(
                _LoggedAsset(
                    section=section,
                    key=table_name,
                    run_path=run_path,
                    kind="table",
                    title=table_name,
                    caption=caption,
                )
            )
        for fig_name, fig, caption in figures:
            if fig is None:
                continue
            image = wandb.Image(fig)
            run.log({fig_name: image})
            logged.append(
                _LoggedAsset(
                    section=section,
                    key=fig_name,
                    run_path=run_path,
                    kind="image",
                    title=fig_name,
                    caption=caption,
                )
            )
            with contextlib.suppress(Exception):
                fig.clf()
    finally:
        run.finish()
    return logged


def _build_overview_assets(
    runs: Sequence[RunRecord],
    metrics: Sequence[str],
    configs: Sequence[str],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    metric_keys = list(metrics)[: min(len(metrics), 20)]
    overview_table = runs_to_table(runs, metric_keys)
    aggregated = aggregate_metrics(runs, metric_keys)
    seed_table = pd.DataFrame(
        {"seed": list(group_runs_by_seed(runs, ("seed", "global_seed")).keys())}
    )

    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    if not overview_table.empty:
        tables.append(("overview_metrics", overview_table, "Per-run metrics"))
    if not aggregated.empty:
        tables.append(("seed_aggregates", aggregated, "Mean ± std grouped by seed"))
    if not seed_table.empty:
        tables.append(("available_seeds", seed_table, "Unique seeds observed"))

    config_subset = list(configs)[: min(len(configs), 15)]
    config_table = _build_config_table(runs, config_subset)
    if not config_table.empty:
        tables.append(("config_snapshot", config_table, "Selected configuration parameters"))

    return _log_assets_to_wandb("Overview", entity, project, tables, [])


def _build_sweep_assets(
    runs: Sequence[RunRecord],
    metrics: Sequence[str],
    configs: Sequence[str],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    if plots_pretrain is None:
        LOGGER.debug("Pretraining plotting helpers unavailable; skipping sweep figures")
        figures: List[Tuple[str, Any, Optional[str]]] = []
    else:
        figures = []
    metric_keys = _select_metric_keys(metrics, ("val", "valid", "auc", "rmse", "loss"))
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    sweep_table = _build_config_table(runs, configs[: min(len(configs), 20)])
    if not sweep_table.empty:
        tables.append(("sweep_configurations", sweep_table, "Sweep hyper-parameters"))
    summary = runs_to_table(runs, metric_keys)
    if not summary.empty:
        tables.append(("sweep_metrics", summary, "Metrics captured during sweeps"))
    aggregated = aggregate_metrics(runs, metric_keys)
    if not aggregated.empty:
        tables.append(("sweep_seed_aggregates", aggregated, "Aggregated sweep metrics"))

    if plots_pretrain is not None:
        histories = [run.history for run in runs if run.history is not None]
        if histories and metric_keys:
            metric = metric_keys[0]
            fig = plots_pretrain.plot_metric_curves(histories, metric, label=metric)
            figures.append((f"sweep_curve_{metric}", fig, f"Sweep trajectories for {metric}"))

    return _log_assets_to_wandb("Sweeps & Ablations", entity, project, tables, figures)


def _build_pretraining_assets(
    runs: Sequence[RunRecord],
    metrics: Sequence[str],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    if plots_pretrain is None:
        LOGGER.debug("Pretraining plotting helpers unavailable; skipping diagnostics")
        return []
    histories = [run.history for run in runs if run.history is not None]
    figures: List[Tuple[str, Any, Optional[str]]] = []
    if histories:
        loss_metrics = _select_metric_keys(metrics, ("loss", "info", "contrast"))
        if loss_metrics:
            metric = loss_metrics[0]
            fig = plots_pretrain.plot_metric_curves(histories, metric, label=metric)
            figures.append((f"pretrain_curve_{metric}", fig, f"Pretraining trajectories for {metric}"))

    variances = _collect_summary_sequences(runs, ("variance",))
    if variances:
        fig = plots_pretrain.plot_embedding_variance(variances)
        figures.append(("embedding_variance", fig, "Embedding variance across runs"))

    cosine = _collect_summary_sequences(runs, ("cosine", "similarity"))
    if cosine:
        fig = plots_pretrain.plot_cosine_similarity(cosine)
        figures.append(("cosine_similarity", fig, "Cosine similarity diagnostics"))

    ema = _collect_summary_sequences(runs, ("ema", "drift"))
    if ema:
        steps = list(range(len(next(iter(ema.values())))))
        fig = plots_pretrain.plot_ema_drift(steps, ema)
        figures.append(("ema_drift", fig, "EMA drift over time"))

    return _log_assets_to_wandb(
        "Pretraining Diagnostics", entity, project, [], figures
    )


def _build_representation_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    if plots_repr is None:
        LOGGER.debug("Representation plotting helpers unavailable")
        return []
    figures: List[Tuple[str, Any, Optional[str]]] = []
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    for run in runs:
        embeddings = None
        labels = None
        metadata: Dict[str, Sequence[Any]] = {}
        for key, value in run.summary.items():
            lower = str(key).lower()
            if embeddings is None and "embedding" in lower and _is_numeric_sequence(value):
                embeddings = value
            elif labels is None and ("label" in lower or "target" in lower) and _is_numeric_sequence(value):
                labels = value
            elif _is_numeric_sequence(value):
                metadata[key] = value
        if embeddings is None:
            continue
        try:
            import numpy as np

            embeddings_array = np.asarray(embeddings)
            coords = plots_repr.compute_embedding_2d(embeddings_array)
            label_seq: Sequence[Any]
            if labels is not None and len(labels) == len(coords):
                label_seq = labels
            else:
                label_seq = list(range(len(coords)))
            fig = plots_repr.plot_embedding(coords, label_seq, title=f"Embedding for {run.name or run.run_id}")
            tables.append(
                (
                    f"embedding_table_{run.run_id}",
                    plots_repr.build_embedding_table(coords, metadata),
                    f"2D embedding table for {run.name or run.run_id}",
                )
            )
            figures.append(
                (
                    f"embedding_plot_{run.run_id}",
                    fig,
                    f"UMAP/t-SNE projection for {run.name or run.run_id}",
                )
            )
        except Exception as exc:  # pragma: no cover - depends on optional deps
            LOGGER.debug("Failed to render embedding for run %s: %s", run.run_id, exc)
    return _log_assets_to_wandb("Representation", entity, project, tables, figures)


def _build_regression_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    figures: List[Tuple[str, Any, Optional[str]]] = []
    if plots_regression is None:
        LOGGER.debug("Regression plotting helpers unavailable; only tables will be logged")

    metric_keys = _select_metric_keys(
        _flatten_schema_values({"metrics": [key for run in runs for key in run.summary.keys()]})
        if runs
        else [],
        ("rmse", "mae", "r2", "pearson"),
    )
    if metric_keys:
        summary = runs_to_table(runs, metric_keys)
        if not summary.empty:
            tables.append(("regression_metrics", summary, "Regression metrics per run"))

    pair = _collect_prediction_pairs(runs, ("true", "target"), ("pred", "prediction"))
    if pair and plots_regression is not None:
        y_true, y_pred = pair
        figures.append(("regression_parity", plots_regression.parity_plot(y_true, y_pred), "Parity plot"))
        figures.append(
            (
                "regression_residuals",
                plots_regression.residual_plots(y_true, y_pred),
                "Residual diagnostics",
            )
        )

    fractions = None
    learning_metrics: Dict[str, Sequence[float]] = {}
    for run in runs:
        for key, value in run.summary.items():
            lower = str(key).lower()
            if fractions is None and "fraction" in lower and _is_numeric_sequence(value):
                fractions = [float(v) for v in value]
            elif "learning" in lower and _is_numeric_sequence(value):
                learning_metrics[key] = [float(v) for v in value]
    if fractions and learning_metrics and plots_regression is not None:
        figures.append(
            (
                "regression_learning_curve",
                plots_regression.learning_curve_plot(fractions, learning_metrics),
                "Learning curve across fractions",
            )
        )

    return _log_assets_to_wandb("Finetuning — Regression", entity, project, tables, figures)


def _build_classification_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    figures: List[Tuple[str, Any, Optional[str]]] = []
    if plots_classification is None:
        LOGGER.debug("Classification plotting helpers unavailable; only tables will be logged")

    metric_keys = _select_metric_keys(
        [key for run in runs for key in run.summary.keys()],
        ("auc", "roc", "pr", "f1", "accuracy"),
    )
    if metric_keys:
        summary = runs_to_table(runs, metric_keys)
        if not summary.empty:
            tables.append(("classification_metrics", summary, "Classification metrics"))

    pair = _collect_prediction_pairs(runs, ("label", "true", "target"), ("prob", "score", "pred"))
    if pair and plots_classification is not None:
        y_true, y_scores = pair
        try:
            from sklearn.metrics import precision_recall_curve, roc_curve

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            curves = {
                "overall": {
                    "roc": list(zip(fpr, tpr)),
                    "pr": list(zip(recall, precision)),
                }
            }
            figures.append(
                (
                    "classification_roc_pr",
                    plots_classification.plot_roc_pr_curves(curves),
                    "ROC and PR curves",
                )
            )
            reliability_fig, reliability_table = plots_classification.reliability_diagram(
                y_scores, [int(v) for v in y_true]
            )
            figures.append(("classification_reliability", reliability_fig, "Reliability diagram"))
            tables.append(("classification_reliability_table", reliability_table, "Calibration bins"))
        except Exception as exc:
            LOGGER.debug("Classification metric plotting failed: %s", exc)

    return _log_assets_to_wandb("Finetuning — Classification", entity, project, tables, figures)


def _build_tox21_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    figures: List[Tuple[str, Any, Optional[str]]] = []
    if plots_tox21 is None:
        LOGGER.debug("Tox21 plotting helpers unavailable; only tables will be logged")

    metric_keys = _select_metric_keys(
        [key for run in runs for key in run.summary.keys()],
        ("tox", "auc", "roc", "pr"),
    )
    if metric_keys:
        summary = runs_to_table(runs, metric_keys)
        if not summary.empty:
            tables.append(("tox21_metrics", summary, "Tox21 metrics"))

    for run in runs:
        ranks: Optional[Sequence[float]] = None
        labels: Optional[Sequence[int]] = None
        retained: Dict[str, Sequence[float]] = {}
        workload: Optional[Sequence[float]] = None
        avoided: Dict[str, float] = {}
        for key, value in run.summary.items():
            lower = str(key).lower()
            if ranks is None and "rank" in lower and _is_numeric_sequence(value):
                ranks = [float(v) for v in value]
            elif labels is None and "label" in lower and _is_numeric_sequence(value):
                labels = [int(v) for v in value]
            elif "retention" in lower and isinstance(value, MappingABC):
                for name, seq in value.items():
                    if _is_numeric_sequence(seq):
                        retained[str(name)] = [float(v) for v in seq]
            elif workload is None and "workload" in lower and _is_numeric_sequence(value):
                workload = [float(v) for v in value]
            elif "assay" in lower and isinstance(value, MappingABC):
                for name, val in value.items():
                    if _is_numeric(val):
                        avoided[str(name)] = float(val)
        if ranks and labels and plots_tox21 is not None:
            figures.append(
                (
                    f"tox21_enrichment_{run.run_id}",
                    plots_tox21.enrichment_curve(ranks, labels),
                    f"Enrichment curve for {run.name or run.run_id}",
                )
            )
        if retained and workload and plots_tox21 is not None:
            figures.append(
                (
                    f"tox21_retention_{run.run_id}",
                    plots_tox21.retention_vs_workload(retained, workload),
                    f"Retention vs workload for {run.name or run.run_id}",
                )
            )
        if avoided and plots_tox21 is not None:
            figures.append(
                (
                    f"tox21_assays_{run.run_id}",
                    plots_tox21.avoided_assays_bar(avoided),
                    f"Assays avoided for {run.name or run.run_id}",
                )
            )

    return _log_assets_to_wandb("Tox21 Utility", entity, project, tables, figures)


def _build_method_comparison_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    metrics = _select_metric_keys(
        [key for run in runs for key in run.summary.keys()],
        ("auc", "rmse", "accuracy", "score", "f1"),
    )
    summary = runs_to_table(runs, metrics)
    if summary.empty:
        return []
    figures: List[Tuple[str, Any, Optional[str]]] = []
    tables = [("method_comparison", summary, "Method comparison metrics")]
    if plots_compare is None:
        LOGGER.debug("Comparison plotting helpers unavailable; only tables will be logged")
    try:
        dataset_column = "dataset" if "dataset" in summary.columns else None
        if dataset_column:
            datasets = summary[dataset_column].tolist()
        else:
            datasets = [run.name or run.run_id for run in runs]
        metric_columns = [col for col in summary.columns if col not in {"run_id", "name", "group", "job_type", "url", dataset_column}]
        metric_map: Dict[str, Sequence[float]] = {}
        for column in metric_columns:
            metric_map[column] = pd.to_numeric(summary[column], errors="coerce").tolist()
        if metric_map and plots_compare is not None:
            fig_bar = plots_compare.comparison_bar(datasets, metric_map, ylabel="Metric")
            figures.append(("comparison_bar", fig_bar, "Method comparison bar chart"))
    except Exception as exc:
        LOGGER.debug("Comparison bar plotting helper unavailable: %s", exc)
    try:
        radar_metrics = {
            column: float(summary[column].mean())
            for column in summary.columns
            if column not in {"run_id", "name", "group", "job_type", "url"}
        }
        if radar_metrics and plots_compare is not None:
            fig_radar = plots_compare.radar_plot(radar_metrics)
            figures.append(("comparison_radar", fig_radar, "Method comparison radar chart"))
    except Exception as exc:
        LOGGER.debug("Comparison radar plotting helper unavailable: %s", exc)
    return _log_assets_to_wandb("Method Comparison", entity, project, tables, figures)


def _build_interpretability_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    metrics = _collect_summary_sequences(runs, ("attention", "interpret", "saliency", "umap"))
    if metrics:
        df = pd.DataFrame(
            {"metric": list(metrics.keys()), "values": [";".join(map(str, v)) for v in metrics.values()]}
        )
        tables.append(("interpretability_metrics", df, "Interpretability diagnostics"))
    return _log_assets_to_wandb("Interpretability", entity, project, tables, [])


def _build_robustness_assets(
    runs: Sequence[RunRecord],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if not runs:
        return []
    metrics = _collect_summary_sequences(runs, ("robust", "variance", "stability", "seed"))
    tables: List[Tuple[str, pd.DataFrame, Optional[str]]] = []
    if metrics:
        df = pd.DataFrame(
            {
                "metric": list(metrics.keys()),
                "values": [";".join(map(str, values)) for values in metrics.values()],
            }
        )
        tables.append(("robustness_metrics", df, "Robustness indicators"))
    return _log_assets_to_wandb("Robustness & Reproducibility", entity, project, tables, [])


def _build_section_assets(
    section: str,
    runs: Sequence[RunRecord],
    metrics: Sequence[str],
    configs: Sequence[str],
    entity: Optional[str],
    project: str,
) -> List[_LoggedAsset]:
    if section == "Overview":
        return _build_overview_assets(runs, metrics, configs, entity, project)
    if section == "Sweeps & Ablations":
        return _build_sweep_assets(runs, metrics, configs, entity, project)
    if section == "Pretraining Diagnostics":
        return _build_pretraining_assets(runs, metrics, entity, project)
    if section == "Representation":
        return _build_representation_assets(runs, entity, project)
    if section == "Finetuning — Regression":
        return _build_regression_assets(runs, entity, project)
    if section == "Finetuning — Classification":
        return _build_classification_assets(runs, entity, project)
    if section == "Tox21 Utility":
        return _build_tox21_assets(runs, entity, project)
    if section == "Method Comparison":
        return _build_method_comparison_assets(runs, entity, project)
    if section == "Interpretability":
        return _build_interpretability_assets(runs, entity, project)
    if section == "Robustness & Reproducibility":
        return _build_robustness_assets(runs, entity, project)
    return []



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
    per_page: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> Optional[str]:
    """Construct the report metadata and optionally upload it to W&B.

    When ``wandb_soft_fail`` is enabled and W&B remains unavailable after all
    retries, the function returns :data:`REPORT_UNAVAILABLE_SENTINEL` instead of
    raising so downstream automation can continue.
    """
    root = Path(__file__).resolve().parents[1]
    schema = _ensure_schema(root, max_runs=max_runs, schema_path=schema_path)

    capabilities = detect_wandb_capabilities()
    generated_at = datetime.utcnow()
    base_url = _resolve_base_url()

    resolved_per_page = 75
    if per_page is not None:
        if per_page < 1:
            LOGGER.warning(
                "Invalid per_page override %s; falling back to minimum of 1", per_page
            )
        resolved_per_page = max(1, per_page)
    else:
        per_page_override = os.getenv("REPORT_PER_PAGE")
        if per_page_override:
            try:
                resolved_per_page = max(1, int(per_page_override))
            except ValueError:
                LOGGER.warning(
                    "Invalid REPORT_PER_PAGE override %r; falling back to %s",
                    per_page_override,
                    resolved_per_page,
                )

    resolved_max_attempts = RetrySettings().max_attempts
    if max_attempts is not None:
        if max_attempts < 1:
            LOGGER.warning(
                "Invalid max_attempts override %s; falling back to minimum of 1",
                max_attempts,
            )
        resolved_max_attempts = max(1, max_attempts)
    else:
        attempts_override = os.getenv("REPORT_MAX_ATTEMPTS")
        if attempts_override:
            try:
                resolved_max_attempts = max(1, int(attempts_override))
            except ValueError:
                LOGGER.warning(
                    "Invalid REPORT_MAX_ATTEMPTS override %r; falling back to %s",
                    attempts_override,
                    resolved_max_attempts,
                )

    retry_settings = RetrySettings(max_attempts=resolved_max_attempts)

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
    LOGGER.info(
        "Report pagination configured with per_page=%s max_attempts=%s",
        resolved_per_page,
        resolved_max_attempts,
    )

    filters: Dict[str, Any] = {}
    resolved_manifest = manifest_path or root / "reports" / "FIGURE_MANIFEST.md"
    try:
        fetch_result = fetch_runs(
            entity,
            project,
            filters=filters,
            max_runs=max_runs,
            api=api,
            soft_fail=wandb_soft_fail,
            per_page=resolved_per_page,
            retry_settings=retry_settings,
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

    runs = fetch_result.runs
    partial_fetch = fetch_result.partial
    if not runs:
        LOGGER.warning(
            "No runs were fetched from W&B; the generated report will contain placeholders."
        )

    flattened_metrics = _flatten_schema_values(schema.metrics)
    flattened_configs = _flatten_schema_values(schema.configs)
    available_tags = {normalise_tag(tag) for tag in schema.tags}

    LOGGER.debug(
        "Discovered %d unique metrics, %d config keys and %d tags from schema",
        len(flattened_metrics),
        len(flattened_configs),
        len(available_tags),
    )

    section_runs = _group_runs_by_section(runs, available_tags)

    manifest: Dict[str, List[str]] = {section: [] for section in REPORT_SECTIONS}
    assets_by_section: Dict[str, List[_LoggedAsset]] = {section: [] for section in REPORT_SECTIONS}

    for section in REPORT_SECTIONS:
        section_specific_runs = section_runs.get(section, [])
        assets = _build_section_assets(
            section,
            section_specific_runs,
            flattened_metrics,
            flattened_configs,
            entity,
            project,
        )
        for asset in assets:
            manifest[section].append(asset.manifest_entry)
        assets_by_section[section].extend(assets)

    _write_manifest(manifest, resolved_manifest)

    report_url = _publish_report(
        capabilities,
        api,
        entity,
        project,
        assets_by_section,
        generated_at=generated_at,
        base_url=base_url,
        partial_fetch=partial_fetch,
    )
    if report_url:
        LOGGER.info("Report output available at %s", report_url)
    else:
        LOGGER.error("Report generation failed to produce a report or artifact URL")
    return report_url


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
        "--per-page",
        type=int,
        default=None,
        help="Number of runs to request per W&B page (overrides REPORT_PER_PAGE)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Retry attempts for W&B API pagination (overrides REPORT_MAX_ATTEMPTS)",
    )
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
        per_page=args.per_page,
        max_attempts=args.max_attempts,
    )
    if url == REPORT_UNAVAILABLE_SENTINEL:
        LOGGER.warning(
            "Report generation skipped because W&B data was unavailable and soft-fail mode is enabled."
        )
    elif url:
        print(url)
    else:
        LOGGER.error("Report generation failed: no report or artifact URL was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
