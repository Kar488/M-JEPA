"""Build a W&B report summarising the M-JEPA project."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    aggregate_metrics,
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


def _assemble_report(
    api: Any,
    entity: Optional[str],
    project: str,
    assets_by_section: Mapping[str, Sequence[_LoggedAsset]],
) -> Optional[str]:
    try:
        from wandb_workspaces.reports import v2 as reports_v2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("W&B Reports API unavailable: %s", exc)
        return None

    base_kwargs: Dict[str, Any] = {
        "entity": entity,
        "project": project,
        "title": "M-JEPA Project Report",
        "description": "Auto-generated summary built on "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
    }

    attempts: Sequence[Mapping[str, Any]]
    if api is None:
        attempts = ({},)
    else:
        attempts = (
            {"api": api},
            {"client": api},
            {},
        )

    errors: List[str] = []
    report = None
    for extra in attempts:
        try:
            report = reports_v2.Report(**{**base_kwargs, **extra})
            break
        except Exception as exc:  # pragma: no cover - external API dependent
            errors.append(f"{sorted(extra.keys()) or ['<none>']}: {exc}")

    if report is None:
        details = "; ".join(errors) if errors else "no attempts"
        LOGGER.warning("Failed to initialise W&B report object (%s)", details)
        return None

    blocks: List[Any] = []
    for section in REPORT_SECTIONS:
        assets = assets_by_section.get(section, [])
        if not assets:
            continue
        panels: List[Any] = []
        for asset in assets:
            try:
                if asset.kind == "table" and hasattr(reports_v2.panels, "RunTable"):
                    panels.append(
                        reports_v2.panels.RunTable(
                            run_path=asset.run_path,
                            table_key=asset.key,
                            title=asset.title,
                            caption=asset.caption,
                        )
                    )
                elif asset.kind == "image" and hasattr(reports_v2.panels, "RunImage"):
                    panels.append(
                        reports_v2.panels.RunImage(
                            run_path=asset.run_path,
                            image_key=asset.key,
                            title=asset.title,
                            caption=asset.caption,
                        )
                    )
            except Exception as exc:  # pragma: no cover - external API dependent
                LOGGER.debug("Failed to create panel for %s: %s", asset.manifest_entry, exc)
        if not panels:
            continue
        try:
            blocks.append(
                reports_v2.blocks.PanelGrid(
                    title=section,
                    panels=panels,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on API
            LOGGER.debug("Failed to build PanelGrid for section %s: %s", section, exc)

    if not blocks:
        LOGGER.warning("No report blocks were generated; skipping report upload")
        return None

    try:
        report.blocks = blocks
        report.save()
    except Exception as exc:  # pragma: no cover - depends on API
        LOGGER.warning("Failed to save W&B report: %s", exc)
        return None
    return getattr(report, "url", None)


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

    if not runs:
        LOGGER.warning("No runs were fetched from W&B; nothing to report.")
        manifest = {section: [] for section in REPORT_SECTIONS}
        _write_manifest(manifest, resolved_manifest)
        return None

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

    report_url = _assemble_report(api, entity, project, assets_by_section)
    if report_url:
        LOGGER.info("Created W&B report at %s", report_url)
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
