"""Discover schema information for W&B runs used in the repository.

This module inspects the local repository (sweep specs, Python modules) and
optionally remote W&B runs to infer the keys that are commonly logged.  The
resulting schema is persisted to :mod:`reports/schema.json` and reused by other
reporting utilities.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set

import yaml

from .wandb_utils import resolve_wandb_http_timeout

try:
    from utils.wandb_filters import silence_pydantic_field_warnings
except Exception:  # pragma: no cover - optional dependency in offline use
    def silence_pydantic_field_warnings() -> None:  # type: ignore
        return

LOGGER = logging.getLogger(__name__)

SCHEMA_FILENAME = "schema.json"


@dataclass
class Schema:
    """Container for discovered schema details."""

    metrics: Mapping[str, Set[str]] = field(default_factory=dict)
    configs: Mapping[str, Set[str]] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    run_types: Set[str] = field(default_factory=set)
    default_entity: Optional[str] = None
    default_project: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "metrics": {k: sorted(v) for k, v in self.metrics.items()},
            "configs": {k: sorted(v) for k, v in self.configs.items()},
            "tags": sorted(self.tags),
            "run_types": sorted(self.run_types),
            "default_entity": self.default_entity,
            "default_project": self.default_project,
        }


class WandbLogVisitor(ast.NodeVisitor):
    """Collect keys passed to ``wandb.log`` from Python source files."""

    def __init__(self) -> None:
        self.keys: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (AST interface)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "log":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "wandb":
                if node.args:
                    first = node.args[0]
                    if isinstance(first, (ast.Dict, ast.Call)):
                        self._extract_from_node(first)
                for kw in node.keywords:
                    self._extract_from_node(kw.value)
        self.generic_visit(node)

    def _extract_from_node(self, node: ast.AST) -> None:
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self.keys.add(key.value)
        elif isinstance(node, ast.Call):
            # Handle ``dict(metric=value)`` style invocations.
            if isinstance(node.func, ast.Name) and node.func.id == "dict":
                for kw in node.keywords:
                    if isinstance(kw.arg, str):
                        self.keys.add(kw.arg)


def _extract_wandb_keys_from_python(paths: Iterable[Path]) -> Set[str]:
    keys: Set[str] = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        visitor = WandbLogVisitor()
        visitor.visit(tree)
        keys.update(visitor.keys)
    return keys


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _extract_sweep_config_keys(sweep_path: Path) -> Mapping[str, Set[str]]:
    configs: MutableMapping[str, Set[str]] = defaultdict(set)
    try:
        data = yaml.safe_load(sweep_path.read_text())
    except (OSError, yaml.YAMLError):
        return configs

    parameters = data.get("parameters", {}) if isinstance(data, Mapping) else {}
    for name, value in parameters.items():
        configs[name].add("sweep")
        if isinstance(value, Mapping):
            for nested_key in value.keys():
                configs[f"{name}.{nested_key}"].add("sweep")
    return configs


def _discover_entity_project(root: Path) -> Mapping[str, Optional[str]]:
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT") or os.getenv("WANDB_DEFAULT_PROJECT")

    if project is None:
        project = "m-jepa"

    if entity is None:
        readme = root / "README.md"
        if readme.exists():
            text = readme.read_text()
            match = re.search(r"wandb\.ai/([\w\-]+)/([\w\-]+)/sweeps", text)
            if match:
                entity = match.group(1)
                project = project or match.group(2)

    return {"entity": entity, "project": project}


def _normalise_summary(run_ref: Any, summary: Any) -> Optional[Mapping[str, Any]]:
    if summary is None:
        return {}
    if isinstance(summary, Mapping):
        return summary
    if isinstance(summary, str):
        try:
            parsed = json.loads(summary)
        except json.JSONDecodeError as exc:
            snippet = summary[:200].replace("\n", " ")
            LOGGER.warning(
                "[ci][warn] Unable to decode W&B summary for run %s: %s (snippet=%s)",
                getattr(run_ref, "id", getattr(run_ref, "name", "<unknown>")),
                exc,
                snippet,
            )
            return None
        if isinstance(parsed, Mapping):
            return parsed
        LOGGER.warning(
            "[ci][warn] Summary for run %s decoded to %s; expected mapping",
            getattr(run_ref, "id", getattr(run_ref, "name", "<unknown>")),
            type(parsed).__name__,
        )
        return None
    if hasattr(summary, "to_json"):
        try:
            converted = summary.to_json()
        except Exception as exc:
            LOGGER.warning(
                "[ci][warn] Failed to convert summary via to_json for run %s: %s",
                getattr(run_ref, "id", getattr(run_ref, "name", "<unknown>")),
                exc,
            )
            return None
        return _normalise_summary(run_ref, converted)
    if hasattr(summary, "to_dict"):
        try:
            converted = summary.to_dict()
        except Exception as exc:
            LOGGER.warning(
                "[ci][warn] Failed to convert summary via to_dict for run %s: %s",
                getattr(run_ref, "id", getattr(run_ref, "name", "<unknown>")),
                exc,
            )
            return None
        if isinstance(converted, Mapping):
            return converted
        return _normalise_summary(run_ref, converted)
    return None


def _collect_remote_schema(
    entity: Optional[str],
    project: Optional[str],
    *,
    max_runs: int,
) -> Mapping[str, object]:
    if project is None:
        return {}

    if not os.getenv("WANDB_API_KEY") and not os.getenv("WANDB_API_KEY_FILE"):
        return {}

    try:
        silence_pydantic_field_warnings()
        import wandb  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return {}

    timeout = resolve_wandb_http_timeout(60)

    try:
        api = wandb.Api(timeout=timeout)
    except TypeError as type_error:  # pragma: no cover - depends on wandb version
        LOGGER.debug(
            "wandb.Api does not accept a timeout argument; retrying without explicit timeout",
            exc_info=type_error,
        )
        try:
            api = wandb.Api()
        except Exception:  # pragma: no cover - requires remote API
            return {}
    except Exception:  # pragma: no cover - requires remote API
        return {}

    project_path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(project_path, per_page=max_runs)
    except Exception:
        return {}

    metrics: MutableMapping[str, Set[str]] = defaultdict(set)
    configs: MutableMapping[str, Set[str]] = defaultdict(set)
    tags: Set[str] = set()
    run_types: Set[str] = set()

    skipped_summaries: List[str] = []

    for run in runs:
        tags.update(run.tags or [])
        if run.group:
            run_types.add(run.group)
        if run.job_type:
            run_types.add(run.job_type)
        summary_obj = getattr(run, "summary", None)
        if summary_obj:
            summary_data = _normalise_summary(run, summary_obj)
            if summary_data is None:
                skipped_summaries.append(
                    str(getattr(run, "id", getattr(run, "name", "<unknown>")))
                )
                summary_data = {}
            for key in summary_data.keys():
                metrics[key].add("summary")
        config = getattr(run, "config", {}) or {}
        if isinstance(config, Mapping):
            for key in config.keys():
                configs[key].add("config")

    if skipped_summaries:
        LOGGER.warning(
            "[ci][warn] Skipped %d runs with malformed summaries: %s",
            len(skipped_summaries),
            ", ".join(skipped_summaries[:5]),
        )

    return {
        "metrics": metrics,
        "configs": configs,
        "tags": tags,
        "run_types": run_types,
    }


def discover_schema(root: Path, *, max_runs: int = 200) -> Schema:
    schema = Schema()

    python_files = list(_iter_python_files(root))
    wandb_keys = _extract_wandb_keys_from_python(python_files)
    if wandb_keys:
        schema.metrics["logged"] = set(sorted(wandb_keys))

    sweep_configs: MutableMapping[str, Set[str]] = defaultdict(set)
    for sweep in root.glob("sweeps/*.yaml"):
        sweep_data = _extract_sweep_config_keys(sweep)
        for key, value in sweep_data.items():
            sweep_configs[key].update(value)
    if sweep_configs:
        schema.configs["sweeps"] = set(sorted(sweep_configs.keys()))

    entity_project = _discover_entity_project(root)
    schema.default_entity = entity_project.get("entity")
    schema.default_project = entity_project.get("project")

    remote = _collect_remote_schema(
        schema.default_entity, schema.default_project, max_runs=max_runs
    )
    if remote:
        for key, value in remote.get("metrics", {}).items():
            schema.metrics.setdefault("remote", set()).add(key)
            for source in value:
                schema.metrics.setdefault(source, set()).add(key)
            schema.metrics.setdefault("all", set()).add(key)
        for key, value in remote.get("configs", {}).items():
            schema.configs.setdefault("remote", set()).add(key)
            for source in value:
                schema.configs.setdefault(source, set()).add(key)
            schema.configs.setdefault("all", set()).add(key)
        schema.tags.update(remote.get("tags", set()))
        schema.run_types.update(remote.get("run_types", set()))

    if not schema.metrics:
        LOGGER.warning(
            "[ci][warn] Schema discovery returned no metrics; remote summaries may be unavailable."
        )

    return schema


def save_schema_to(schema: Schema, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(schema.to_json(), indent=2, sort_keys=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    os.replace(tmp_path, path)
    return path


def save_schema(schema: Schema, root: Path) -> Path:
    reports_dir = root / "reports"
    output_path = reports_dir / SCHEMA_FILENAME
    return save_schema_to(schema, output_path)


def load_schema_file(path: Path) -> Schema:
    if not path.exists():
        raise FileNotFoundError("Schema file not found; run discover_schema first")
    data = json.loads(path.read_text())
    schema = Schema(
        metrics={k: set(v) for k, v in data.get("metrics", {}).items()},
        configs={k: set(v) for k, v in data.get("configs", {}).items()},
        tags=set(data.get("tags", [])),
        run_types=set(data.get("run_types", [])),
        default_entity=data.get("default_entity"),
        default_project=data.get("default_project"),
    )
    return schema


def load_schema(root: Path) -> Schema:
    path = root / "reports" / SCHEMA_FILENAME
    return load_schema_file(path)


def main(argv: Optional[Iterable[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Discover W&B schema for the project")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=200,
        help="Maximum number of runs to inspect remotely",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[1]
    schema = discover_schema(root, max_runs=args.max_runs)
    path = save_schema(schema, root)
    print(path)
    return path


if __name__ == "__main__":
    main()
