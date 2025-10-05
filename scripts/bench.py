from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback parser
    yaml = None  # type: ignore[assignment]


DEFAULT_RULES: Dict[str, Tuple[str, float]] = {
    "tox21": ("roc_auc", 0.65),
    "esol": ("rmse", 0.60),
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmarks.yml"


@dataclass(frozen=True)
class BenchmarkRule:
    metric: str
    threshold: float


def _normalise(key: str | None) -> str | None:
    if key is None:
        return None
    return str(key).strip().lower()


def _coerce_rule(data: Mapping[str, Any]) -> BenchmarkRule:
    metric = data.get("metric")
    threshold = data.get("threshold")
    if metric is None or threshold is None:
        raise ValueError("benchmark rule requires 'metric' and 'threshold' keys")
    try:
        threshold_val = float(threshold)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("threshold must be numeric") from exc
    return BenchmarkRule(metric=str(metric), threshold=threshold_val)


def _iter_rule_entries(block: Mapping[str, Any]) -> Iterable[Tuple[str, BenchmarkRule]]:
    # Support a compact form where the dataset-level entry defines the default rule.
    if "metric" in block and "threshold" in block:
        yield "__default__", _coerce_rule(block)

    for key, value in block.items():
        if key in {"metric", "threshold"}:
            continue
        if isinstance(value, Mapping):
            try:
                yield _normalise(key) or key, _coerce_rule(value)
            except ValueError:
                continue


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        stripped = raw_line.split("#", 1)[0]
        line = stripped.rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        key, _, value = line.lstrip().partition(":")
        key = key.strip()
        value = value.strip()
        parent = stack[-1][1] if stack else root
        if not value:
            new_block: Dict[str, Any] = {}
            parent[key] = new_block
            stack.append((indent, new_block))
            continue
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            parsed: Any = value[1:-1]
        else:
            lower = value.lower()
            if lower in {"true", "false"}:
                parsed = lower == "true"
            else:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        parent[key] = parsed
    return root


def load_rules(config_path: Path | None = None) -> Dict[str, Dict[str, BenchmarkRule]]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            raw = yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover - file level failure
            raise RuntimeError(f"failed to read benchmark config: {path}") from exc
    else:
        raw = _parse_simple_yaml(text)

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark config must map dataset names to tasks")

    rules: Dict[str, Dict[str, BenchmarkRule]] = {}
    for dataset, block in raw.items():
        if not isinstance(block, Mapping):
            continue
        dataset_key = _normalise(dataset)
        if dataset_key is None:
            continue
        entries: Dict[str, BenchmarkRule] = {}
        for task_key, rule in _iter_rule_entries(block):
            entries[_normalise(task_key) or task_key] = rule
        if entries:
            rules[dataset_key] = entries
    return rules


def resolve_metric_threshold(
    dataset: str,
    task: str | None = None,
    *,
    config_path: Path | None = None,
) -> BenchmarkRule:
    dataset_key = _normalise(dataset)
    if dataset_key is None:
        raise KeyError("dataset name is required")

    rules = load_rules(config_path)
    dataset_rules = rules.get(dataset_key)
    task_key = _normalise(task)

    search_keys = []
    if task_key is not None:
        search_keys.append(task_key)
    search_keys.extend(["__default__", "default"])

    if dataset_rules:
        for key in search_keys:
            rule = dataset_rules.get(key)
            if rule is not None:
                return rule

    default_rule = DEFAULT_RULES.get(dataset_key)
    if default_rule:
        metric, threshold = default_rule
        return BenchmarkRule(metric=metric, threshold=threshold)

    raise KeyError(f"No benchmark rule defined for dataset={dataset!r} task={task!r}")


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve benchmark metric/threshold for a dataset/task",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. tox21, esol)")
    parser.add_argument("--task", help="Optional task identifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override path to benchmarks.yml",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON response",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _cli()
    args = parser.parse_args(argv)

    try:
        rule = resolve_metric_threshold(
            args.dataset,
            args.task,
            config_path=args.config,
        )
    except KeyError as exc:
        parser.exit(status=2, message=f"{exc}\n")

    payload = {
        "dataset": args.dataset,
        "task": args.task,
        "metric": rule.metric,
        "threshold": rule.threshold,
    }
    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=bool(indent)))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
