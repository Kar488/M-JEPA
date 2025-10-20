#!/usr/bin/env python3
"""Render best_config_args stage policy as a Markdown table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _as_list(values: Iterable[str]) -> list[str]:
    return sorted({str(v) for v in values if v is not None})


def _merge(manifest: dict, stage: str, category: str) -> list[str]:
    default = manifest.get("default", {}) or {}
    section = manifest.get(stage, {}) or {}
    merged: list[str] = []
    merged.extend(default.get(category) or [])
    merged.extend(section.get(category) or [])
    return _as_list(merged)


def _simple_yaml_load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    result: dict[str, object] = {}
    stack: list[tuple[int, object]] = [(-1, result)]
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped_comment = raw.split("#", 1)[0].rstrip()
        if not stripped_comment.strip():
            i += 1
            continue
        if stripped_comment.strip() == "---":
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        stripped = stripped_comment.strip()
        if stripped.startswith("- "):
            if not isinstance(container, list):
                raise ValueError("policy list item without list context")
            container.append(stripped[2:].strip())
            i += 1
            continue
        if ":" in stripped:
            key, remainder = stripped.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            if remainder:
                if not isinstance(container, dict):
                    raise ValueError("policy scalar outside mapping")
                container[key] = remainder
                i += 1
                continue
            lookahead = None
            j = i + 1
            while j < len(lines):
                look_raw = lines[j]
                look = look_raw.split("#", 1)[0]
                if not look.strip():
                    j += 1
                    continue
                look_indent = len(look_raw) - len(look_raw.lstrip(" "))
                if look_indent <= indent:
                    break
                look_stripped = look.strip()
                lookahead = [] if look_stripped.startswith("- ") else {}
                break
            if lookahead is None:
                lookahead = {}
            if not isinstance(container, dict):
                raise ValueError("policy mapping outside dict context")
            container[key] = lookahead
            stack.append((indent, lookahead))
            i += 1
            continue
        raise ValueError(f"unsupported policy line: {raw!r}")
    return result


def _format(values: list[str]) -> str:
    if not values:
        return "—"
    return ", ".join(f"`{value}`" for value in values)


def render_markdown(manifest: dict) -> str:
    stages = ["default"] + sorted(stage for stage in manifest if stage != "default")
    rows = [
        "| Stage | YAML-owned keys | Best-config keys | Allow either |",
        "| --- | --- | --- | --- |",
    ]
    for stage in stages:
        yaml_only = _merge(manifest, stage, "yaml_only")
        bestcfg_only = _merge(manifest, stage, "bestcfg_only")
        allow = _merge(manifest, stage, "allow")
        rows.append(
            f"| `{stage}` | {_format(yaml_only)} | {_format(bestcfg_only)} | {_format(allow)} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "ci" / "bestcfg_policy.yml",
        help="Path to bestcfg policy manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the rendered Markdown. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    if yaml is not None:
        data = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    else:
        data = _simple_yaml_load(args.manifest)
    markdown = render_markdown(data)

    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
