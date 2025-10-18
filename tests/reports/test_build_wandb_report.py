"""Tests covering report publication fallbacks."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from typing import Any, Dict

fake_pandas = types.ModuleType("pandas")
fake_pandas.DataFrame = lambda *args, **kwargs: None
fake_pandas.Series = lambda *args, **kwargs: None
sys.modules.setdefault("pandas", fake_pandas)

fake_yaml = types.SimpleNamespace(
    safe_load=lambda *args, **kwargs: {},
    safe_dump=lambda *args, **kwargs: "",
    dump=lambda *args, **kwargs: "",
)
sys.modules.setdefault("yaml", fake_yaml)

import pytest

from reports import wandb_utils
import reports.build_wandb_report as build


def _install_report_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_panels: bool,
    has_blocks: bool,
) -> wandb_utils.WandbCapabilities:
    """Install stub ``wandb_workspaces`` modules with selective capabilities."""

    if has_panels:
        class RunTable:  # pragma: no cover - trivial stub
            def __init__(self, runs=None, columns=None, **_: Any) -> None:
                self.runs = runs
                self.columns = columns

        class RunImage:  # pragma: no cover - trivial stub
            def __init__(self, run_path=None, image_key=None, **_: Any) -> None:
                self.run_path = run_path
                self.image_key = image_key

        panels_mod = types.SimpleNamespace(RunTable=RunTable, RunImage=RunImage)
    else:
        panels_mod = types.SimpleNamespace()

    if has_blocks:
        class Markdown:  # pragma: no cover - trivial stub
            def __init__(self, text: str | None = None, **kwargs: Any) -> None:
                self.text = text or kwargs.get("text")

        blocks_mod = types.SimpleNamespace(Markdown=Markdown)
    else:
        blocks_mod = types.SimpleNamespace()

    class StubReport:  # pragma: no cover - trivial stub
        def __init__(self, **_: Any) -> None:
            self.url = "https://wandb.test/report"

        def save(self) -> None:
            return None

    reports_v2 = types.SimpleNamespace(
        Report=StubReport,
        panels=panels_mod,
        blocks=blocks_mod,
    )
    reports_module = types.SimpleNamespace(v2=reports_v2, __version__="0.0-test")

    monkeypatch.setitem(
        sys.modules,
        "wandb_workspaces",
        types.SimpleNamespace(reports=reports_module, __version__="0.0-test"),
    )
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports", reports_module)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports.v2", reports_v2)

    return wandb_utils.detect_wandb_capabilities()


def _empty_assets() -> Dict[str, list[Any]]:
    return {section: [] for section in build.REPORT_SECTIONS}


def test_publish_report_prefers_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = _install_report_stubs(monkeypatch, has_panels=True, has_blocks=True)

    called = {"panels": 0, "markdown": 0, "artifact": 0}

    def panels_stub(*args: Any, **kwargs: Any) -> str:
        called["panels"] += 1
        return "https://wandb.test/panels"

    def markdown_stub(*args: Any, **kwargs: Any) -> str:
        called["markdown"] += 1
        return "https://wandb.test/markdown"

    def artifact_stub(*args: Any, **kwargs: Any) -> str:
        called["artifact"] += 1
        return "https://wandb.test/artifact"

    monkeypatch.setattr(build, "_build_report_with_panels", panels_stub)
    monkeypatch.setattr(build, "_build_markdown_only_report", markdown_stub)
    monkeypatch.setattr(build, "_upload_static_report_artifact", artifact_stub)

    url = build._publish_report(  # pylint: disable=protected-access
        capabilities,
        api=object(),
        entity="entity",
        project="project",
        assets_by_section=_empty_assets(),
        generated_at=datetime.utcnow(),
        base_url="https://wandb.test",
        partial_fetch=False,
    )

    assert url == "https://wandb.test/panels"
    assert called == {"panels": 1, "markdown": 0, "artifact": 0}


def test_publish_report_falls_back_to_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = _install_report_stubs(monkeypatch, has_panels=False, has_blocks=True)

    def panels_stub(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("panel path should not run when panels are unavailable")

    called = {"markdown": 0, "artifact": 0}

    def markdown_stub(*args: Any, **kwargs: Any) -> str:
        called["markdown"] += 1
        return "https://wandb.test/markdown"

    def artifact_stub(*args: Any, **kwargs: Any) -> str:
        called["artifact"] += 1
        return "https://wandb.test/artifact"

    monkeypatch.setattr(build, "_build_report_with_panels", panels_stub)
    monkeypatch.setattr(build, "_build_markdown_only_report", markdown_stub)
    monkeypatch.setattr(build, "_upload_static_report_artifact", artifact_stub)

    url = build._publish_report(  # pylint: disable=protected-access
        capabilities,
        api=object(),
        entity="entity",
        project="project",
        assets_by_section=_empty_assets(),
        generated_at=datetime.utcnow(),
        base_url="https://wandb.test",
        partial_fetch=True,
    )

    assert url == "https://wandb.test/markdown"
    assert called == {"markdown": 1, "artifact": 0}


def test_publish_report_uses_static_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = _install_report_stubs(monkeypatch, has_panels=False, has_blocks=False)

    def panels_stub(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("panel path should not run without panels")

    def markdown_stub(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("markdown path should not run without blocks")

    called = {"artifact": 0}

    def artifact_stub(*args: Any, **kwargs: Any) -> str:
        called["artifact"] += 1
        return "https://wandb.test/artifact"

    monkeypatch.setattr(build, "_build_report_with_panels", panels_stub)
    monkeypatch.setattr(build, "_build_markdown_only_report", markdown_stub)
    monkeypatch.setattr(build, "_upload_static_report_artifact", artifact_stub)

    url = build._publish_report(  # pylint: disable=protected-access
        capabilities,
        api=object(),
        entity="entity",
        project="project",
        assets_by_section=_empty_assets(),
        generated_at=datetime.utcnow(),
        base_url="https://wandb.test",
        partial_fetch=False,
    )

    assert url == "https://wandb.test/artifact"
    assert called == {"artifact": 1}
