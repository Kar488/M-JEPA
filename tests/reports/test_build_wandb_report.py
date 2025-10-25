"""Tests covering report publication fallbacks."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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


def test_static_artifact_run_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[Any] = []
            self.entity = "explicit-entity"
            self.project = "project"

        def log_artifact(self, artifact: Any) -> Any:
            artifact.url = "https://wandb.test/run-artifact"
            self.logged.append(artifact)
            return artifact

    class FakeArtifact:
        def __init__(self, name: str, *, type: str, description: str, metadata: dict[str, Any]) -> None:  # noqa: A002
            self.name = name
            self.type = type
            self.description = description
            self.metadata = metadata
            self.files: list[tuple[str, str]] = []
            self.url: Optional[str] = None

        def add_file(self, path: str, name: Optional[str] = None) -> None:
            self.files.append((path, name or Path(path).name))

    class FakeWandb:
        def __init__(self) -> None:
            self.run: Optional[FakeRun] = None
            self.started = 0
            self.finished = 0

        def Artifact(self, name: str, *, type: str, description: str, metadata: dict[str, Any]) -> FakeArtifact:  # noqa: A003
            return FakeArtifact(name, type=type, description=description, metadata=metadata)

        def finish(self) -> None:
            self.finished += 1
            self.run = None

    fake_wandb = FakeWandb()

    recorded_entities: list[Any] = []

    def fake_maybe_init(
        enable: bool,
        *,
        project: str,
        entity: Any = None,
        tags: Any | None = None,
        job_type: str | None = None,
        config: dict[str, Any] | None = None,
        initialise_run: bool = True,
        **_: Any,
    ) -> Any:
        assert enable is True
        if initialise_run and fake_wandb.run is None:
            fake_wandb.run = FakeRun()
            fake_wandb.started += 1
        recorded_entities.append(entity)
        return fake_wandb

    monkeypatch.setattr(build, "maybe_init_wandb", fake_maybe_init)
    assets = _empty_assets()

    url = build._upload_static_report_artifact(  # pylint: disable=protected-access
        api=types.SimpleNamespace(),
        entity="explicit-entity",
        project="project",
        assets_by_section=assets,
        base_url="https://wandb.test",
        generated_at=datetime.utcnow(),
        partial_fetch=False,
    )

    assert url == "https://wandb.test/run-artifact"
    assert fake_wandb.started == 1
    assert fake_wandb.finished == 1
    assert fake_wandb.run is None
    assert recorded_entities == ["explicit-entity", "explicit-entity"]


def test_ensure_schema_refresh_forces_regeneration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_discover(root: Path, max_runs: int) -> str:
        calls.append("discover")
        return "fresh"

    def fake_load(path: Path) -> str:
        calls.append("load")
        return "cached"

    monkeypatch.setattr(build.discover_schema, "discover_schema", fake_discover)
    monkeypatch.setattr(build.discover_schema, "load_schema_file", fake_load)
    monkeypatch.setattr(build.discover_schema, "save_schema", lambda schema, root: None)

    schema = build._ensure_schema(tmp_path, max_runs=5, schema_path=None, refresh=True)

    assert schema == "fresh"
    assert calls == ["discover"]


def test_ensure_schema_reuses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_discover(root: Path, max_runs: int) -> str:
        calls.append("discover")
        return "fresh"

    def fake_load(path: Path) -> str:
        calls.append("load")
        return "cached"

    monkeypatch.setattr(build.discover_schema, "discover_schema", fake_discover)
    monkeypatch.setattr(build.discover_schema, "load_schema_file", fake_load)
    monkeypatch.setattr(build.discover_schema, "save_schema", lambda schema, root: None)

    schema = build._ensure_schema(tmp_path, max_runs=5, schema_path=None, refresh=False)

    assert schema == "cached"
    assert calls == ["load"]


def test_load_manifest_parses_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "FIGURE_MANIFEST.md"
    manifest_path.write_text(
        "\n".join(
            [
                "# Figure Manifest",
                "",
                "## Overview",
                "- table:overview_metrics (entity/project/run::overview_metrics) – Summary table",
                "",
                "## Sweeps & Ablations",
                "- image:sweep_curve (entity/project/sweep::sweep_curve)",
            ]
        )
    )

    cached = build._load_manifest(manifest_path)

    assert cached["Overview"]["overview_metrics"].caption == "Summary table"
    assert cached["Sweeps & Ablations"]["sweep_curve"].kind == "image"


def test_log_assets_reuses_cached_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    reused = build._LoggedAsset(
        section="Overview",
        key="overview_metrics",
        run_path="entity/project/run",
        kind="table",
        title="overview_metrics",
        caption="existing",
    )

    class DummyFrame:
        empty = False

    tables = [("overview_metrics", DummyFrame(), "Overview metrics")]

    # ``wandb`` should never be imported when we reuse cached assets
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace())

    result = build._log_assets_to_wandb(
        "Overview",
        entity=None,
        project="project",
        tables=tables,
        figures=[],
        existing_assets={"overview_metrics": reused},
        refresh=False,
    )

    assert result == [reused]
