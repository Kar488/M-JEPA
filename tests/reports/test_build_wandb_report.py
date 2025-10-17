"""Tests for the W&B report assembly helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import importlib


def _install_pandas_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFrame(dict):
        pass

    pandas_stub = types.SimpleNamespace(
        DataFrame=lambda *args, **kwargs: _FakeFrame(data=args, kwargs=kwargs),
        concat=lambda frames, **kwargs: list(frames),
    )
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)


def _install_wandb_stub(monkeypatch: pytest.MonkeyPatch, report_cls: type) -> List[Dict[str, Any]]:
    """Install a stub ``wandb_workspaces.reports.v2`` module."""

    calls: List[Dict[str, Any]] = []

    class RecordingReport(report_cls):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - thin wrapper
            calls.append(dict(kwargs))
            super().__init__(**kwargs)
            self.url = "https://wandb.test/report"

        def save(self) -> None:  # pragma: no cover - simple stub
            return None

    panels = types.SimpleNamespace(
        RunTable=lambda **kwargs: ("table", kwargs),
        RunImage=lambda **kwargs: ("image", kwargs),
    )
    blocks = types.SimpleNamespace(
        PanelGrid=lambda panels, title: {"title": title, "panels": panels},
    )

    v2_module = types.SimpleNamespace(Report=RecordingReport, panels=panels, blocks=blocks)
    reports_module = types.SimpleNamespace(v2=v2_module)

    monkeypatch.setitem(sys.modules, "wandb_workspaces", types.SimpleNamespace(reports=reports_module))
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports", reports_module)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports.v2", v2_module)

    return calls


def _load_report_module(
    monkeypatch: pytest.MonkeyPatch, report_cls: type
) -> Tuple[Any, List[Dict[str, Any]]]:
    _install_pandas_stub(monkeypatch)
    calls = _install_wandb_stub(monkeypatch, report_cls)
    monkeypatch.delitem(sys.modules, "reports.build_wandb_report", raising=False)
    monkeypatch.delitem(sys.modules, "reports", raising=False)
    import tests.conftest as test_conftest

    original_rmtree = test_conftest.shutil.rmtree

    def guarded_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        target = Path(path)
        if target.resolve() == Path.cwd() / "reports":
            return None
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(test_conftest.shutil, "rmtree", guarded_rmtree)
    module = importlib.import_module("reports.build_wandb_report")
    return module, calls


def _empty_assets() -> Dict[str, list[Any]]:
    from reports.build_wandb_report import REPORT_SECTIONS

    return {section: [] for section in REPORT_SECTIONS}


def test_assemble_report_passes_supported_api_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ApiOnlyReport:
        def __init__(self, *, api: Any, **_: Any) -> None:  # pragma: no cover - simple stub
            self.api = api

    build_wandb_report, calls = _load_report_module(monkeypatch, ApiOnlyReport)

    assets = _empty_assets()
    assets["Overview"].append(
        build_wandb_report._LoggedAsset(
            section="Overview",
            key="dummy-table",
            run_path="entity/project/run",
            kind="table",
            title="Overview",
        )
    )

    report_url = build_wandb_report._assemble_report(
        api="sentinel-api",
        entity="entity",
        project="project",
        assets_by_section=assets,
    )

    assert report_url is not None
    assert len(calls) == 1
    assert calls[0]["api"] == "sentinel-api"


def test_assemble_report_prefers_client_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClientOnlyReport:
        def __init__(self, *, client: Any, **_: Any) -> None:  # pragma: no cover - simple stub
            self.client = client

    build_wandb_report, calls = _load_report_module(monkeypatch, ClientOnlyReport)

    assets = _empty_assets()
    assets["Overview"].append(
        build_wandb_report._LoggedAsset(
            section="Overview",
            key="dummy-table",
            run_path="entity/project/run",
            kind="table",
            title="Overview",
        )
    )

    report_url = build_wandb_report._assemble_report(
        api="sentinel-client",
        entity="entity",
        project="project",
        assets_by_section=assets,
    )

    assert report_url is not None
    assert len(calls) == 1
    assert calls[0]["client"] == "sentinel-client"


def test_assemble_report_handles_connection_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConnectionReport:
        def __init__(self, *, connection: Any, **_: Any) -> None:  # pragma: no cover - simple stub
            self.connection = connection

    build_wandb_report, calls = _load_report_module(monkeypatch, ConnectionReport)

    assets = _empty_assets()
    assets["Overview"].append(
        build_wandb_report._LoggedAsset(
            section="Overview",
            key="dummy-table",
            run_path="entity/project/run",
            kind="table",
            title="Overview",
        )
    )

    report_url = build_wandb_report._assemble_report(
        api="sentinel-connection",
        entity="entity",
        project="project",
        assets_by_section=assets,
    )

    assert report_url is not None
    assert len(calls) == 1
    assert calls[0]["connection"] == "sentinel-connection"


def test_assemble_report_falls_back_to_plain_initialisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlainReport:
        _allowed = {"entity", "project", "title", "description"}

        def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - simple stub
            unexpected = {k: v for k, v in kwargs.items() if k not in self._allowed}
            if unexpected:
                raise AssertionError(f"Unexpected kwargs: {unexpected}")

    build_wandb_report, calls = _load_report_module(monkeypatch, PlainReport)

    assets = _empty_assets()
    assets["Overview"].append(
        build_wandb_report._LoggedAsset(
            section="Overview",
            key="dummy-table",
            run_path="entity/project/run",
            kind="table",
            title="Overview",
        )
    )

    report_url = build_wandb_report._assemble_report(
        api="sentinel-do-not-pass",
        entity="entity",
        project="project",
        assets_by_section=assets,
    )

    assert report_url is not None
    assert len(calls) == 1
    assert set(calls[0]) == {"entity", "project", "title", "description"}
    assert calls[0]["entity"] == "entity"
    assert calls[0]["project"] == "project"

