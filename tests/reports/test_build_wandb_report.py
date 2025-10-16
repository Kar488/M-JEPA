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


def test_assemble_report_prefers_api(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert calls[0]["api"] == "sentinel-api"


def test_assemble_report_falls_back_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class ClientOnlyReport:
        def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - simple stub
            if "api" in kwargs:
                raise TypeError("unexpected api argument")
            if "client" not in kwargs:
                raise TypeError("missing client argument")

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
    assert calls[0].get("api") == "sentinel-client"
    assert calls[1]["client"] == "sentinel-client"

