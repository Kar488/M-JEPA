"""Project-level pytest configuration hooks."""
from __future__ import annotations

import warnings


def pytest_addoption(parser):
    """Register compatibility shims when optional plugins are unavailable."""
    try:
        import pytest_cov  # type: ignore # noqa: F401
    except Exception:  # pragma: no cover - best-effort guard only
        group = parser.getgroup("cov", "coverage reporting")
        group.addoption(
            "--cov",
            action="append",
            default=[],
            metavar="PATH",
            help=(
                "No-op placeholder added because pytest-cov is not installed. "
                "Install pytest-cov to enable coverage reporting."
            ),
        )
        group.addoption(
            "--cov-report",
            action="append",
            default=[],
            metavar="TYPE",
            help="No-op placeholder because pytest-cov is not installed.",
        )
        group.addoption(
            "--cov-fail-under",
            action="store",
            default=0,
            metavar="MIN",
            type=int,
            help="No-op placeholder because pytest-cov is not installed.",
        )
        warnings.warn(
            "pytest-cov is not installed; coverage options from pytest.ini will be ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
