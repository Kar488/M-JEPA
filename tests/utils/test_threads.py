"""Tests for OpenMP threading helpers."""

from __future__ import annotations

import os

from utils import threads


def test_recommend_omp_threads_accepts_string_world_size(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert threads.recommend_omp_threads(num_workers=2, world_size="2") == 3


def test_recommend_omp_threads_accepts_int_world_size(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert threads.recommend_omp_threads(num_workers=1, world_size=3) == 1
