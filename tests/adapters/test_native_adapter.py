import sys

import pytest

from adapters.native_adapter import NativeBaseline


def test_native_baseline_dispatch_and_syspath(tmp_path, monkeypatch):
    train_mod = tmp_path / "train_mod.py"
    train_mod.write_text("def train():\n    return 'trained'\n")
    embed_mod = tmp_path / "embed_mod.py"
    embed_mod.write_text("def embed():\n    return 'embedded'\n")

    monkeypatch.setattr(sys, "path", sys.path.copy())
    nb = NativeBaseline(
        str(tmp_path),
        {"module": "train_mod", "function": "train"},
        {"module": "embed_mod", "function": "embed"},
    )
    assert sys.path[0] == nb.repo_path
    assert nb.train() == "trained"
    assert nb.embed() == "embedded"


def test_native_baseline_missing_module_or_function(tmp_path, monkeypatch):
    # valid module for embed
    (tmp_path / "train_mod.py").write_text("def train():\n    pass\n")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(ModuleNotFoundError) as exc:
        NativeBaseline(
            str(tmp_path),
            {"module": "missing_mod", "function": "train"},
            {"module": "train_mod", "function": "train"},
        )
    assert "missing_mod" in str(exc.value)

    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(AttributeError) as exc:
        NativeBaseline(
            str(tmp_path),
            {"module": "train_mod", "function": "missing_fn"},
            {"module": "train_mod", "function": "train"},
        )
    assert "missing_fn" in str(exc.value)