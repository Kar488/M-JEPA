import torch
import importlib
import pytest
import sys

import pathlib


import importlib.util, sys, pathlib, inspect, os

def import_pooling_fresh():
    path = pathlib.Path(__file__).resolve().parents[2] / "utils" / "pooling.py"
    mod_name = "pooling_under_test"

    # ensure we don't reuse a previous module object
    sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None, "spec_from_file_location returned None"
    assert spec.origin == str(path), f"Spec origin mismatch: {spec.origin}"
    assert spec.loader is not None, "No loader for pooling.py (cannot execute)"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # PROVE it's the right pooling
    assert hasattr(mod, "global_mean_pool"), "global_mean_pool missing"
    fn_file = inspect.getsourcefile(mod.global_mean_pool)
    assert fn_file is not None
    # samefile handles case/sep differences on Windows
    assert os.path.samefile(fn_file, str(path)), f"Loaded from {fn_file}, expected {path}"
    # also confirm module file path
    assert os.path.samefile(mod.__file__, str(path)), f"Module file {mod.__file__}"

    return mod

def test_global_mean_pool_basic():
    pooling = import_pooling_fresh()
    node_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    graph_ptr = torch.tensor([0, 3, 4])
    pooled = pooling.global_mean_pool(node_emb, graph_ptr)
    expected = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    assert pooled.shape[0] == graph_ptr.numel() - 1  # tripwire
    assert torch.allclose(pooled, expected)


def test_global_mean_pool_handles_empty_graph():
    pooling = import_pooling_fresh()
    node_emb = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    graph_ptr = torch.tensor([0, 2, 2])
    pooled = pooling.global_mean_pool(node_emb, graph_ptr)
    expected = torch.tensor([[1.5, 1.5], [0.0, 0.0]])
    assert pooled.shape[0] == graph_ptr.numel() - 1
    assert torch.allclose(pooled, expected)