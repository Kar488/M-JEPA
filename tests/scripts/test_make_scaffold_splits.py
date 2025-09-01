import runpy
import sys
import types

import pandas as pd


def test_make_scaffold_splits_invokes_writer(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"smiles": ["C"]}).to_csv(csv_path, index=False)
    out_dir = tmp_path / "splits"

    calls = {}

    def fake_writer(**kwargs):
        calls.update(kwargs)

    stub = types.SimpleNamespace(write_scaffold_splits=fake_writer)
    monkeypatch.setitem(sys.modules, "data.scaffold_split", stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(csv_path),
            "--out_dir",
            str(out_dir),
            "--format",
            "csv",
        ],
    )

    runpy.run_module("scripts.make_scaffold_splits", run_name="__main__")
    assert calls["out_dir"] == str(out_dir)
    assert calls["fmt"] == "csv"
