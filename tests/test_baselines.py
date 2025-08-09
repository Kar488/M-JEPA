import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from training import baselines


def test_run_baseline_with_mock(tmp_path, monkeypatch):
    base = tmp_path / "MolCLR" / "molclr"
    base.mkdir(parents=True)
    (base / "__init__.py").write_text(
        "result = {}\n\n"
        "def main(*, dataset, device='cpu', **kwargs):\n"
        "    result.update({'dataset': dataset, 'device': device, **kwargs})\n"
        "    return result\n"
    )
    monkeypatch.setattr(baselines, "THIRD_PARTY", tmp_path)
    repo_path = tmp_path / "MolCLR"
    assert str(repo_path) not in sys.path

    out = baselines.run_baseline(
        "molclr", dataset="data", device="cpu", cfg={"a": 1}, extra=2
    )
    assert out == {"dataset": "data", "device": "cpu", "a": 1, "extra": 2}
    assert str(repo_path) in sys.path


def test_run_baseline_unknown():
    with pytest.raises(ValueError):
        baselines.run_baseline("unknown", dataset=None)
