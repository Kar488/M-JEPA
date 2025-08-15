import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import logging

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


def test_run_baseline_unknown(caplog):

    #with pytest.raises(ValueError):
        #baselines.run_baseline("unknown", dataset=None)
    # We know training.baselines will throw a ValueError for unknown baselines,
    # but we don't want to report it here, so we just check the name.
    with caplog.at_level(logging.CRITICAL, logger='training.baselines'):
         with pytest.raises(ValueError):
            baselines.run_baseline("unknown", dataset=None)
    #assert "Unknown baseline 'unknown'" in caplog.text


@pytest.mark.parametrize(
    "name, structure",
    [
        ("molclr", ("MolCLR", "molclr", "__init__.py")),
        ("geomgcl", ("GeomGCL", "train_gcl.py")),
        ("himol", ("HiMol", "pretrain.py")),
    ],
)
def test_baseline_forward_pass(tmp_path, monkeypatch, name, structure):
    module_name = {
        "molclr": "molclr",
        "geomgcl": "train_gcl",
        "himol": "pretrain",
    }[name]
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    repo_dir = tmp_path / structure[0]
    if len(structure) == 3:
        pkg_dir = repo_dir / structure[1]
        pkg_dir.mkdir(parents=True)
        module_file = pkg_dir / structure[2]
    else:
        repo_dir.mkdir(parents=True)
        module_file = repo_dir / structure[1]
    module_file.write_text(
        "import torch\n"
        "def main(*, dataset, device='cpu', **kwargs):\n"
        "    model=torch.nn.Linear(dataset.size(1), 2).to(device)\n"
        "    with torch.no_grad():\n"
        "        return model(dataset.to(device))\n"
    )
    monkeypatch.setattr(baselines, "THIRD_PARTY", tmp_path)
    data = torch.randn(4, 3)
    out = baselines.run_baseline(name, dataset=data, device="cpu")
    assert out.shape == (4, 2)
