import subprocess
import yaml

from adapters.cli_runner import BaselineCLI


def test_baseline_cli(tmp_path, monkeypatch):
    cfg = {
        "paths": {"foo": "repo/foo"},
        "outputs": {"root": str(tmp_path / "root")},
        "commands": {
            "foo": {
                "train": "echo train {repo} {unlabeled} {out}",
                "embed": "echo emb {ckpt} {smiles} {emb}",
            }
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    runner = BaselineCLI(str(cfg_path))
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner.train("foo", "data.smi", tmp_path / "out")
    runner.embed("foo", "ckpt.pt", "smiles.smi", tmp_path / "emb.npy")

    assert calls[0][0] == "echo" and calls[0][-1] == str(tmp_path / "out")
    assert calls[1][0] == "echo" and calls[1][-1] == str(tmp_path / "emb.npy")
    assert runner.outputs_dir("foo") == str(tmp_path / "root" / "foo")