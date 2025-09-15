import os
import pytest
import torch

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_state_dict_forgiving,
    resolve_ckpt_path,
    safe_load_checkpoint,
    safe_link_or_copy,
)


def test_save_and_load_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    state = {"epoch": 1, "metric": torch.tensor(0.5)}
    save_checkpoint(str(path), **state)
    loaded = load_checkpoint(str(path))
    assert loaded["epoch"] == state["epoch"]
    assert torch.equal(loaded["metric"], state["metric"])


class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


def test_load_state_dict_forgiving_handles_missing_and_extra():
    mod = ToyModule()
    state = {"lin.weight": torch.ones_like(mod.lin.weight), "extra": torch.ones(1, 1)}
    res = load_state_dict_forgiving(mod, state)
    assert res is not None


def test_load_state_dict_forgiving_no_loader():
    class Dummy:
        pass
    assert load_state_dict_forgiving(Dummy(), {"w": torch.tensor(1.0)}) is None


def test_resolve_ckpt_path(tmp_path):
    primary = tmp_path / "p.pt"
    primary.write_text("x")
    assert resolve_ckpt_path(str(primary)) == str(primary)
    fb_dir = tmp_path / "d"
    fb_dir.mkdir()
    fb = fb_dir / "head.pt"
    fb.write_text("y")
    assert resolve_ckpt_path(None, str(fb_dir)) == str(fb)
    with pytest.raises(FileNotFoundError):
        resolve_ckpt_path(str(tmp_path / "missing"))



def test_safe_link_or_copy(tmp_path):
    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    src.write_text("hi")
    mode = safe_link_or_copy(str(src), str(dst))
    assert os.path.exists(dst)
    assert dst.read_text() == "hi"
    assert mode in {"symlink", "hardlink", "copy"}

def test_safe_load_checkpoint(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"a": 1}, path)
    state, used = safe_load_checkpoint(str(path), allow_missing=False)
    assert state["a"] == 1 and used == str(path)
    missing_dir = tmp_path / "none"
    state2, used2 = safe_load_checkpoint("missing", ckpt_dir=str(missing_dir))
    assert state2["encoder"] == {} and used2 is None
