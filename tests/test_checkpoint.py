import torch

from utils.checkpoint import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    state = {"epoch": 1, "metric": torch.tensor(0.5)}
    save_checkpoint(str(path), **state)

    loaded = load_checkpoint(str(path))
    assert loaded["epoch"] == state["epoch"]
    assert torch.equal(loaded["metric"], state["metric"])