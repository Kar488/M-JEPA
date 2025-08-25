import pytest
import torch

from utils.schedule import cosine_with_warmup


def test_cosine_with_warmup_lr_values():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sch = cosine_with_warmup(opt, warmup_steps=10, total_steps=100)

    lrs = {}
    for step in [0, 10, 50, 100]:
        opt.step()
        sch.step(step)
        lrs[step] = opt.param_groups[0]["lr"]

    assert lrs[0] == pytest.approx(0.0)
    assert lrs[10] == pytest.approx(1.0)
    assert lrs[50] == pytest.approx(0.5868240888)
    assert lrs[100] == pytest.approx(0.0)