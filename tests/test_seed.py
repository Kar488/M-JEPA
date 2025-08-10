import numpy as np
import torch

from utils.seed import set_seed


def test_set_seed_reproducible_numpy_pytorch():
    set_seed(42)
    np_first = np.random.rand(5)
    torch_first = torch.rand(5)

    set_seed(42)
    np_second = np.random.rand(5)
    torch_second = torch.rand(5)

    assert np.array_equal(np_first, np_second)
    assert torch.equal(torch_first, torch_second)
