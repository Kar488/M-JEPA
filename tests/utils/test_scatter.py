import pytest

torch = pytest.importorskip("torch")

import utils.scatter as scatter_mod


def _manual_scatter_sum(index: torch.Tensor, src: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype)
    if src.numel() == 0:
        return out
    view = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, view, src)
    return out


def test_scatter_sum_fallback_matches_manual(monkeypatch):
    monkeypatch.setattr(scatter_mod, "_TORCH_SCATTER_ADD", None, raising=False)

    index = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    src = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )

    expected = _manual_scatter_sum(index, src, dim_size=4)
    result = scatter_mod.scatter_sum(index, src, dim_size=4)

    assert torch.allclose(result, expected)


def test_scatter_sum_prefers_torch_scatter(monkeypatch):
    captured = {}

    def fake_scatter_add(src, index, *, dim=0, out=None, dim_size=None):  # type: ignore[override]
        captured["dim"] = dim
        captured["index_dtype"] = index.dtype
        assert out is not None
        # ``scatter_sum`` should zero the buffer before calling the backend.
        assert torch.count_nonzero(out) == 0
        # When an ``out`` tensor is provided its dtype should drive the computation.
        captured["src_dtype"] = src.dtype

        manual = _manual_scatter_sum(index, src, dim_size=out.size(0))
        out.copy_(manual)
        return out

    monkeypatch.setattr(scatter_mod, "_TORCH_SCATTER_ADD", fake_scatter_add, raising=False)

    index = torch.tensor([0, 1, 0], dtype=torch.long)
    src = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float16)
    out = torch.empty((2, 1), dtype=torch.float32)

    result = scatter_mod.scatter_sum(index, src, dim_size=2, out=out)
    expected = torch.tensor([[4.0], [2.0]], dtype=torch.float32)

    assert torch.allclose(result, expected)
    assert captured["dim"] == 0
    assert captured["index_dtype"] == torch.long
    assert captured["src_dtype"] == torch.float32


def test_scatter_sum_handles_empty_sources(monkeypatch):
    monkeypatch.setattr(scatter_mod, "_TORCH_SCATTER_ADD", None, raising=False)

    index = torch.tensor([], dtype=torch.long)
    src = torch.zeros((0, 3), dtype=torch.float32)

    result = scatter_mod.scatter_sum(index, src, dim_size=5)

    assert result.shape == (5, 3)
    assert torch.count_nonzero(result) == 0
