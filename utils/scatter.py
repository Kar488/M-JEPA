from __future__ import annotations

import torch


def scatter_sum(
    index: torch.Tensor,
    src: torch.Tensor,
    dim_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum ``src`` entries grouped by ``index`` along dimension 0.

    Parameters
    ----------
    index:
        Indices that specify the target row for each entry in ``src``. The tensor
        is flattened internally and must align with ``src``'s first dimension.
    src:
        Values to accumulate. The operation is differentiable with respect to
        this tensor.
    dim_size:
        The size of the first dimension of the output tensor. It typically
        corresponds to the number of nodes in a graph.
    out:
        Optional pre-allocated tensor used to store the result. When provided the
        tensor is zeroed in-place before accumulation, enabling memory reuse.

    Returns
    -------
    torch.Tensor
        A tensor whose first dimension equals ``dim_size`` and whose remaining
        dimensions match ``src[1:]``. The tensor lives on the same device as
        ``src``.
    """

    dim_size = int(dim_size)
    if dim_size < 0:
        raise ValueError("dim_size must be non-negative")

    # Normalise the optional ``out`` tensor to match ``src``.
    expected_shape = (dim_size, *src.shape[1:])
    if out is None:
        out = src.new_zeros(expected_shape)
    else:
        if out.shape != expected_shape:
            raise ValueError(
                "out tensor has incompatible shape: "
                f"expected {expected_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != src.dtype:
            src = src.to(out.dtype)
        out.zero_()

    if src.numel() == 0:
        return out

    flat_index = index.reshape(-1).to(device=src.device, dtype=torch.long)
    if flat_index.numel() != src.shape[0]:
        raise ValueError(
            "index and src must agree on the leading dimension: "
            f"got {flat_index.numel()} and {src.shape[0]}"
        )

    view_shape = (flat_index.numel(),) + (1,) * (src.dim() - 1)
    expanded_index = flat_index.view(view_shape).expand_as(src)
    out.scatter_add_(0, expanded_index, src)
    return out
