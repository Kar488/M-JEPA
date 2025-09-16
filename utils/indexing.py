from __future__ import annotations

import torch


class _GatherNodes(torch.autograd.Function):
    """Gather rows along dimension 0 with a cudagraph-friendly backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if index.dtype != torch.long:
            index = index.to(dtype=torch.long)
        if index.device != x.device:
            index = index.to(device=x.device)
        # ``ctx.needs_input_grad`` isn't safely traceable by ``torch.compile`` in
        # some PyTorch builds. Cache the required information explicitly to avoid
        # hitting Dynamo's unsupported ``__getitem__`` on the autograd context.
        ctx._needs_grad_x = x.requires_grad
        ctx.index_shape = tuple(index.shape)
        ctx.rest_shape = tuple(x.shape[1:])
        flat_index = index.reshape(-1)
        ctx.save_for_backward(flat_index)
        ctx.x_shape = tuple(x.shape)
        out = torch.index_select(x, 0, flat_index)
        if ctx.index_shape != (flat_index.numel(),):
            out = out.view(*ctx.index_shape, *ctx.rest_shape)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (flat_index,) = ctx.saved_tensors
        if not ctx._needs_grad_x:
            return None, None
        grad_input = grad_output.reshape(flat_index.numel(), *ctx.rest_shape)
        grad_x = grad_output.new_zeros(ctx.x_shape)
        if grad_input.numel() == 0:
            return grad_x, None

        view_shape = (flat_index.numel(),) + (1,) * (grad_input.dim() - 1)
        expanded_index = flat_index.view(view_shape).expand_as(grad_input)
        grad_x.scatter_add_(0, expanded_index, grad_input)

        return grad_x, None


def gather_nodes(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``x[index]`` with a backward pass implemented via ``scatter_add_``.

    PyTorch's advanced indexing dispatches to ``aten::_index_put_impl_`` in the
    backward pass, which prevents CUDA graph capture. This helper mirrors the
    behaviour of ``x[index]`` while providing a cudagraph-friendly backward based
    on ``scatter_add_``.
    """
    
    return _GatherNodes.apply(x, index)
