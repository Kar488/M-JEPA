from __future__ import annotations

import contextlib
import errno
import logging
import math
import os
import time as _t
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import tqdm
from torch.utils.data import DataLoader

try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None  # type: ignore[assignment]

_AMP_GRAD_SCALER_CANDIDATES: List[type] = []
_torch_amp = getattr(torch, "amp", None)
_amp_grad_scaler = getattr(_torch_amp, "GradScaler", None) if _torch_amp else None
if isinstance(_amp_grad_scaler, type):
    _AMP_GRAD_SCALER_CANDIDATES.append(_amp_grad_scaler)

_torch_cuda = getattr(torch, "cuda", None)
_torch_cuda_amp = getattr(_torch_cuda, "amp", None) if _torch_cuda else None
_cuda_grad_scaler = (
    getattr(_torch_cuda_amp, "GradScaler", None) if _torch_cuda_amp else None
)
if isinstance(_cuda_grad_scaler, type):
    _AMP_GRAD_SCALER_CANDIDATES.append(_cuda_grad_scaler)


class _NoOpGradScaler:
    """Fallback GradScaler for CPU-only or minimal Torch builds."""

    def __init__(self, enabled: bool = False):
        self._enabled = False

    def is_enabled(self) -> bool:  # pragma: no cover - trivial shim
        return False

    def get_scale(self) -> float:  # pragma: no cover - trivial shim
        return 1.0

    def scale(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial shim
        return tensor

    def step(self, optimizer: optim.Optimizer) -> None:  # pragma: no cover - trivial shim
        optimizer.step()

    def update(self) -> None:  # pragma: no cover - trivial shim
        return None

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial shim
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:  # pragma: no cover - trivial shim
        return None


if not _AMP_GRAD_SCALER_CANDIDATES:
    _AMP_GRAD_SCALER_CANDIDATES.append(_NoOpGradScaler)

GradScaler = _AMP_GRAD_SCALER_CANDIDATES[0]
_GRAD_SCALER_TYPES: Tuple[type, ...] = tuple({cls for cls in _AMP_GRAD_SCALER_CANDIDATES})


def _is_grad_scaler(obj: Any) -> bool:
    return isinstance(obj, _GRAD_SCALER_TYPES)

try:
    from data.augment import (
        apply_graph_augmentations,
        delete_random_bond,
        mask_random_atom,
        remove_random_subgraph,
        mask_subgraph,
        generate_views,
    )
except ImportError:  # pragma: no cover - used in minimal test stubs
    from data.augment import apply_graph_augmentations

    def delete_random_bond(g):
        return g

    def mask_random_atom(g):
        return g

    def remove_random_subgraph(g):
        return g

    def mask_subgraph(g, mask_ratio, contiguous):
        return g, g

    def generate_views(graph, structural_ops=None, geometric_ops=None):
        return [graph]


from data.mdataset import GraphData, GraphDataset
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ddp import DistributedSamplerList, cleanup, init_distributed, is_main_process
from utils.dataloader import normalize_prefetch_factor
from utils.graph_ops import _encode_graph, _pool_graph_emb
from utils.logging import maybe_init_wandb
logger = logging.getLogger(__name__)
from utils.schedule import cosine_with_warmup


def _ensure_open_file_limit(min_soft_limit: int = 4096) -> None:
    """Best-effort bump of ``RLIMIT_NOFILE`` so dataloaders stay healthy."""

    if resource is None:  # pragma: no cover - platform dependent
        return

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):  # pragma: no cover - depends on runtime
        return

    desired = max(int(min_soft_limit), soft)
    if desired <= soft:
        return

    target_hard = hard if hard >= desired else desired
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, target_hard))
        logger.debug(
            "Raised RLIMIT_NOFILE soft limit from %d to %d (hard %d -> %d)",
            soft,
            desired,
            hard,
            target_hard,
        )
        return
    except (OSError, ValueError):
        pass

    fallback_soft = min(max(desired, soft), hard)
    if fallback_soft <= soft:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (fallback_soft, hard))
        logger.debug(
            "Raised RLIMIT_NOFILE soft limit to hard limit %d after initial attempt failed", fallback_soft
        )
    except (OSError, ValueError):  # pragma: no cover - depends on runtime
        logger.debug(
            "Unable to raise RLIMIT_NOFILE beyond current soft limit %d despite request for %d",
            soft,
            desired,
        )


def _make_wandb_handlers(wb: Any) -> Tuple[Callable[..., None], Callable[[], None]]:
    """Return safe ``(log_fn, finish_fn)`` closures for W&B runs."""

    wandb_active = bool(wb)

    def _resolve_log_callable() -> Optional[Callable[..., Any]]:
        target = getattr(wb, "log", None)
        if callable(target):
            return target
        run = getattr(wb, "run", None)
        target = getattr(run, "log", None)
        return target if callable(target) else None

    def _log(payload: Mapping[str, Any], **kwargs: Any) -> None:
        nonlocal wandb_active
        if not wandb_active or not is_main_process():
            return
        log_fn = _resolve_log_callable()
        if log_fn is None:
            wandb_active = False
            return
        try:
            log_fn(payload, **kwargs)
        except Exception as exc:  # pragma: no cover - backend dependent
            wandb_active = False
            logger.warning(
                "Disabling Weights & Biases logging after failure: %s", exc
            )

    def _finish() -> None:
        nonlocal wandb_active
        if not wandb_active or not is_main_process():
            return
        finish_fn = getattr(wb, "finish", None)
        if callable(finish_fn):
            with contextlib.suppress(Exception):  # pragma: no cover - backend dependent
                finish_fn()
        wandb_active = False

    return _log, _finish


# ``torch.compile`` incurs a noticeable warm-up cost per model variant.  Phase-2
# sweeps cap each trial to a few hundred mini-batches (see
# ``max_pretrain_batches`` in the sweep specs), which meant the warm-up often
# consumed a significant fraction of the allocated budget.  Raising the
# amortisation threshold ensures compilation only activates when the trial has
# enough batches to hide the warm-up latency.  The new value still allows the
# long-form pretraining stage to use compilation.
_COMPILE_WARMUP_BATCHES = 512


def _should_compile_models(
    compile_requested: bool,
    device: torch.device,
    planned_batches: int,
) -> bool:
    """Return ``True`` when ``torch.compile`` is worth the warm-up cost."""

    if not compile_requested:
        return False
    if planned_batches <= 0:
        return False
    if device.type == "cpu":
        return False
    if planned_batches < _COMPILE_WARMUP_BATCHES:
        return False
    return True


def _maybe_pin(tensor: Optional[torch.Tensor], device: Optional[str | torch.device] = None) -> Optional[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    try:
        return tensor.pin_memory(device)
    except NotImplementedError:  # pragma: no cover - backend dependent
        return tensor


def _graph_to_serialisable(graph: GraphData) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "x": _to_numpy(getattr(graph, "x", None)),
        "edge_index": _to_numpy(getattr(graph, "edge_index", None)),
        "edge_attr": _to_numpy(getattr(graph, "edge_attr", None)),
        "pos": _to_numpy(getattr(graph, "pos", None)),
    }
    extras: Dict[str, Any] = {}
    for name, value in vars(graph).items():
        if name in state:
            continue
        converted = value
        try:
            import torch as _t  # local import to avoid hard dependency in stubs

            if isinstance(value, _t.Tensor):
                converted = _to_numpy(value)
        except Exception:  # pragma: no cover - torch optional in some stubs
            pass
        extras[name] = converted
    if extras:
        state["extras"] = extras
    return state


def _graph_from_serialisable(state: Mapping[str, Any]) -> GraphData:
    g = GraphData(
        x=state.get("x"),
        edge_index=state.get("edge_index"),
        edge_attr=state.get("edge_attr"),
        pos=state.get("pos"),
    )
    extras = state.get("extras", {})
    for name, value in extras.items():
        setattr(g, name, value)
    return g

class PackedGraphBatch(dict):
    """Dictionary subclass that reports the number of graphs in the batch."""

    __slots__ = ()

    def __len__(self) -> int:  # pragma: no cover - trivial
        graphs = self.get("graphs")
        if isinstance(graphs, Sequence):
            return len(graphs)
        return 0

    @property
    def num_graphs(self) -> int:
        graphs = self.get("graphs")
        if isinstance(graphs, Sequence):
            return len(graphs)
        return 0


@dataclass
class GraphBatch:
    """Container mirroring a PyG ``Batch`` with serialisation helpers."""

    graphs: List[GraphData]
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterable[GraphData]:
        return iter(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphData:
        return self.graphs[idx]

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    def to(self, device: torch.device | str) -> "GraphBatch":
        device = torch.device(device)
        edge_attr = (
            self.edge_attr.to(device)
            if isinstance(self.edge_attr, torch.Tensor)
            else self.edge_attr
        )
        pos = self.pos.to(device) if isinstance(self.pos, torch.Tensor) else self.pos
        return GraphBatch(
            graphs=self.graphs,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            edge_attr=edge_attr,
            pos=pos,
        )

    def pin_memory(self, device: Optional[str | torch.device] = None) -> "GraphBatch":
        return GraphBatch(
            graphs=self.graphs,
            x=_maybe_pin(self.x, device),
            edge_index=_maybe_pin(self.edge_index, device),
            batch=_maybe_pin(self.batch, device),
            ptr=_maybe_pin(self.ptr, device),
            edge_attr=_maybe_pin(self.edge_attr, device),
            pos=_maybe_pin(self.pos, device),
        )

    def pack(self) -> PackedGraphBatch:
        return PackedGraphBatch(
            {
                "graphs": [_graph_to_serialisable(g) for g in self.graphs],
                "x": self.x,
                "edge_index": self.edge_index,
                "batch": self.batch,
                "ptr": self.ptr,
                "edge_attr": self.edge_attr,
                "pos": self.pos,
            }
        )

    @staticmethod
    def from_packed(state: Mapping[str, Any]) -> "GraphBatch":
        graphs_state = state.get("graphs", [])
        graphs = [_graph_from_serialisable(gs) for gs in graphs_state]
        return GraphBatch(
            graphs=graphs,
            x=state["x"],
            edge_index=state["edge_index"],
            batch=state["batch"],
            ptr=state["ptr"],
            edge_attr=state.get("edge_attr"),
            pos=state.get("pos"),
        )


_DEVICE_MOVE_LOGGED = False


def _move_graph_batch_to_device(batch: GraphBatch, device: torch.device | str) -> GraphBatch:
    """Move ``batch`` to ``device`` from the main process."""

    global _DEVICE_MOVE_LOGGED
    try:
        worker_info = torch.utils.data.get_worker_info()
    except Exception:  # pragma: no cover - torch may not expose worker info
        worker_info = None
    if worker_info is not None:
        raise RuntimeError(
            "_move_graph_batch_to_device was called inside a DataLoader worker; "
            "batches must remain on CPU until the main process performs the device transfer."
        )
    device_obj = torch.device(device)
    if not _DEVICE_MOVE_LOGGED:
        logger.info("Moving graph batches to %s in the main process", device_obj)
        _DEVICE_MOVE_LOGGED = True
    return batch.to(device_obj)

def _step_optimizer(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    *,
    scaler: Optional[Any] = None,
    scheduler: Optional[_LRScheduler] = None,
) -> None:
    """Backpropagate ``loss`` and update ``optimizer``/``scheduler`` safely.

    When ``GradScaler`` skips an optimiser step due to inf/NaN gradients we avoid
    stepping the scheduler so PyTorch does not warn about the order of
    operations.  ``GradScaler`` is optional and treated as disabled when not
    provided or not enabled.  We also guard against the scheduler being advanced
    before the optimiser's internal step counter increases (e.g. after resume)
    to suppress the PyTorch warning about an out-of-order call sequence.
    """

    step_executed = True
    if _is_grad_scaler(scaler):
        scaler_enabled = bool(getattr(scaler, "is_enabled", lambda: True)())
        prev_scale = scaler.get_scale() if scaler_enabled else None
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler_enabled and prev_scale is not None:
            step_executed = scaler.get_scale() >= prev_scale
    else:
        loss.backward()
        optimizer.step()

    if scheduler is not None and step_executed:
        opt_steps = getattr(optimizer, "_step_count", None)
        sched_steps = getattr(scheduler, "_step_count", None)
        if (
            opt_steps is not None
            and sched_steps is not None
            and opt_steps <= sched_steps
        ):
            logger.debug(
                "Skipping scheduler.step() because optimizer has not advanced: opt=%s sched=%s",
                opt_steps,
                sched_steps,
            )
            return
        scheduler.step()



def _collate_graph_batch(batch: Sequence[GraphData] | GraphBatch) -> Dict[str, Any]:
    """Collate a sequence of ``GraphData`` objects into a packed batch state."""

    if isinstance(batch, GraphBatch):
        return batch.pack()

    graphs = list(batch)
    if not graphs:
        zero = torch.zeros(0, dtype=torch.long)
        zero_mat = torch.zeros((0, 0), dtype=torch.float32)
        zero_edge = torch.zeros((2, 0), dtype=torch.long)
        ptr = torch.zeros(1, dtype=torch.long)
        empty = GraphBatch([], zero_mat, zero_edge, zero, ptr)
        return empty.pack()

    x_blocks: List[torch.Tensor] = []
    batch_index: List[torch.Tensor] = []
    edge_blocks: List[torch.Tensor] = []
    ptr: List[int] = [0]
    node_offset = 0

    all_have_pos = True
    edge_attr_dim = 0
    has_edge_attr = False
    edge_attr_blocks: List[Optional[torch.Tensor]] = []
    edge_counts: List[int] = []

    for idx, g in enumerate(graphs):
        x = torch.as_tensor(getattr(g, "x"), dtype=torch.float32)
        if x.dim() != 2:
            raise ValueError("GraphData.x must be a 2D array/tensor")
        x_blocks.append(x)
        n_i = int(x.size(0))
        batch_index.append(torch.full((n_i,), idx, dtype=torch.long))
        ptr.append(ptr[-1] + n_i)

        pos = getattr(g, "pos", None)
        if pos is None:
            all_have_pos = False

        edge_index = getattr(g, "edge_index", None)
        if edge_index is None:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            ei = torch.as_tensor(edge_index, dtype=torch.long)
            if ei.dim() == 2 and ei.size(0) != 2 and ei.size(1) == 2:
                ei = ei.t()
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError("edge_index must have shape [2, E]")
        edge_blocks.append(ei + node_offset if ei.numel() else ei)
        edge_count = int(ei.size(1))
        edge_counts.append(edge_count)

        edge_attr = getattr(g, "edge_attr", None)
        if edge_attr is not None:
            ea = torch.as_tensor(edge_attr, dtype=torch.float32)
            if ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            edge_attr_dim = max(edge_attr_dim, int(ea.size(-1)))
            has_edge_attr = True
            edge_attr_blocks.append(ea)
        else:
            edge_attr_blocks.append(None)

        node_offset += n_i

    x_cat = torch.cat(x_blocks, dim=0)
    batch_vec = torch.cat(batch_index, dim=0)
    ptr_tensor = torch.tensor(ptr, dtype=torch.long)
    edge_index_cat = (
        torch.cat(edge_blocks, dim=1)
        if edge_blocks
        else torch.zeros((2, 0), dtype=torch.long)
    )

    batch_edge_attr = None
    if has_edge_attr:
        filled: List[torch.Tensor] = []
        for ea, edge_count in zip(edge_attr_blocks, edge_counts):
            if ea is None:
                filled.append(
                    torch.zeros((edge_count, edge_attr_dim), dtype=torch.float32)
                )
            else:
                if ea.size(-1) != edge_attr_dim:
                    pad_dim = edge_attr_dim - ea.size(-1)
                    if pad_dim > 0:
                        pad = torch.zeros((ea.size(0), pad_dim), dtype=ea.dtype)
                        ea = torch.cat([ea, pad], dim=-1)
                    else:
                        ea = ea[:, :edge_attr_dim]
                filled.append(ea)
        batch_edge_attr = (
            torch.cat(filled, dim=0)
            if filled
            else torch.zeros((0, edge_attr_dim), dtype=torch.float32)
        )

    batch_pos = None
    if all_have_pos:
        pos_blocks = [torch.as_tensor(g.pos, dtype=torch.float32) for g in graphs]
        batch_pos = torch.cat(pos_blocks, dim=0) if pos_blocks else None

    batch_obj = GraphBatch(
        graphs=graphs,
        x=x_cat,
        edge_index=edge_index_cat,
        batch=batch_vec,
        ptr=ptr_tensor,
        edge_attr=batch_edge_attr,
        pos=batch_pos,
    )
    return batch_obj.pack()


def _ensure_worker_cpu_tensor(tensor: Any, *, context: str) -> None:
    """Raise when CUDA tensors leak into DataLoader workers."""

    try:  # pragma: no cover - torch optional in lightweight stubs
        import torch as _t
    except Exception:
        return
    if not isinstance(tensor, _t.Tensor) or tensor.device.type == "cpu":
        return
    try:
        worker_info_fn = getattr(_t.utils.data, "get_worker_info", None)
        worker = worker_info_fn() if callable(worker_info_fn) else None
    except Exception:
        worker = None
    if worker is None:
        return
    location = f"DataLoader worker {worker.id}" if hasattr(worker, "id") else "a DataLoader worker"
    raise RuntimeError(
        f"{context} received a tensor on device '{tensor.device}' inside {location}. "
        "Graph featurisation and augmentation must stay on CPU; move tensors to the target "
        "device in the main process."
    )


def _clone_graph_data(graph: GraphData) -> GraphData:
    """Deep copy a :class:`GraphData` instance (numpy or torch backed)."""

    try:  # pragma: no cover - optional dependency
        import torch as _t

        has_torch = True
    except Exception:  # pragma: no cover - torch not available in light stubs
        has_torch = False
        _t = None  # type: ignore

    def _cp(arr):
        if arr is None:
            return None
        if has_torch and isinstance(arr, _t.Tensor):
            _ensure_worker_cpu_tensor(arr, context="_clone_graph_data")
            return arr.detach().clone()
        if hasattr(arr, "copy"):
            return arr.copy()
        return np.array(arr, copy=True)

    g2 = GraphData(
        x=_cp(getattr(graph, "x", None)),
        edge_index=_cp(getattr(graph, "edge_index", None)),
        edge_attr=_cp(getattr(graph, "edge_attr", None)),
        pos=_cp(getattr(graph, "pos", None)),
    )
    for name in ("y", "mask", "batch", "smiles", "id"):
        if hasattr(graph, name):
            try:
                setattr(g2, name, _cp(getattr(graph, name)))
            except Exception:
                setattr(g2, name, getattr(graph, name))
    return g2


def _graph_num_nodes(graph: GraphData) -> int:
    x = getattr(graph, "x", None)
    if x is None:
        return 0
    try:
        return int(x.shape[0])
    except Exception:
        return int(len(x)) if hasattr(x, "__len__") else 0


def _to_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr
    try:
        import torch as _t  # type: ignore

        if isinstance(arr, _t.Tensor):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)


def _match_type(data, template):
    if data is None:
        return None
    if template is None:
        return data
    try:
        import torch as _t  # type: ignore

        if isinstance(template, _t.Tensor):
            device = template.device if hasattr(template, "device") else None
            return _t.as_tensor(data, dtype=template.dtype, device=device)
    except Exception:
        pass
    if isinstance(template, np.ndarray):
        return np.asarray(data, dtype=template.dtype)
    return data


def _slice_first_dim(arr, indices: List[int]):
    if arr is None:
        return None
    try:
        import torch as _t  # type: ignore

        if isinstance(arr, _t.Tensor):
            idx = _t.as_tensor(indices, dtype=_t.long, device=arr.device)
            if idx.numel() == 0:
                shape = list(arr.shape)
                shape[0] = 0
                return arr.new_empty(shape)
            return arr.index_select(0, idx)
    except Exception:
        pass
    arr_np = np.asarray(arr)
    if not indices:
        shape = list(arr_np.shape)
        shape[0] = 0
        return np.zeros(shape, dtype=arr_np.dtype)
    return arr_np[np.array(indices, dtype=np.int64)]


def _graph_select_nodes(graph: GraphData, node_indices: List[int]) -> GraphData:
    node_indices = sorted({int(i) for i in node_indices})
    x = getattr(graph, "x", None)
    pos = getattr(graph, "pos", None)
    edge_index = getattr(graph, "edge_index", None)
    edge_attr = getattr(graph, "edge_attr", None)

    x_sel = _slice_first_dim(x, node_indices)
    pos_sel = _slice_first_dim(pos, node_indices)

    edge_index_sel = None
    edge_attr_sel = None
    if edge_index is not None:
        ei_np = _to_numpy(edge_index)
        if ei_np.ndim != 2:
            ei_np = ei_np.reshape(2, -1)
        if ei_np.shape[0] != 2 and ei_np.shape[1] == 2:
            ei_np = ei_np.T
        node_map = {int(old): idx for idx, old in enumerate(node_indices)}
        kept_edges: List[Tuple[int, int]] = []
        kept_cols: List[int] = []
        for col in range(ei_np.shape[1]):
            u = int(ei_np[0, col])
            v = int(ei_np[1, col])
            if u in node_map and v in node_map:
                kept_edges.append((node_map[u], node_map[v]))
                kept_cols.append(col)
        if kept_edges:
            edges_np = np.asarray(kept_edges, dtype=np.int64).T
        else:
            edges_np = np.zeros((2, 0), dtype=np.int64)
        edge_index_sel = _match_type(edges_np, edge_index)
        if edge_attr is not None:
            ea_np = _to_numpy(edge_attr)
            if kept_cols:
                new_attr = ea_np[kept_cols]
            else:
                if ea_np.ndim == 2:
                    new_attr = np.zeros((0, ea_np.shape[1]), dtype=ea_np.dtype)
                else:
                    new_attr = np.zeros((0,), dtype=ea_np.dtype)
            edge_attr_sel = _match_type(new_attr, edge_attr)

    GraphCls = graph.__class__
    try:
        new_graph = GraphCls(
            x=_match_type(x_sel, x),
            edge_index=edge_index_sel,
            edge_attr=edge_attr_sel,
            pos=_match_type(pos_sel, pos),
        )
    except Exception:
        from types import SimpleNamespace

        new_graph = SimpleNamespace(
            x=_match_type(x_sel, x),
            edge_index=edge_index_sel,
            edge_attr=edge_attr_sel,
            pos=_match_type(pos_sel, pos),
        )
    for name in ("y", "mask", "batch", "smiles", "id"):
        if hasattr(graph, name):
            setattr(new_graph, name, getattr(graph, name))
    return new_graph


def _looks_like_mask_pair(original: GraphData, ctx, tgt) -> bool:
    orig_nodes = _graph_num_nodes(original)
    ctx_nodes = _graph_num_nodes(ctx)
    tgt_nodes = _graph_num_nodes(tgt)
    if ctx_nodes <= 0 or tgt_nodes <= 0:
        return False
    if orig_nodes <= 0:
        return True
    if ctx_nodes + tgt_nodes != orig_nodes:
        return False
    if ctx_nodes == orig_nodes and tgt_nodes == orig_nodes:
        return False
    return True


def _fallback_mask_pair(graph: GraphData, mask_ratio: float) -> Tuple[GraphData, GraphData]:
    total_nodes = _graph_num_nodes(graph)
    if total_nodes <= 1:
        return _clone_graph_data(graph), _clone_graph_data(graph)
    k = int(math.ceil(float(mask_ratio) * total_nodes))
    if k <= 0:
        k = 1
    if k >= total_nodes:
        k = total_nodes - 1
    tgt_indices = list(range(k))
    ctx_indices = [i for i in range(total_nodes) if i not in tgt_indices]
    if not ctx_indices:
        ctx_indices = [total_nodes - 1]
        tgt_indices = [i for i in range(total_nodes) if i not in ctx_indices]
    ctx_graph = _graph_select_nodes(graph, ctx_indices)
    tgt_graph = _graph_select_nodes(graph, tgt_indices)
    return ctx_graph, tgt_graph


def _ensure_mask_pair(graph: GraphData, ctx, tgt, mask_ratio: float) -> Tuple[GraphData, GraphData]:
    if _looks_like_mask_pair(graph, ctx, tgt):
        return ctx, tgt
    return _fallback_mask_pair(graph, mask_ratio)


class _MaskSubgraphPairOp:
    """Callable wrapper that returns context and target graphs."""

    def __init__(self, mask_ratio: float, contiguous: bool) -> None:
        self.mask_ratio = mask_ratio
        self.contiguous = contiguous

    def __call__(self, graph: GraphData) -> Tuple[GraphData, GraphData]:
        return mask_subgraph(graph, self.mask_ratio, self.contiguous)


class _MaskSubgraphContextOp:
    """Callable wrapper returning only the context portion of ``mask_subgraph``."""

    def __init__(self, mask_ratio: float, contiguous: bool = False) -> None:
        self.mask_ratio = mask_ratio
        self.contiguous = contiguous

    def __call__(self, graph: GraphData) -> GraphData:
        ctx, _ = mask_subgraph(graph, self.mask_ratio, self.contiguous)
        return ctx


class _ApplyGraphAugmentationsOp:
    """Callable applying geometric augmentations in ``generate_views``."""

    def __init__(self, random_rotate: bool, mask_angle: bool, perturb_dihedral: bool) -> None:
        self.random_rotate = random_rotate
        self.mask_angle = mask_angle
        self.perturb_dihedral = perturb_dihedral

    def __call__(self, graph: GraphData) -> GraphData:
        return apply_graph_augmentations(
            graph,
            rotate=self.random_rotate,
            random_rotate=self.random_rotate,
            mask_angle=self.mask_angle,
            perturb_dihedral=self.perturb_dihedral,
        )


class _JEPAAugmentor:
    """Generate context/target graph pairs for JEPA training."""

    def __init__(
        self,
        mask_ratio: float,
        contiguous: bool,
        *,
        random_rotate: bool = False,
        mask_angle: bool = False,
        perturb_dihedral: bool = False,
        bond_deletion: bool = False,
        atom_masking: bool = False,
        subgraph_removal: bool = False,
        max_retries: int = 4,
    ) -> None:
        structural_ops: List[Callable[[GraphData], GraphData | tuple[GraphData, ...]]] = []
        pre_mask_ops: List[Callable[[GraphData], GraphData]] = []
        if bond_deletion:
            structural_ops.append(delete_random_bond)
            pre_mask_ops.append(delete_random_bond)
        if atom_masking:
            structural_ops.append(mask_random_atom)
            pre_mask_ops.append(mask_random_atom)
        if subgraph_removal:
            structural_ops.append(remove_random_subgraph)
            pre_mask_ops.append(remove_random_subgraph)
        structural_ops.append(_MaskSubgraphPairOp(mask_ratio, contiguous))
        self._structural_ops = tuple(structural_ops)
        self._pre_mask_ops = tuple(pre_mask_ops)
        geom_ops: List[Callable[[GraphData], GraphData]] = []
        if random_rotate or mask_angle or perturb_dihedral:
            geom_ops.append(
                _ApplyGraphAugmentationsOp(
                    random_rotate=random_rotate,
                    mask_angle=mask_angle,
                    perturb_dihedral=perturb_dihedral,
                )
            )
        self._geometric_ops = tuple(geom_ops)
        self._mask_ratio = mask_ratio
        self._contiguous = contiguous
        self._max_retries = max(1, int(max_retries))

    def __call__(self, graph: GraphData) -> Tuple[GraphData, GraphData]:
        for _ in range(self._max_retries):
            views = generate_views(
                graph,
                structural_ops=self._structural_ops,
                geometric_ops=self._geometric_ops,
            )
            if len(views) >= 2:
                ctx, tgt = _ensure_mask_pair(graph, views[0], views[1], self._mask_ratio)
                ctx = _clone_graph_data(ctx)
                tgt = _clone_graph_data(tgt)
                x_ctx = getattr(ctx, "x", None)
                x_tgt = getattr(tgt, "x", None)
                if x_ctx is not None and x_tgt is not None:
                    if getattr(x_ctx, "shape", (0,))[0] > 0 and getattr(x_tgt, "shape", (0,))[0] > 0:
                        return ctx, tgt

        base = _clone_graph_data(graph)
        for op in self._pre_mask_ops:
            base = op(base)
        ctx_graph, tgt_graph = mask_subgraph(
            base, self._mask_ratio, self._contiguous
        )
        ctx_graph, tgt_graph = _ensure_mask_pair(base, ctx_graph, tgt_graph, self._mask_ratio)
        ctx_graph = _clone_graph_data(ctx_graph)
        tgt_graph = _clone_graph_data(tgt_graph)
        for op in self._geometric_ops:
            ctx_graph = op(ctx_graph)
            tgt_graph = op(tgt_graph)
        return ctx_graph, tgt_graph


class _ContrastiveAugmentor:
    """Produce two augmented graph views for contrastive training."""

    def __init__(
        self,
        *,
        mask_ratio: float,
        random_rotate: bool,
        mask_angle: bool,
        perturb_dihedral: bool,
        bond_deletion: bool,
        atom_masking: bool,
        subgraph_removal: bool,
    ) -> None:
        structural_ops: List[Callable[[GraphData], GraphData]] = []
        if bond_deletion:
            structural_ops.append(delete_random_bond)
        if atom_masking:
            structural_ops.append(mask_random_atom)
        if subgraph_removal:
            structural_ops.append(remove_random_subgraph)
        if mask_ratio and mask_ratio > 0.0:
            structural_ops.append(_MaskSubgraphContextOp(mask_ratio, contiguous=False))
        self._structural_ops: Tuple[Callable[[GraphData], GraphData], ...] = tuple(structural_ops)

        geom_ops: List[Callable[[GraphData], GraphData]] = []
        if random_rotate or mask_angle or perturb_dihedral:
            geom_ops.append(
                _ApplyGraphAugmentationsOp(
                    random_rotate=random_rotate,
                    mask_angle=mask_angle,
                    perturb_dihedral=perturb_dihedral,
                )
            )
        self._geometric_ops: Tuple[Callable[[GraphData], GraphData], ...] = tuple(geom_ops)

    def __call__(self, graph: GraphData) -> Tuple[GraphData, GraphData]:
        return self._generate_view(graph), self._generate_view(graph)

    def _generate_view(self, graph: GraphData) -> GraphData:
        views = generate_views(
            graph,
            structural_ops=self._structural_ops,
            geometric_ops=self._geometric_ops,
        )
        if views:
            return views[0]
        return _clone_graph_data(graph)


class _AugmentedPairDataset(Sequence[Tuple[GraphData, GraphData]]):
    """Wrap a sequence of graphs to yield augmented pairs per sample."""

    def __init__(
        self,
        graphs: Sequence[GraphData],
        augmenter: Callable[[GraphData], Tuple[GraphData, GraphData]],
    ) -> None:
        self._graphs = graphs
        self._augmenter = augmenter

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Tuple[GraphData, GraphData]:
        return self._augmenter(self._graphs[idx])


def _collate_graph_pair(
    batch: Sequence[Tuple[GraphData, GraphData]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not batch:
        empty = _collate_graph_batch([])
        return empty, empty
    ctx_graphs, tgt_graphs = zip(*batch)
    return _collate_graph_batch(ctx_graphs), _collate_graph_batch(tgt_graphs)


def _build_graph_dataloader(
    data_source: Sequence[Any],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
    multiprocessing_context: Optional[Any] = None,
) -> DataLoader:
    """Construct a ``DataLoader`` that yields packed ``GraphBatch`` states."""

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn or _collate_graph_batch,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(data_source, **loader_kwargs)


def _is_pin_memory_failure(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` came from the pin-memory worker."""

    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        message = " ".join(str(arg) for arg in getattr(cur, "args", ()))
        if (
            "Pin memory thread exited unexpectedly" in message
            or "received 0 items of ancdata" in message
        ):
            return True
        next_exc = getattr(cur, "__cause__", None)
        if next_exc is None:
            next_exc = getattr(cur, "__context__", None)
        cur = next_exc
    return False


def _is_too_many_open_files(exc: BaseException) -> bool:
    """Detect ``OSError(EMFILE)`` buried in a ``RuntimeError`` chain."""

    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError) and getattr(cur, "errno", None) == errno.EMFILE:
            return True
        message = " ".join(str(arg) for arg in getattr(cur, "args", ()))
        if "Too many open files" in message:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _ensure_file_system_sharing_strategy() -> None:
    """Switch PyTorch multiprocessing to ``file_system`` sharing when possible."""

    mp = getattr(torch, "multiprocessing", None)
    if mp is None:  # pragma: no cover - depends on torch build
        return
    get_strategy = getattr(mp, "get_sharing_strategy", None)
    set_strategy = getattr(mp, "set_sharing_strategy", None)
    if not callable(get_strategy) or not callable(set_strategy):  # pragma: no cover - backend dependent
        return
    try:
        if get_strategy() != "file_system":
            set_strategy("file_system")
    except RuntimeError:  # pragma: no cover - backend dependent
        pass


def _backoff_data_loader_workers(
    persistent_workers: bool, prefetch_factor: Optional[int]
) -> Tuple[bool, bool, Optional[int]]:
    """Reduce worker resource usage when recovering from ``EMFILE`` failures."""

    next_persistent_workers = persistent_workers
    next_prefetch_factor = prefetch_factor
    changed = False

    if persistent_workers:
        next_persistent_workers = False
        changed = True

    if prefetch_factor and prefetch_factor > 1:
        reduced_prefetch = max(1, prefetch_factor // 2)
        if reduced_prefetch != prefetch_factor:
            next_prefetch_factor = reduced_prefetch
            changed = True

    return changed, next_persistent_workers, next_prefetch_factor


def _backoff_num_workers(num_workers: int) -> int:
    """Gradually reduce the number of DataLoader workers when needed."""

    if num_workers <= 1:
        return 0
    return max(0, num_workers // 2)


def _graph_to_tensors(
    g: GraphData, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert GraphData -> (x, adj) tensors on `device`.
    x: [N, F] float32; adj: [N, N] float32, symmetric 0/1.
    """
    x = torch.as_tensor(g.x, dtype=torch.float32, device=device)
    n = int(x.size(0))
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)
    if g.edge_index is not None:
        ei = torch.as_tensor(g.edge_index, dtype=torch.long, device=device)
        # normalize to [2, E]
        if ei.dim() == 2 and ei.size(0) != 2 and ei.size(1) == 2:
            ei = ei.t()
        if ei.numel() > 0:
            adj[ei[0], ei[1]] = 1.0
            adj[ei[1], ei[0]] = 1.0
    return x, adj


def _encode(encoder: nn.Module, g: GraphData, device: torch.device) -> torch.Tensor:
    """
    Call `encoder` in a signature-agnostic way:
      1) try encoder(g)
      2) fallback to encoder(x, adj) if required
    """
    try:
        return encoder(g)
    except TypeError:
        x, adj = _graph_to_tensors(g, device)
        return encoder(x, adj)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.unsqueeze(0)




def train_jepa(
    dataset: GraphDataset,
    encoder: nn.Module,
    ema_encoder: nn.Module,
    predictor: nn.Module,
    ema,
    epochs: int = 100,
    batch_size: int = 256,
    mask_ratio: float = 0.15,
    contiguous: bool = False,
    lr: float = 1e-4,
    device: str = "cuda",
    devices: int = 1,
    reg_lambda: float = 1e-4,
    use_wandb: bool = False,
    wandb_project: str = "m-jepa",
    wandb_tags: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    ckpt_every: int = 10,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
    use_amp: bool = True,
    random_rotate: bool = False,
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
    bond_deletion: bool = False,
    atom_masking: bool = False,
    subgraph_removal: bool = False,
    resume_from: Optional[str] = None,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
    disable_tqdm: bool = False,
    num_workers=0, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=4, 
    bf16=False,
    compile_models: bool = True,
    **unused
) -> List[float]:
    ddp_backend = os.getenv("DDP_BACKEND")  # optional override
    distributed = (devices > 1) and init_distributed(ddp_backend)
    device_t = torch.device(device)
    pin_memory_enabled = bool(
        pin_memory and device_t.type == "cuda" and torch.cuda.is_available()
    )
    if pin_memory and not pin_memory_enabled:
        warnings.warn(
            "pin_memory=True requested but no CUDA device is active; disabling pinned-memory dataloader.",
            RuntimeWarning,
            stacklevel=2,
        )
    if num_workers > 0:
        _ensure_file_system_sharing_strategy()
        normalized_prefetch, bad_prefetch = normalize_prefetch_factor(prefetch_factor)
        if bad_prefetch is not None and is_main_process():
            logger.warning(
                "prefetch_factor=%s is not positive; clamping to %s so DataLoader workers can start.",
                bad_prefetch,
                normalized_prefetch,
            )
        prefetch_factor = normalized_prefetch
    worker_count = int(num_workers) if num_workers else 0
    prefetch_budget = int(prefetch_factor) if isinstance(prefetch_factor, (int, float)) else 0
    if worker_count > 0:
        prefetch_budget = max(prefetch_budget, 2)
    min_fd_budget = max(4096, 1024 + 128 * max(worker_count, 1) * max(prefetch_budget, 1))
    _ensure_open_file_limit(min_fd_budget)
    active_persistent_workers = bool(num_workers) and persistent_workers
    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch
    planned_batches = total_steps
    if max_batches > 0:
        planned_batches = min(planned_batches, max_batches)
    compile_requested = compile_models
    compile_models = _should_compile_models(
        compile_models,
        device_t,
        planned_batches,
    )
    if compile_requested and not compile_models and device_t.type != "cpu":
        logger.debug(
            "Skipping torch.compile because planned batch budget (%d) is below the warm-up threshold (%d).",
            planned_batches,
            _COMPILE_WARMUP_BATCHES,
        )
    can_compile = (
        compile_models and hasattr(torch, "compile") and device_t.type != "cpu"
    )
    compile_fn = torch.compile if can_compile else None

    def _maybe_compile(module: nn.Module, name: str) -> nn.Module:
        if compile_fn is None:
            return module
        try:
            return compile_fn(module)
        except Exception as exc:  # pragma: no cover - backend dependent
            warnings.warn(
                f"torch.compile failed for {name}: {exc}. Falling back to eager execution.",
                RuntimeWarning,
            )
            return module

    encoder = _maybe_compile(encoder.to(device_t), "encoder")
    predictor = _maybe_compile(predictor.to(device_t), "predictor")
    ema_encoder = ema_encoder.to(device_t).eval()

    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[torch.cuda.current_device()]
            if device_t.type == "cuda"
            else None,
        )
        predictor = nn.parallel.DistributedDataParallel(
            predictor,
            device_ids=[torch.cuda.current_device()]
            if device_t.type == "cuda"
            else None,
        )

    encoder.train()
    predictor.train()

    # --- Torch compatibility shim: some Torch builds (e.g., 1.13)
    # don't expose ``torch._dynamo``
    import torch as _torch

    if not hasattr(_torch, "_dynamo"):

        class _DummyDynamo:
            def disable(self, fn=None, recursive=False):
                # Behave like a no-op decorator or direct passthrough
                if fn is None:

                    def _decorator(f):
                        return f

                    return _decorator
                return fn

        _torch._dynamo = _DummyDynamo()

    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    # AMP setup (bf16 on 4090). GradScaler is only for fp16, not bf16.
    amp_enabled = (device_t.type == "cuda") and (use_amp or bf16)
    _amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = GradScaler(enabled=(use_amp and (not bf16) and device_t.type == "cuda"))


    sch = cosine_with_warmup(opt, warmup_steps, total_steps) if use_scheduler else None
    # EMA momentum schedule: start from current decay, finish near 1.0 for a stable target
    ema_start = float(getattr(ema, "decay", 0.996))
    ema_end   = 0.9999
    wb = maybe_init_wandb(
        use_wandb,
        project=wandb_project,
        config=dict(method="jepa", lr=lr, mask_ratio=mask_ratio, contiguous=contiguous),
        tags=wandb_tags,
    )
    wb_log, wb_finish = _make_wandb_handlers(wb)

    start_epoch = 1
    ckpt: Optional[Dict[str, Any]] = None
    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        if "ema_encoder" in ckpt:
            ema_encoder.load_state_dict(ckpt["ema_encoder"])
        if "predictor" in ckpt:
            predictor.load_state_dict(ckpt["predictor"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and _is_grad_scaler(scaler):
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

    def _unwrap_encoder_module() -> nn.Module:
        return (
            encoder.module
            if isinstance(encoder, nn.parallel.DistributedDataParallel)
            else encoder
        )

    def _refresh_ema_from(model: nn.Module) -> None:
        copy_from_fn = getattr(ema, "copy_from", None)
        if callable(copy_from_fn):
            copy_from_fn(model)
            return
        params = getattr(ema, "params", None)
        if params is None:
            return
        with torch.no_grad():
            for buf, src in zip(params, model.parameters()):
                tensor = src.detach()
                if getattr(ema, "use_fp32", False):
                    tensor = tensor.to(torch.float32)
                else:
                    tensor = tensor.to(buf.dtype)
                buf.copy_(tensor.to(buf.device))

    def _copy_ema_to(target: nn.Module, *, fallback: Optional[nn.Module] = None) -> None:
        copy_fn = getattr(ema, "copy_to", None)
        if callable(copy_fn):
            copy_fn(target)
            return
        params = getattr(ema, "params", None)
        if params is not None:
            with torch.no_grad():
                for dest, buf in zip(target.parameters(), params):
                    dest.data.copy_(buf.to(dest.dtype).to(dest.device))
            return
        if fallback is not None:
            try:
                with torch.no_grad():
                    target.load_state_dict(fallback.state_dict(), strict=False)
            except Exception:
                with torch.no_grad():
                    for dest, src in zip(target.parameters(), fallback.parameters()):
                        dest.data.copy_(src.data.to(dest.dtype).to(dest.device))

    def _sync_ema_encoder() -> None:
        src_model = _unwrap_encoder_module()
        _copy_ema_to(ema_encoder, fallback=src_model)

    if ckpt is not None and "ema_encoder" in ckpt:
        _refresh_ema_from(ema_encoder)
    else:
        _refresh_ema_from(_unwrap_encoder_module())
    _sync_ema_encoder()

    if ckpt is not None and "ema_encoder" in ckpt:
        _refresh_ema_from(ema_encoder)
    else:
        _refresh_ema_from(_unwrap_encoder_module())
    _sync_ema_encoder()

    losses: List[float] = []
    mse = nn.MSELoss()
    step = 0
    if ckpt_path:
        os.makedirs(ckpt_path, exist_ok=True)


    _start_wall = _t.perf_counter()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or (
            (_t.perf_counter() - _start_wall) < time_budget_mins * 60
        )

    # Determine whether to disable the progress bar.  We disable bars either
    # explicitly via disable_tqdm or whenever stdout isn’t a TTY (e.g. tests).
    import sys

    disable_bar = disable_tqdm or not sys.stdout.isatty()
    pbar = (
        tqdm.tqdm(
            total=epochs * steps_per_epoch,
            desc=f"Epoch {start_epoch}/{epochs}",
            leave=False,
            disable=disable_bar,
        )
        if is_main_process() and not disable_bar
        else None
    )
    if pbar is None and is_main_process():
        logger.info(
            "Starting JEPA training for %d epochs (%d steps/epoch)",
            epochs,
            steps_per_epoch,
        )
    total_batches_done = 0
    for ep in range(start_epoch, epochs + 1):
        if max_batches > 0 and total_batches_done >= max_batches:
            if is_main_process():
                logger.info(
                    "Max JEPA batches reached (%d); stopping pretraining.",
                    max_batches,
                )
            break
        if not _time_left():
            if is_main_process():
                wb_log({"early/stop_reason": "time_budget"})
                tqdm.tqdm.write(
                    "Time budget exhausted before next JEPA epoch; stopping."
                )
            break
        # Update the bar description when the epoch changes
        if pbar is not None:
            pbar.set_description(f"Epoch {ep}/{epochs}")

        ep_loss = 0.0

        data_source = (
            list(DistributedSamplerList(dataset.graphs))
            if distributed
            else dataset.graphs
        )
        augmenter = _JEPAAugmentor(
            mask_ratio=mask_ratio,
            contiguous=contiguous,
            random_rotate=random_rotate,
            mask_angle=mask_angle,
            perturb_dihedral=perturb_dihedral,
            bond_deletion=bond_deletion,
            atom_masking=atom_masking,
            subgraph_removal=subgraph_removal,
        )
        pair_dataset = _AugmentedPairDataset(data_source, augmenter)
        loader_pin_memory = pin_memory_enabled
        loader_persistent_workers = active_persistent_workers
        loader_prefetch_factor = prefetch_factor
        loader_num_workers = num_workers
        while True:
            ep_loss = 0.0
            epoch_batches = 0
            hit_batch_cap = False
            dataloader = _build_graph_dataloader(
                pair_dataset,
                batch_size=batch_size,
                num_workers=loader_num_workers,
                pin_memory=loader_pin_memory,
                persistent_workers=loader_persistent_workers,
                prefetch_factor=loader_prefetch_factor,
                collate_fn=_collate_graph_pair,
            )

            try:
                for ctx_state, tgt_state in dataloader:
                    ctx_batch = GraphBatch.from_packed(ctx_state)
                    tgt_batch = GraphBatch.from_packed(tgt_state)
                    if max_batches > 0 and total_batches_done >= max_batches:
                        hit_batch_cap = True
                        break
                    if not _time_left():
                        if is_main_process():
                            tqdm.tqdm.write(
                                "Time budget exhausted during JEPA epoch; breaking."
                            )
                        break
                    if ctx_batch.num_graphs == 0 or tgt_batch.num_graphs == 0:
                        continue

                    ctx_batch = _move_graph_batch_to_device(ctx_batch, device_t)
                    tgt_batch = _move_graph_batch_to_device(tgt_batch, device_t)

                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=amp_enabled,
                        dtype=_amp_dtype,
                    ):
                        h_c_nodes = _ensure_2d(_encode_graph(encoder, ctx_batch))
                        with torch.no_grad():
                            h_t_nodes = _ensure_2d(_encode_graph(ema_encoder, tgt_batch))

                    h_c_g = _ensure_2d(_pool_graph_emb(h_c_nodes, ctx_batch))
                    h_t_g = _ensure_2d(_pool_graph_emb(h_t_nodes, tgt_batch))

                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=amp_enabled,
                        dtype=_amp_dtype,
                    ):
                        pred = predictor(h_c_g)

                        if pred.dim() == 1 and h_t_g.dim() == 2 and h_t_g.size(0) == 1:
                            pred = pred.unsqueeze(0)
                        if h_t_g.dim() == 1 and pred.dim() == 2 and pred.size(0) == 1:
                            h_t_g = h_t_g.unsqueeze(0)

                        l2_reg = sum((p**2).sum() for p in predictor.parameters())
                        loss = mse(pred, h_t_g) + reg_lambda * l2_reg

                    opt.zero_grad(set_to_none=True)
                    _step_optimizer(loss, opt, scaler=scaler, scheduler=sch)

                    lv = float(loss.detach().cpu().item())
                    ep_loss += lv
                    step += 1
                    if hasattr(ema, "set_decay"):
                        alpha = min(1.0, step / float(total_steps)) if total_steps > 0 else 1.0
                        w = 0.5 * (1.0 - math.cos(math.pi * alpha))
                        ema_now = ema_start + (ema_end - ema_start) * w
                        ema.set_decay(ema_now)
                    _enc = (
                        encoder.module
                        if isinstance(encoder, nn.parallel.DistributedDataParallel)
                        else encoder
                    )
                    ema.update(_enc)
                    _sync_ema_encoder()
                    wb_log(
                        {
                            "train/jepa_loss": lv,
                            "lr": float(opt.param_groups[0]["lr"]),
                            "step": step,
                            "epoch": ep,
                        }
                    )
                    epoch_batches += 1
                    total_batches_done += 1
                    if pbar is not None:
                        pbar.update(1)
            except (RuntimeError, OSError) as exc:
                if (
                    epoch_batches == 0
                    and loader_pin_memory
                    and _is_pin_memory_failure(exc)
                ):
                    if is_main_process():
                        logger.warning(
                            "Pinned-memory DataLoader failed; retrying without pinned memory. %s",
                            exc,
                        )
                    loader_pin_memory = False
                    loader_persistent_workers = False
                    pin_memory_enabled = False
                    active_persistent_workers = False
                    continue
                if (
                    epoch_batches == 0
                    and loader_num_workers > 0
                    and _is_too_many_open_files(exc)
                ):
                    (
                        strategy_changed,
                        next_persistent_workers,
                        next_prefetch_factor,
                    ) = _backoff_data_loader_workers(
                        loader_persistent_workers, loader_prefetch_factor
                    )
                    next_num_workers = _backoff_num_workers(loader_num_workers)
                    retry = strategy_changed or next_num_workers != loader_num_workers
                    if next_persistent_workers != loader_persistent_workers:
                        loader_persistent_workers = next_persistent_workers
                        retry = True
                        if not next_persistent_workers:
                            active_persistent_workers = False
                    if next_prefetch_factor != loader_prefetch_factor:
                        loader_prefetch_factor = next_prefetch_factor
                        retry = True
                    if next_num_workers != loader_num_workers:
                        loader_num_workers = next_num_workers
                        if loader_num_workers == 0:
                            loader_persistent_workers = False
                            active_persistent_workers = False
                    _ensure_file_system_sharing_strategy()
                    if retry:
                        if is_main_process():
                            logger.warning(
                                "DataLoader workers exhausted file descriptors; retrying with num_workers=%s, persistent_workers=%s, prefetch_factor=%s",  # noqa: E501
                                loader_num_workers,
                                loader_persistent_workers,
                                loader_prefetch_factor,
                            )
                        continue
                raise
            break

        pin_memory_enabled = loader_pin_memory
        active_persistent_workers = (
            loader_persistent_workers and loader_num_workers > 0
        )
        prefetch_factor = loader_prefetch_factor
        num_workers = loader_num_workers

        ep_loss /= max(1, epoch_batches)
        if is_main_process():
            losses.append(ep_loss)
            wb_log({"epoch/jepa_loss": ep_loss, "epoch": ep})
            if pbar is None:
                logger.info(
                    "Epoch %d/%d finished: JEPA loss %.6f",
                    ep,
                    epochs,
                    ep_loss,
                )
            if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
                save_checkpoint(
                    os.path.join(ckpt_path, f"jepa_ep{ep:04d}.pt"),
                    encoder=(encoder.module.state_dict() if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder.state_dict()),
                    ema_encoder=ema_encoder.state_dict(),
                    predictor=predictor.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=(scaler.state_dict() if _is_grad_scaler(scaler) else None),
                    epoch=ep,
                )

        if hit_batch_cap and max_batches > 0 and total_batches_done >= max_batches:
            if is_main_process():
                logger.info(
                    "Max JEPA batches reached (%d); stopping pretraining.",
                    max_batches,
                )
            break

    # Expose the averaged weights to downstream consumers (fine-tuning, checkpointing)
    _copy_ema_to(_unwrap_encoder_module(), fallback=ema_encoder)
    _sync_ema_encoder()


    if pbar is not None:
        # Close the progress bar after training
        pbar.close()
    wb_finish()
    if distributed:
        cleanup()
    return losses


def train_contrastive(
    dataset: GraphDataset,
    encoder: nn.Module,
    projection_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 64,
    mask_ratio: float = 0.15,
    lr: float = 1e-4,
    device: str = "cuda",
    devices: int = 1,
    temperature: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "m-jepa",
    wandb_tags: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    ckpt_every: int = 10,
    use_scheduler: bool = True,
    warmup_steps: int = 1000,
    use_amp: bool = True,
    random_rotate: bool = False,
    # gemoetric augmentations (contrastive-only)
    mask_angle: bool = False,
    perturb_dihedral: bool = False,
    resume_from: Optional[str] = None,
    # structural augmentations (contrastive-only)
    bond_deletion: bool = False,
    atom_masking: bool = False,
    subgraph_removal: bool = False,
    *,
    max_batches: int = 0,
    time_budget_mins: int = 0,
    disable_tqdm: bool = False,
    num_workers=0, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=4, 
    bf16=False, 
    **unused
) -> List[float]:
    ddp_backend = os.getenv("DDP_BACKEND")  # optional override
    distributed = (devices > 1) and init_distributed(ddp_backend)
    device_t = torch.device(device)
    pin_memory_enabled = bool(
        pin_memory and device_t.type == "cuda" and torch.cuda.is_available()
    )
    if pin_memory and not pin_memory_enabled:
        warnings.warn(
            "pin_memory=True requested but no CUDA device is active; disabling pinned-memory dataloader.",
            RuntimeWarning,
            stacklevel=2,
        )
    if num_workers > 0:
        _ensure_file_system_sharing_strategy()
        normalized_prefetch, bad_prefetch = normalize_prefetch_factor(prefetch_factor)
        if bad_prefetch is not None and is_main_process():
            logger.warning(
                "prefetch_factor=%s is not positive; clamping to %s so DataLoader workers can start.",
                bad_prefetch,
                normalized_prefetch,
            )
        prefetch_factor = normalized_prefetch
    active_persistent_workers = bool(num_workers) and persistent_workers
    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder.to(device_t),
            device_ids=[torch.cuda.current_device()]
            if device_t.type == "cuda"
            else None,
        )
        proj = nn.Sequential(
            nn.Linear(256, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        ).to(device_t)
        if device_t.type == "cuda":
            from torch.nn.parallel import DistributedDataParallel as DDP
            proj = DDP(proj, device_ids=[torch.cuda.current_device()])
        opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)

        encoder.train()
        proj.train()
    else:
        encoder.to(device_t).train()
        proj = nn.Sequential(
            nn.Linear(256, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        ).to(device_t)
        opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=lr)

    # GradScaler is for fp16; disable when using bf16
    scaler = GradScaler(enabled=(use_amp and (not bf16) and device_t.type == "cuda"))
    # AMP setup (bf16 on 4090)
    amp_enabled = (device_t.type == "cuda") and (use_amp or bf16)
    _amp_dtype = torch.bfloat16 if bf16 else torch.float16

    if len(dataset.graphs) < 2 or batch_size < 2:
        raise ValueError(
            "Contrastive training requires at least two graphs per batch"
    )


    steps_per_epoch = max(1, math.ceil(len(dataset.graphs) / batch_size))
    total_steps = epochs * steps_per_epoch
    sch = cosine_with_warmup(opt, warmup_steps, total_steps) if use_scheduler else None
    wb = maybe_init_wandb(
        use_wandb,
        project=wandb_project,
        config=dict(method="contrastive", lr=lr, mask_ratio=mask_ratio),
        tags=wandb_tags,
    )
    wb_log, wb_finish = _make_wandb_handlers(wb)

    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        if "projector" in ckpt:
            proj.load_state_dict(ckpt["projector"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and _is_grad_scaler(scaler):
            scaler.load_state_dict(ckpt["scaler"])

    losses: List[float] = []
    step = 0
    if ckpt_path:
        os.makedirs(ckpt_path, exist_ok=True)

    _start_wall = _t.perf_counter()

    def _time_left() -> bool:
        return (time_budget_mins <= 0) or ((_t.perf_counter() - _start_wall) < time_budget_mins * 60)

    # Determine whether to disable the progress bar.  Disable when disable_tqdm
    # is set or stdout isn’t a TTY (tests).
    import sys

    start_epoch = 1
    disable_bar = disable_tqdm or not sys.stdout.isatty()
    pbar = (
        tqdm.tqdm(
            total=epochs * steps_per_epoch,
            desc=f"Epoch {start_epoch}/{epochs}",
            leave=False,
            disable=disable_bar,
        )
        if is_main_process() and not disable_bar
        else None
    )
    if pbar is None and is_main_process():
        logger.info(
            "Starting contrastive training for %d epochs (%d steps/epoch)",
            epochs,
            steps_per_epoch,
        )

    total_batches_done = 0

    for ep in range(start_epoch, epochs + 1):
        if max_batches > 0 and total_batches_done >= max_batches:
            if is_main_process():
                logger.info(
                    "Max contrastive batches reached (%d); stopping pretraining.",
                    max_batches,
                )
            break
        ep_loss = 0.0

        # Update the bar description when the epoch changes
        if pbar is not None:
            pbar.set_description(f"Epoch {ep}/{epochs}")

        data_source = (
            list(DistributedSamplerList(dataset.graphs))
            if distributed
            else dataset.graphs
        )
        augmenter = _ContrastiveAugmentor(
            mask_ratio=mask_ratio,
            random_rotate=random_rotate,
            mask_angle=mask_angle,
            perturb_dihedral=perturb_dihedral,
            bond_deletion=bond_deletion,
            atom_masking=atom_masking,
            subgraph_removal=subgraph_removal,
        )
        pair_dataset = _AugmentedPairDataset(data_source, augmenter)
        loader_pin_memory = pin_memory_enabled
        loader_persistent_workers = active_persistent_workers
        loader_prefetch_factor = prefetch_factor
        loader_num_workers = num_workers
        while True:
            ep_loss = 0.0
            epoch_batches = 0
            hit_batch_cap = False
            dataloader = _build_graph_dataloader(
                pair_dataset,
                batch_size=batch_size,
                num_workers=loader_num_workers,
                pin_memory=loader_pin_memory,
                persistent_workers=loader_persistent_workers,
                prefetch_factor=loader_prefetch_factor,
                collate_fn=_collate_graph_pair,
            )

            try:
                for view1_state, view2_state in dataloader:
                    view1_batch = GraphBatch.from_packed(view1_state)
                    view2_batch = GraphBatch.from_packed(view2_state)
                    if max_batches > 0 and total_batches_done >= max_batches:
                        hit_batch_cap = True
                        break
                    if not _time_left():
                        if is_main_process():
                            tqdm.tqdm.write(
                                "Time budget exhausted during contrastive epoch; breaking."
                            )
                        break
                    if view1_batch.num_graphs == 0 or view2_batch.num_graphs == 0:
                        continue

                    view1_batch = _move_graph_batch_to_device(view1_batch, device_t)
                    view2_batch = _move_graph_batch_to_device(view2_batch, device_t)

                    with torch.amp.autocast(
                        device_type="cuda", enabled=amp_enabled, dtype=_amp_dtype
                    ):
                        h1_nodes = _ensure_2d(_encode_graph(encoder, view1_batch))
                        h2_nodes = _ensure_2d(_encode_graph(encoder, view2_batch))

                        g1_pool = _ensure_2d(_pool_graph_emb(h1_nodes, view1_batch))
                        g2_pool = _ensure_2d(_pool_graph_emb(h2_nodes, view2_batch))

                        if (
                            isinstance(proj[0], nn.Linear)
                            and proj[0].in_features != g1_pool.size(1)
                        ):
                            proj[0] = nn.Linear(g1_pool.size(1), proj[0].out_features).to(device_t)
                            opt = optim.Adam(
                                list(encoder.parameters()) + list(proj.parameters()),
                                lr=lr,
                            )

                        z1 = torch.nan_to_num(F.normalize(proj(g1_pool), dim=-1))
                        z2 = torch.nan_to_num(F.normalize(proj(g2_pool), dim=-1))

                    N = z1.size(0)
                    if N < 2:
                        raise ValueError(
                            "Contrastive loss requires at least two graphs per batch"
                        )

                    with torch.amp.autocast(
                        device_type="cuda", enabled=amp_enabled, dtype=_amp_dtype
                    ):
                        logits_12 = (z1 @ z2.t()) / float(temperature)
                        with torch.no_grad():
                            pos = logits_12.diag().mean().item()
                            neg = (
                                (logits_12.sum() - logits_12.diag().sum())
                                / (N * (N - 1))
                            ).item()
                            wb_log(
                                {"diag/pos_minus_neg": pos - neg, "diag/lnN": math.log(N)},
                                commit=False,
                            )
                        logits_21 = logits_12.t()

                        labels = torch.arange(N, device=device_t)

                        loss = 0.5 * (
                            F.cross_entropy(logits_12, labels)
                            + F.cross_entropy(logits_21, labels)
                        )

                    opt.zero_grad(set_to_none=True)
                    _step_optimizer(loss, opt, scaler=scaler, scheduler=sch)

                    lv = float(loss.detach().cpu().item())
                    ep_loss += lv
                    step += 1
                    wb_log(
                        {
                            "train/contrastive_loss": lv,
                            "lr": float(opt.param_groups[0]["lr"]),
                            "step": step,
                            "epoch": ep,
                        }
                    )
                    epoch_batches += 1
                    total_batches_done += 1
                    if pbar is not None:
                        pbar.update(1)
            except (RuntimeError, OSError) as exc:
                if (
                    epoch_batches == 0
                    and loader_pin_memory
                    and _is_pin_memory_failure(exc)
                ):
                    if is_main_process():
                        logger.warning(
                            "Pinned-memory DataLoader failed; retrying without pinned memory. %s",
                            exc,
                        )
                    loader_pin_memory = False
                    loader_persistent_workers = False
                    pin_memory_enabled = False
                    active_persistent_workers = False
                    continue
                if (
                    epoch_batches == 0
                    and loader_num_workers > 0
                    and _is_too_many_open_files(exc)
                ):
                    (
                        strategy_changed,
                        next_persistent_workers,
                        next_prefetch_factor,
                    ) = _backoff_data_loader_workers(
                        loader_persistent_workers, loader_prefetch_factor
                    )
                    next_num_workers = _backoff_num_workers(loader_num_workers)
                    retry = strategy_changed or next_num_workers != loader_num_workers
                    if next_persistent_workers != loader_persistent_workers:
                        loader_persistent_workers = next_persistent_workers
                        retry = True
                        if not next_persistent_workers:
                            active_persistent_workers = False
                    if next_prefetch_factor != loader_prefetch_factor:
                        loader_prefetch_factor = next_prefetch_factor
                        retry = True
                    if next_num_workers != loader_num_workers:
                        loader_num_workers = next_num_workers
                        if loader_num_workers == 0:
                            loader_persistent_workers = False
                            active_persistent_workers = False
                    _ensure_file_system_sharing_strategy()
                    if retry:
                        if is_main_process():
                            logger.warning(
                                "Contrastive DataLoader workers exhausted file descriptors; retrying with num_workers=%s, persistent_workers=%s, prefetch_factor=%s",  # noqa: E501
                                loader_num_workers,
                                loader_persistent_workers,
                                loader_prefetch_factor,
                            )
                        continue
                raise
            break

        pin_memory_enabled = loader_pin_memory
        active_persistent_workers = (
            loader_persistent_workers and loader_num_workers > 0
        )
        prefetch_factor = loader_prefetch_factor
        num_workers = loader_num_workers
        ep_loss /= max(1, epoch_batches)
        if is_main_process():
            losses.append(ep_loss)
            wb_log({"epoch/contrastive_loss": ep_loss, "epoch": ep})
            if pbar is None:
                logger.info(
                    "Epoch %d/%d finished: contrastive loss %.6f",
                    ep,
                    epochs,
                    ep_loss,
                )
            if ckpt_path and (ep % ckpt_every == 0 or ep == epochs):
                save_checkpoint(
                    os.path.join(ckpt_path, f"contrastive_ep{ep:04d}.pt"),
                    encoder=(encoder.module.state_dict() if isinstance(encoder, nn.parallel.DistributedDataParallel) else encoder.state_dict()),
                    projector=proj.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=(scaler.state_dict() if _is_grad_scaler(scaler) else None),
                    epoch=ep,
                )
        if hit_batch_cap and max_batches > 0 and total_batches_done >= max_batches:
            if is_main_process():
                logger.info(
                    "Max contrastive batches reached (%d); stopping pretraining.",
                    max_batches,
                )
            break
    if pbar is not None:
        # Close the progress bar after training
        pbar.close()
    wb_finish()
    if distributed:
        cleanup()

    return losses
