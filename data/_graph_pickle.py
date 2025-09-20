"""Helpers to make :class:`data.mdataset.GraphData` robustly picklable."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Type

__all__ = ["register_graph_class", "rebuild_graph_data"]

_CACHED_GRAPH_CLS: Optional[Type[object]] = None


def register_graph_class(cls: Type[object]) -> None:
    """Remember the canonical ``GraphData`` implementation for pickling."""

    global _CACHED_GRAPH_CLS
    _CACHED_GRAPH_CLS = cls


def _resolve_graph_class() -> Type[object]:
    """Return the canonical ``GraphData`` class, importing it on demand."""

    if _CACHED_GRAPH_CLS is not None:
        return _CACHED_GRAPH_CLS

    from importlib import import_module

    module = import_module("data.mdataset")
    graph_cls = getattr(module, "GraphData", None)
    if graph_cls is None:
        raise RuntimeError("GraphData class is not available for reconstruction")
    register_graph_class(graph_cls)
    return graph_cls


def rebuild_graph_data(state: Mapping[str, Any]) -> object:
    """Recreate a ``GraphData`` instance from its serialised mapping."""

    graph_cls = _resolve_graph_class()
    from_state = getattr(graph_cls, "_from_state", None)
    if callable(from_state):
        return from_state(state)

    instance = graph_cls.__new__(graph_cls)  # type: ignore[misc]
    if hasattr(instance, "__setstate__"):
        instance.__setstate__(state)  # type: ignore[attr-defined]
        return instance

    if not isinstance(state, MutableMapping):
        raise TypeError("GraphData state must be a mapping")
    for key, value in state.items():
        setattr(instance, key, value)
    return instance
