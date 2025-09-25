from __future__ import annotations

import pytest


def test_iterator_finalizer_handles_missing_workers_status() -> None:
    pytest.importorskip("torch")

    # Importing ``utils.dataloader`` applies the patch as a side effect.
    import utils.dataloader  # noqa: F401  # pylint: disable=unused-import

    from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

    if _MultiProcessingDataLoaderIter is None:  # pragma: no cover - defensive
        pytest.skip("multiprocessing DataLoader iterator is unavailable")

    ghost_iter = object.__new__(_MultiProcessingDataLoaderIter)

    # ``__del__`` should be a no-op even though the instance is missing the
    # private ``_workers_status`` attribute that upstream PyTorch expects.
    _MultiProcessingDataLoaderIter.__del__(ghost_iter)
