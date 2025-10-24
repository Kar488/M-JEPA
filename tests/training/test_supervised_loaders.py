from __future__ import annotations

import types

import pytest


pytest.importorskip("torch")


from training import supervised as supervised_mod
from tests.training.test_supervised import DummyDataset, DummyEncoder


class RecordingLoader:
    instances = []
    fail_next = True

    def __init__(self, indices, **kwargs):
        self.indices = list(indices)
        self.kwargs = kwargs
        self.num_workers = kwargs.get("num_workers", 0)
        self._shutdown_calls = 0
        self._iterator = types.SimpleNamespace(_shutdown_workers=self._record_shutdown)
        type(self).instances.append(self)

    def _record_shutdown(self) -> None:
        self._shutdown_calls += 1

    def __iter__(self):
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("Too many open files")

        collate_fn = self.kwargs["collate_fn"]
        batch_size = int(self.kwargs.get("batch_size", 1)) or 1
        for start in range(0, len(self.indices), batch_size):
            batch_indices = self.indices[start : start + batch_size]
            if not batch_indices:
                continue
            yield collate_fn(batch_indices)

    def __len__(self) -> int:
        batch_size = int(self.kwargs.get("batch_size", 1)) or 1
        total = len(self.indices)
        return max(1, (total + batch_size - 1) // batch_size)


def test_loader_rebuild_path_closes_iterators(monkeypatch):
    dataset = DummyDataset([0, 1, 0, 1], feat_dim=2)
    encoder = DummyEncoder(2)

    RecordingLoader.instances = []
    RecordingLoader.fail_next = True

    reset_calls = []

    class DummyMultiProcessingIter:
        @classmethod
        def _reset(cls, loader, *args, **kwargs):
            reset_calls.append(loader)

    monkeypatch.setattr(supervised_mod, "DataLoader", RecordingLoader)
    monkeypatch.setattr(
        supervised_mod.torch.utils.data.dataloader,
        "_MultiProcessingDataLoaderIter",
        DummyMultiProcessingIter,
        raising=False,
    )

    metrics = supervised_mod.train_linear_head(
        dataset,
        encoder,
        "classification",
        epochs=1,
        batch_size=2,
        lr=0.01,
        patience=0,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        cache_graph_embeddings=False,
        persistent_workers=False,
    )

    assert len(RecordingLoader.instances) >= 2, "expected loader rebuild after failure"
    first_loader = RecordingLoader.instances[0]
    assert first_loader._shutdown_calls == 1
    assert not hasattr(first_loader, "_iterator")
    assert first_loader in reset_calls
    assert metrics["head"] is not None
