import sys
import types


def test_pretrain_main(monkeypatch, tmp_path):
    # stub out heavy parquet loader before importing module
    loader_mod = types.ModuleType("data.parquet_loader")

    class DummyLoader:
        def __init__(self, n):
            self.dataset = [None] * n

    def dummy_load(root, batch_size):
        return DummyLoader(1), DummyLoader(2), DummyLoader(3)

    loader_mod.load_dataloaders = dummy_load
    monkeypatch.setitem(sys.modules, "data.parquet_loader", loader_mod)

    from training import pretrain as p

    class DummyWB:
        def __init__(self):
            self.logged = None
            self.finished = False

        def log(self, data):
            self.logged = data

        def finish(self):
            self.finished = True

    wb_instance = DummyWB()

    def dummy_wandb(enable, **kwargs):
        return wb_instance

    monkeypatch.setattr(p, "maybe_init_wandb", dummy_wandb)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain",
            "--parquet-root",
            str(tmp_path),
            "--batch-size",
            "4",
            "--use-wandb",
        ],
    )

    p.main()

    assert wb_instance.logged == {
        "dataset/train_graphs": 1,
        "dataset/val_graphs": 2,
        "dataset/test_graphs": 3,
    }
    assert wb_instance.finished