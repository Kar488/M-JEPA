import types
from pathlib import Path

import pandas as pd

import scripts.commands.grid_search as grid


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args))

    def exception(self, *args, **kwargs):
        self.messages.append(("exception", args))


class DummyDataset:
    def __init__(self, name: str):
        self.name = name
        self.graphs = [object(), object()]

    def __len__(self):
        return len(self.graphs)


def _base_args(tmp_path):
    return types.SimpleNamespace(
        force_refresh=False,
        out_csv=str(tmp_path / "results.csv"),
        best_config_out=str(tmp_path / "best.json"),
        methods=["jepa"],
        contiguities=[0, 1],
        add_3d_options=[0],
        aug_rotate_options=[0],
        aug_mask_angle_options=[0],
        aug_dihedral_options=[0],
        seeds=[1],
        dataset_dir=None,
        unlabeled_dir="unlabeled",
        labeled_dir="labeled",
        label_col="label",
        smiles_col="smiles",
        num_workers=0,
        n_rows_per_file=None,
        sample_unlabeled=0,
        sample_labeled=0,
        no_cache=True,
        cache_dir=None,
        task_type="classification",
        mask_ratios=[0.1],
        hidden_dims=[64],
        num_layers_list=[2],
        gnn_types=["gcn"],
        ema_decays=[0.99],
        pretrain_batch_sizes=[32],
        finetune_batch_sizes=[16],
        pretrain_epochs_options=[1],
        finetune_epochs_options=[1],
        learning_rates=[0.001],
        temperatures=[0.2],
        device="cpu",
        use_wandb=False,
        wandb_project="proj",
        wandb_tags=["tag"],
        ckpt_dir=str(tmp_path / "ckpt"),
        ckpt_every=1,
        use_scheduler=False,
        warmup_steps=0,
        target_pretrain_samples=0,
        max_pretrain_batches=0,
        max_finetune_batches=0,
        time_budget_mins=0,
        force_tqdm=False,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
    )


def test_cmd_grid_search_skips_when_outputs_exist(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.out_csv = str(tmp_path / "results.csv")
    args.best_config_out = str(tmp_path / "best.json")
    Path(args.out_csv).write_text("done")
    Path(args.best_config_out).write_text("done")

    called = {}
    monkeypatch.setattr(grid, "logger", DummyLogger(), raising=False)
    monkeypatch.setattr(grid, "run_grid_search", lambda *a, **k: called.setdefault("called", True), raising=False)

    grid.cmd_grid_search(args)
    assert "called" not in called


def test_cmd_grid_search_runs_and_logs(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.force_refresh = True

    monkeypatch.setattr(grid, "logger", DummyLogger(), raising=False)
    monkeypatch.setattr(grid, "CONFIG", {"finetune": {"seeds": [7]}}, raising=False)
    monkeypatch.setattr(grid, "iter_augmentation_options", lambda *a: [types.SimpleNamespace(flag=True)], raising=False)
    monkeypatch.setattr(grid, "load_directory_dataset", lambda path, **kw: DummyDataset(path), raising=False)

    class DummyWB:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, payload):
            self.logged.append(payload)

        def finish(self):
            self.finished = True

    monkeypatch.setattr(grid, "maybe_init_wandb", lambda *a, **k: DummyWB(), raising=False)

    records = {}

    def fake_run_grid_search(**kwargs):
        records["kwargs"] = kwargs
        df = pd.DataFrame([{"score": 0.5, "config": "a"}])
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
        return df

    monkeypatch.setattr(grid, "run_grid_search", fake_run_grid_search, raising=False)

    grid.cmd_grid_search(args)

    assert "dataset_fn" in records["kwargs"]
    assert Path(args.best_config_out).exists()
    assert Path(args.out_csv).exists()

