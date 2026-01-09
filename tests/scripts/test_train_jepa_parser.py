import argparse
import pytest

from scripts.train_jepa import (
    build_parser,
    CONFIG,
    cmd_pretrain,
    cmd_finetune,
    cmd_evaluate,
    cmd_benchmark,
    cmd_tox21,
    cmd_grid_search,
)

import pytest, argparse
# Suppress argparse usage messages in tests
@pytest.fixture(autouse=True)
def _silence_argparse_usage(monkeypatch):
    def _quiet_error(self, message):
        raise SystemExit(2)
    monkeypatch.setattr(argparse.ArgumentParser, "error", _quiet_error, raising=False)
 

def test_pretrain_parser_defaults_and_handler(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pretrain"])
    args = parser.parse_args([
        "pretrain",
        f"--unlabeled-dir={tmp_path}",
        f"--plot-dir={tmp_path}",
    ])
    assert args.func is cmd_pretrain
    assert args.unlabeled_dir == str(tmp_path)
    assert args.output == "encoder.pt"
    assert args.epochs == CONFIG["pretrain"]["epochs"]
    assert args.batch_size == CONFIG["pretrain"]["batch_size"]
    assert args.lr == CONFIG["pretrain"]["lr"]
    assert args.probe_dataset == CONFIG["pretrain"]["probe_dataset"]
    assert args.probe_interval == CONFIG["pretrain"]["probe_interval"]
    #assert args.gnn_type == CONFIG["model"]["gnn_type"]
    assert args.hidden_dim == CONFIG["model"]["hidden_dim"]
    assert args.dropout == CONFIG["model"]["dropout"]


def test_finetune_parser_defaults_and_handler(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["finetune"])
    args = parser.parse_args([
        "finetune",
        "--labeled-dir",
        str(tmp_path),
        "--encoder",
        "enc.pt",
    ])
    assert args.func is cmd_finetune
    assert args.labeled_dir == str(tmp_path)
    assert args.encoder == "enc.pt"
    assert args.epochs == CONFIG["finetune"]["epochs"]
    assert args.batch_size == CONFIG["finetune"]["batch_size"]
    assert args.lr == CONFIG["finetune"]["lr"]
    assert args.patience == CONFIG["finetune"]["patience"]
    assert args.use_scaffold is False
    assert args.pos_weight is None

    args = parser.parse_args(
        [
            "finetune",
            "--labeled-dir",
            str(tmp_path),
            "--encoder",
            "enc.pt",
            "--pos-class-weight",
            "NR-AR=3.0",
            "--pos-class-weight",
            "1.5",
        ]
    )
    assert args.pos_weight == ["NR-AR=3.0", "1.5"]


def test_evaluate_parser_defaults_and_handler(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    args = parser.parse_args([
        "evaluate",
        "--labeled-dir",
        str(tmp_path),
        "--encoder",
        "enc.pt",
    ])
    assert args.func is cmd_evaluate
    assert args.labeled_dir == str(tmp_path)
    assert args.encoder == "enc.pt"
    assert args.epochs == CONFIG["evaluate"]["epochs"]
    assert args.batch_size == CONFIG["evaluate"]["batch_size"]
    assert args.lr == CONFIG["evaluate"]["lr"]
    assert args.patience == CONFIG["evaluate"]["patience"]


def test_benchmark_parser_defaults_and_handler(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark"])
    args = parser.parse_args([
        "benchmark",
        "--labeled-dir",
        str(tmp_path),
        "--jepa-encoder",
        "jepa.pt",
    ])
    assert args.func is cmd_benchmark
    assert args.labeled_dir == str(tmp_path)
    assert args.test_dir is None
    assert args.jepa_encoder == "jepa.pt"
    assert args.dataset is None
    assert args.task is None
    assert args.epochs == CONFIG["benchmark"]["epochs"]
    assert args.batch_size == CONFIG["benchmark"]["batch_size"]
    assert args.lr == CONFIG["benchmark"]["lr"]
    assert args.patience == CONFIG["benchmark"]["patience"]


def test_tox21_parser_defaults_and_handler(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tox21"])
    args = parser.parse_args([
        "tox21",
        "--csv",
        str(tmp_path / "tox.csv"),
        "--task",
        "NR-AR",
    ])
    assert args.func is cmd_tox21
    assert args.csv == str(tmp_path / "tox.csv")
    assert args.task == "NR-AR"
    assert args.tasks is None
    assert args.dataset == "tox21"
    case_cfg = CONFIG["case_study"]
    assert args.pretrain_epochs == case_cfg["pretrain_epochs"]
    assert args.finetune_epochs == case_cfg["finetune_epochs"]
    assert args.pretrain_lr == case_cfg.get("pretrain_lr", 1e-4)
    assert args.checkpoint_metric == case_cfg.get("checkpoint_metric", "pr_auc")
    #assert args.gnn_type == CONFIG["model"]["gnn_type"]
    assert args.hidden_dim == CONFIG["model"]["hidden_dim"]
    assert args.pretrain_time_budget_mins == 0
    assert args.finetune_time_budget_mins == 0
    assert args._hidden_dim_provided is False
    assert args._num_layers_provided is False
    assert args._gnn_type_provided is False


def test_tox21_help_lists_structural_flags(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tox21", "--help"])
    help_output = capsys.readouterr().out
    for flag in ("--gnn-type", "--hidden-dim", "--num-layers"):
        assert flag in help_output


def test_model_shape_flags_track_provided_state(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "tox21",
        "--csv",
        str(tmp_path / "tox.csv"),
        "--task",
        "NR-AR",
        "--hidden-dim",
        "192",
        "--num-layers",
        "4",
        "--gnn-type",
        "gin",
    ])
    assert args._hidden_dim_provided is True
    assert args._num_layers_provided is True
    assert args._gnn_type_provided is True
    assert args.hidden_dim == 192
    assert args.num_layers == 4
    assert args.gnn_type == "gin"


def test_tox21_parser_accepts_multiple_tasks(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "tox21",
        "--csv",
        str(tmp_path / "tox.csv"),
        "--tasks",
        "NR-AR",
        "NR-ER",
    ])
    assert args.func is cmd_tox21
    assert args.task is None
    assert args.tasks == ["NR-AR", "NR-ER"]


def test_tox21_parser_accepts_hybrid_mode(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "tox21",
        "--csv",
        str(tmp_path / "tox.csv"),
        "--task",
        "NR-AR",
        "--evaluation-mode",
        "hybrid",
    ])
    assert args.evaluation_mode == "hybrid"


def test_grid_search_parser_defaults_and_handler(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "grid-search",
        "--dataset-dir",
        str(tmp_path),
    ])
    assert args.func is cmd_grid_search
    assert args.dataset_dir == str(tmp_path)
    assert args.wandb_project == CONFIG["wandb"]["project"]
    assert args.wandb_tags == CONFIG["wandb"].get("tags", [])
    assert args.target_pretrain_samples == 0
