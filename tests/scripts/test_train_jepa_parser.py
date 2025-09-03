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
    assert args.hidden_dim == CONFIG["model"]["hidden_dim"]


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
    case_cfg = CONFIG["case_study"]
    assert args.pretrain_epochs == case_cfg["pretrain_epochs"]
    assert args.finetune_epochs == case_cfg["finetune_epochs"]
    assert args.num_top_exclude == case_cfg["num_top_exclude"]
    assert args.hidden_dim == CONFIG["model"]["hidden_dim"]


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
