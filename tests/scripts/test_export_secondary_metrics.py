from types import SimpleNamespace

from scripts.ci import export_best_from_wandb as eb


def _dummy_run():
    run = SimpleNamespace()
    run.summary = {"metrics": {"val_r2": 0.42}}
    run.summary_metrics = {"val_r2_mean": 0.4, "val_brier_mean": 0.3}
    run._attrs = {
        "summaryMetrics": {
            "metrics": {"val_brier": 0.2},
            "val_r2_std": "0.07",
            "val_r2_ci95": 0.9,
            "val_brier_std": "0.01",
            "val_brier_ci95": 0.5,
        }
    }
    run.config = {"training_method": "jepa"}
    run.id = "run1"
    run.name = "run-name"
    run.state = "finished"
    return run


def test_build_run_record_secondary_metrics():
    run = _dummy_run()
    record = eb._build_run_record(run, "sweep1")

    assert record["val_r2"] == 0.42
    assert record["metric_r2"] == 0.42
    assert record["val_r2_mean"] == 0.4
    assert record["val_r2_std"] == 0.07
    assert record["val_r2_ci95"] == 0.9

    assert record["val_brier"] == 0.2
    assert record["val_brier_mean"] == 0.3
    assert record["val_brier_std"] == 0.01
    assert record["val_brier_ci95"] == 0.5
