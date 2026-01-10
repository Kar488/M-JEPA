import csv
import json
from types import SimpleNamespace

import pytest

import scripts.commands.tox21 as tox


def test_wandb_log_and_save_safe():
    logged = []
    saved = []

    class DummyRun:
        def __init__(self):
            self.logs = []

        def log(self, payload):
            self.logs.append(payload)

        def save(self, path):
            saved.append(path)

    class DummyWB:
        def __init__(self):
            self.run = DummyRun()

        def log(self, payload):
            logged.append(payload)

        def save(self, path):
            saved.append(path)

    wb = DummyWB()
    tox._wandb_log_safe(wb, {"metric": 0.5})
    tox._wandb_save_safe(wb, "file.txt")
    assert logged[0]["metric"] == 0.5
    assert saved == ["file.txt"]


def test_coerce_case_study_result_variants():
    class DummyEval:
        def __init__(self):
            self.evaluations = [SimpleNamespace(name="ok")]
            self.threshold_rule = "rule"

    evaluations, rule = tox._coerce_case_study_result(DummyEval())
    assert len(evaluations) == 1 and rule == "rule"

    tuple_result = (0.4, 0.5, 0.6, {"base": 0.1}, {"metric": 0.2})
    evals_tuple, rule_tuple = tox._coerce_case_study_result(tuple_result)
    assert evals_tuple[0].mean_true == 0.4 and rule_tuple is None

    dict_result = {"mean_true": 0.7, "mean_random": 0.3, "mean_pred": 0.1, "metrics": {"roc_auc": 0.8}}
    evals_dict, _ = tox._coerce_case_study_result(dict_result)
    assert evals_dict[0].metrics["roc_auc"] == 0.8

    empty, rule_empty = tox._coerce_case_study_result("invalid")
    assert empty == [] and rule_empty is None


def test_run_single_task_passes_explain_kwargs(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    class DummyResult(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    captured = {}

    def fake_case_study(**kwargs):
        captured["mode"] = kwargs.get("explain_mode")
        captured["config"] = kwargs.get("explain_config")
        eval_obj = SimpleNamespace(
            encoder_source="pretrain_frozen",
            name="pretrain_frozen",
            mean_true=0.5,
            mean_random=0.4,
            mean_pred=0.6,
            metrics={"roc_auc": 0.7},
            benchmark_metric="roc_auc",
            benchmark_threshold=0.65,
            met_benchmark=True,
            manifest_path=str(report_dir / "manifest.json"),
            baseline_means={},
        )
        return DummyResult(
            evaluations=[eval_obj],
            diagnostics={
                "allow_shape_coercion_effective": False,
                "test_predictions": {
                    "indices": [0, 1],
                    "logits": [1.2, -0.3],
                    "probabilities": [0.7, 0.4],
                    "true_labels": [1, 0],
                },
            },
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)
    monkeypatch.setenv("TOX21_EXPLAIN_MODE", "ig")
    monkeypatch.setenv("TOX21_EXPLAIN_STEPS", "11")

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    result = tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured.get("mode") == ["ig"]
    assert captured.get("config", {}).get("output_dir") == str(report_dir)
    assert captured.get("config", {}).get("steps") == 11
    assert captured.get("config", {}).get("task_name") == "NR-AR"
    pred_path = report_dir / "tox21_NR-AR_scores.csv"
    assert pred_path.is_file()
    assert result.get("prediction_csv_path") == str(pred_path)
    assert (
        pred_path.read_text().splitlines()[0]
        == "graph_id,assay,true_label,ensemble_logit,ensemble_probability"
    )
    assert result.get("stage_payload", {}).get("prediction_csv") == str(pred_path)


def test_run_single_task_applies_task_overrides(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_case_study(**kwargs):
        captured.update(kwargs)
        eval_obj = SimpleNamespace(
            encoder_source="pretrain_frozen",
            name="pretrain_frozen",
            mean_true=0.5,
            mean_random=0.4,
            mean_pred=0.6,
            metrics={"roc_auc": 0.7},
            benchmark_metric="roc_auc",
            benchmark_threshold=0.65,
            met_benchmark=True,
            manifest_path=str(report_dir / "manifest.json"),
            baseline_means={},
        )
        return SimpleNamespace(
            evaluations=[eval_obj],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=2,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=1e-4,
        encoder_lr=1e-6,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        oversample_minority=False,
        use_focal_loss=False,
        dynamic_pos_weight=False,
        focal_gamma=2.0,
        layerwise_decay=None,
        threshold_metric="roc_auc",
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    task_hparam_map = {
        "nr-ar": {
            "head_lr": 3e-4,
            "encoder_lr": 1e-5,
            "threshold_metric": "pr_auc",
            "checkpoint_metric": "pr_auc",
            "use_focal_loss": True,
            "dynamic_pos_weight": True,
            "pos_weight": 7,
        }
    }

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        calibrate_per_head=False,
        task_hparam_map=task_hparam_map,
        hybrid_defaults={},
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured["head_lr"] == 3e-4
    assert captured["encoder_lr"] == 1e-5
    assert captured["threshold_metric"] == "pr_auc"
    assert captured["checkpoint_metric"] == "pr_auc"
    assert captured["use_focal_loss"] is True
    assert captured["dynamic_pos_weight"] is True
    assert captured["pos_class_weight"] is not None


def test_run_single_task_resolves_calibration_method(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_case_study(**kwargs):
        captured.update(kwargs)
        eval_obj = SimpleNamespace(
            encoder_source="pretrain_frozen",
            name="pretrain_frozen",
            mean_true=0.5,
            mean_random=0.4,
            mean_pred=0.6,
            metrics={"roc_auc": 0.7},
            benchmark_metric="roc_auc",
            benchmark_threshold=0.65,
            met_benchmark=True,
            manifest_path=str(report_dir / "manifest.json"),
            baseline_means={},
        )
        return SimpleNamespace(
            evaluations=[eval_obj],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=2,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=1e-4,
        encoder_lr=1e-6,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        oversample_minority=False,
        use_focal_loss=False,
        dynamic_pos_weight=False,
        focal_gamma=2.0,
        layerwise_decay=None,
        threshold_metric="roc_auc",
        calibration_method="isotonic",
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    task_hparam_map = {"nr-ar": {"calibration_method": "temperature"}}

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        calibrate_per_head=False,
        task_hparam_map=task_hparam_map,
        hybrid_defaults={},
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured["calibration_method"] == "temperature"


def test_run_single_task_accepts_isotonic_calibration(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_case_study(**kwargs):
        captured.update(kwargs)
        eval_obj = SimpleNamespace(
            encoder_source="pretrain_frozen",
            name="pretrain_frozen",
            mean_true=0.5,
            mean_random=0.4,
            mean_pred=0.6,
            metrics={"roc_auc": 0.7},
            benchmark_metric="roc_auc",
            benchmark_threshold=0.65,
            met_benchmark=True,
            manifest_path=str(report_dir / "manifest.json"),
            baseline_means={},
        )
        return SimpleNamespace(
            evaluations=[eval_obj],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=2,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=1e-4,
        encoder_lr=1e-6,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        oversample_minority=False,
        use_focal_loss=False,
        dynamic_pos_weight=False,
        focal_gamma=2.0,
        layerwise_decay=None,
        threshold_metric="roc_auc",
        calibration_method=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    task_hparam_map = {"nr-ar": {"calibration_method": "isotonic"}}

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        calibrate_per_head=False,
        task_hparam_map=task_hparam_map,
        hybrid_defaults={},
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured["calibration_method"] == "isotonic"


def test_run_single_task_exports_seed_metrics_and_reliability(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    labels = [1, 0, 1]
    ensemble_logits = [0.0, 1.0, -1.0]
    ensemble_probs = [0.5, 0.73, 0.27]
    seed_logits = [[0.0, 1.0, -1.0], [0.5, 0.5, -0.5]]
    seed_probs = [[0.5, 0.73, 0.27], [0.62, 0.62, 0.38]]

    ensemble_metrics = tox._compute_seed_metrics(labels, ensemble_probs)

    class DummyWB:
        def __init__(self):
            self.logs = []

        def log(self, payload):
            self.logs.append(payload)

        def save(self, _path):
            return None

    wb = DummyWB()

    def fake_case_study(**_kwargs):
        eval_obj = SimpleNamespace(
            encoder_source="pretrain_frozen",
            name="pretrain_frozen",
            mean_true=0.5,
            mean_random=0.4,
            mean_pred=0.6,
            metrics=ensemble_metrics,
            benchmark_metric="roc_auc",
            benchmark_threshold=0.65,
            met_benchmark=True,
            manifest_path=str(report_dir / "manifest.json"),
            baseline_means={},
        )
        diagnostics = {
            "test_predictions": {
                "indices": [0, 1, 2],
                "logits": ensemble_logits,
                "probabilities": ensemble_probs,
                "true_labels": labels,
            },
            "seed_predictions": {
                "seeds": [0, 1],
                "indices": [0, 1, 2],
                "true_labels": labels,
                "logits": seed_logits,
                "probabilities": seed_probs,
            },
        }
        return SimpleNamespace(
            evaluations=[eval_obj],
            diagnostics=diagnostics,
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=2,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    result = tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=wb,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    prediction_seed_csv = report_dir / "tox21_NR-AR_scores_by_seed.csv"
    reliability_bins = report_dir / "tox21_NR-AR_reliability_bins.json"
    manifest_path = report_dir / "run_manifest_NR-AR.json"
    assert prediction_seed_csv.is_file()
    assert reliability_bins.is_file()
    assert manifest_path.is_file()

    manifest_payload = json.loads(manifest_path.read_text())
    assert "prediction_seed_csv" in manifest_payload["reports"]
    assert "reliability_bins" in manifest_payload["reports"]

    payload = json.loads((report_dir / "tox21_NR-AR.json").read_text())
    seed_metrics = {
        "0": tox._compute_seed_metrics(labels, seed_probs[0]),
        "1": tox._compute_seed_metrics(labels, seed_probs[1]),
    }
    seed_std = tox._compute_seed_metric_std(seed_metrics)
    assert payload["ensemble"]["metrics"]["roc_auc"] == pytest.approx(ensemble_metrics["roc_auc"])
    assert payload["ensemble"]["metrics_std"]["roc_auc_std"] == pytest.approx(seed_std["roc_auc_std"])
    assert payload["seed_metrics"]["0"]["roc_auc"] == pytest.approx(seed_metrics["0"]["roc_auc"])

    csv_rows = {}
    with open(report_dir / "tox21_NR-AR.csv", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            try:
                value = float(row[2])
            except Exception:
                continue
            csv_rows[(row[0], row[1])] = value
    assert csv_rows[("ensemble", "metrics/roc_auc")] == pytest.approx(ensemble_metrics["roc_auc"])
    assert csv_rows[("ensemble", "metrics/roc_auc_std")] == pytest.approx(seed_std["roc_auc_std"])
    assert csv_rows[("seed_0", "metrics/roc_auc")] == pytest.approx(seed_metrics["0"]["roc_auc"])

    logged = {}
    for entry in wb.logs:
        for key, value in entry.items():
            logged[key] = value
    assert logged["NR-AR/metrics/roc_auc"] == pytest.approx(ensemble_metrics["roc_auc"])
    assert logged["NR-AR/metrics/roc_auc_std"] == pytest.approx(seed_std["roc_auc_std"])
    assert logged["NR-AR/seed/0/metrics/roc_auc"] == pytest.approx(seed_metrics["0"]["roc_auc"])
    assert result.get("prediction_seed_csv_path") == str(prediction_seed_csv)
    assert result.get("reliability_bins_path") == str(reliability_bins)


def test_run_single_task_rejects_unsupported_calibration_method(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tox, "run_tox21_case_study", lambda **_kwargs: None)
    monkeypatch.setattr(tox, "resolve_metric_threshold", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=2,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=1e-4,
        encoder_lr=1e-6,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        oversample_minority=False,
        use_focal_loss=False,
        dynamic_pos_weight=False,
        focal_gamma=2.0,
        layerwise_decay=None,
        threshold_metric="roc_auc",
        calibration_method=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    task_hparam_map = {"nr-ar": {"calibration_method": "unsupported"}}

    with pytest.raises(ValueError, match="Unsupported calibration_method"):
        tox._run_tox21_single_task(
            args,
            dataset_name="tox21",
            eval_mode="pretrain_frozen",
            triage_pct=0.0,
            calibrate=True,
            calibrate_per_head=False,
            task_hparam_map=task_hparam_map,
            hybrid_defaults={},
            cache_dir=None,
            report_dir=str(report_dir),
            wb=None,
            class_balance={"NR-AR": {}},
            auto_pos_weights={},
            calibration_warn_threshold=0.2,
        )

def test_run_single_task_allows_disabling_explain(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_case_study(**kwargs):
        captured["mode"] = kwargs.get("explain_mode")
        captured["config"] = kwargs.get("explain_config")
        return SimpleNamespace(
            evaluations=[SimpleNamespace(encoder_source="pretrain_frozen", name="pretrain_frozen")],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)
    monkeypatch.setenv("TOX21_EXPLAIN_MODE", "off")

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured.get("mode") == []
    assert captured.get("config") is None


def test_run_single_task_allows_disabling_explain(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_case_study(**kwargs):
        captured["mode"] = kwargs.get("explain_mode")
        captured["config"] = kwargs.get("explain_config")
        return SimpleNamespace(
            evaluations=[SimpleNamespace(encoder_source="pretrain_frozen", name="pretrain_frozen")],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)
    monkeypatch.setenv("TOX21_EXPLAIN_MODE", "off")

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured.get("mode") == []
    assert captured.get("config") is None


def test_run_single_task_removes_stale_explain_artifacts(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    task_slug = "nr-ar"
    ig_dir = report_dir / "ig_explanations" / task_slug
    ig_motif_dir = report_dir / "ig_motif_explanations" / task_slug
    for path in (ig_dir, ig_motif_dir):
        path.mkdir(parents=True, exist_ok=True)
        (path / "stale.txt").write_text("stale", encoding="utf-8")

    def fake_case_study(**kwargs):
        return SimpleNamespace(
            evaluations=[SimpleNamespace(encoder_source="pretrain_frozen", name="pretrain_frozen")],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)
    monkeypatch.setenv("TOX21_EXPLAIN_MODE", "off")

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert not ig_dir.exists()
    assert not ig_motif_dir.exists()


def test_run_single_task_normalises_motif_alias(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_case_study(**kwargs):
        captured["mode"] = kwargs.get("explain_mode")
        return SimpleNamespace(
            evaluations=[SimpleNamespace(encoder_source="pretrain_frozen", name="pretrain_frozen")],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)
    monkeypatch.setenv("TOX21_EXPLAIN_MODE", "motif_ig")

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode=None,
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured.get("mode") == ["ig_motif"]


def test_run_single_task_handles_multiple_explain_modes(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_case_study(**kwargs):
        captured["mode"] = kwargs.get("explain_mode")
        return SimpleNamespace(
            evaluations=[SimpleNamespace(encoder_source="pretrain_frozen", name="pretrain_frozen")],
            diagnostics={},
            gate_passed=True,
            threshold_payload={},
            target_payload={},
            json_path=str(report_dir / "task.json"),
            csv_path=str(report_dir / "task.csv"),
            calibrator_path=str(report_dir / "task_calib.json"),
            manifest_path=str(report_dir / "task_manifest.json"),
            auc_summary={"pretrain_frozen": 0.7},
        )

    def fake_threshold(*_args, **_kwargs):
        raise KeyError("no threshold")

    monkeypatch.setattr(tox, "run_tox21_case_study", fake_case_study)
    monkeypatch.setattr(tox, "resolve_metric_threshold", fake_threshold)

    args = SimpleNamespace(
        task="NR-AR",
        csv="samples/tox21_mini.csv",
        class_weights="auto",
        pos_class_weight=None,
        head_ensemble_size=1,
        freeze_encoder=False,
        devices=1,
        allow_shape_coercion=False,
        encoder_checkpoint=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        pretrain_lr=None,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        hidden_dim=64,
        num_layers=2,
        dropout=None,
        gnn_type="mpnn",
        add_3d=False,
        contrastive=False,
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        bf16=False,
        bf16_head=False,
        pretrain_time_budget_mins=0,
        finetune_time_budget_mins=0,
        cache_dir=None,
        strict_encoder_config=False,
        encoder_source="pretrain_frozen",
        evaluation_mode="pretrain_frozen",
        allow_equal_hash=False,
        verify_match_threshold=0.98,
        patience=5,
        tox21_head_batch_size=64,
        head_scheduler=None,
        full_finetune=False,
        unfreeze_top_layers=0,
        explain_mode="ig, motif_ig",
        explain_steps=None,
        _hidden_dim_provided=True,
        _num_layers_provided=True,
        _gnn_type_provided=True,
    )

    tox._run_tox21_single_task(
        args,
        dataset_name="tox21",
        eval_mode="pretrain_frozen",
        triage_pct=0.0,
        calibrate=True,
        cache_dir=None,
        report_dir=str(report_dir),
        wb=None,
        class_balance={"NR-AR": {}},
        auto_pos_weights={},
        calibration_warn_threshold=0.2,
    )

    assert captured.get("mode") == ["ig", "ig_motif"]
