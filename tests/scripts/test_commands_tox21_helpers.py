from types import SimpleNamespace

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
    assert pred_path.read_text().splitlines()[0] == "graph_id,assay,true_label,logit,probability"
    assert result.get("stage_payload", {}).get("prediction_csv") == str(pred_path)


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
