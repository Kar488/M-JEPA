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

