import numpy as np

from training.train_on_embeddings import (
    train_linear_on_embeddings_classification,
    train_linear_on_embeddings_regression,
    train_linear_on_embeddings_with_val,
)


def test_classification_and_regression(wb):
    X_clf = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y_clf = np.array([0, 1, 0, 1])
    clf_out = train_linear_on_embeddings_classification(X_clf, y_clf)
    assert set(clf_out) == {"roc_auc", "pr_auc", "acc", "brier"}
    wb.log({"clf_acc": clf_out["acc"]})

    X_reg = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_reg = np.array([0.0, 1.0, 2.0, 3.0])
    reg_out = train_linear_on_embeddings_regression(X_reg, y_reg)
    assert "rmse" in reg_out and reg_out["rmse"] >= 0
    wb.log({"reg_rmse": reg_out["rmse"]})


def test_with_validation(wb):
    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y_train = np.array([0, 1, 0, 1])
    X_val = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=float)
    y_val = np.array([0, 1])
    X_test = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=float)
    y_test = np.array([0, 1])
    clf_metrics = train_linear_on_embeddings_with_val(
        "classification", X_train, y_train, X_val, y_val, X_test, y_test
    )
    assert all(k in clf_metrics for k in ["val_roc_auc", "test_pr_auc", "val_brier"])
    wb.log({"val_acc": clf_metrics["val_acc"]})

    X_train_r = np.arange(9, dtype=float).reshape(3, 3)
    y_train_r = X_train_r.sum(axis=1)
    X_val_r = np.array([[0.5, 0.5, 0.5]])
    y_val_r = np.array([1.5])
    reg_metrics = train_linear_on_embeddings_with_val(
        "regression", X_train_r, y_train_r, X_val_r, y_val_r
    )
    assert "val_rmse" in reg_metrics
    wb.log({"val_rmse": reg_metrics["val_rmse"]})