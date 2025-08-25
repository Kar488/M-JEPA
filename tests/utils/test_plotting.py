import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure
import pandas as pd

from utils.plotting import plot_training_curves, plot_hyperparameter_results


def test_plot_training_curves_returns_figure(wb):
    curves = {
        "model_a": [1.0, 0.8, 0.6],
        "model_b": [1.0, 0.9, 0.7],
    }
    fig = plot_training_curves(curves, title="Test", wb=wb)
    assert isinstance(fig, Figure)

def test_plot_training_curves_normalize():
    curves = {"single": [2.0, 1.0]}
    fig = plot_training_curves(curves, normalize=True)
    assert isinstance(fig, Figure)


def test_plot_hyperparameter_results_empty(wb):
    df = pd.DataFrame()
    assert plot_hyperparameter_results(df, metric="roc_auc", wb=wb) is None


def test_plot_hyperparameter_results_rmse(wb):
    df = pd.DataFrame({"hidden_dim": [32, 64], "rmse": [0.5, 0.3]})
    fig = plot_hyperparameter_results(df, metric="rmse", top_n=0, wb=wb)
    assert isinstance(fig, Figure)
