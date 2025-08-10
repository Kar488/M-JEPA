import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from utils.plotting import plot_training_curves


def test_plot_training_curves_returns_figure():
    curves = {
        "model_a": [1.0, 0.8, 0.6],
        "model_b": [1.0, 0.9, 0.7],
    }
    fig = plot_training_curves(curves, title="Test")
    assert isinstance(fig, Figure)
