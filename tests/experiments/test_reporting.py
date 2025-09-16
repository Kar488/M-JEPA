import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

try:  # prefer real matplotlib when available
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - exercised only when matplotlib missing
    plt = types.SimpleNamespace(
        figure=lambda *a, **k: None,
        bar=lambda *a, **k: None,
        xticks=lambda *a, **k: None,
        title=lambda *a, **k: None,
        tight_layout=lambda *a, **k: None,
        savefig=lambda path, *a, **k: Path(path).touch(),
        close=lambda *a, **k: None,
        imshow=lambda *a, **k: None,
        yticks=lambda *a, **k: None,
        colorbar=lambda *a, **k: None,
    )
    sys.modules.setdefault("matplotlib", types.SimpleNamespace(pyplot=plt))
    sys.modules.setdefault("matplotlib.pyplot", plt)

from experiments import reporting  # noqa: E402


def test_summarize_with_ci_adds_ci_column():
    df = pd.DataFrame(
        {
            "metric": [1.0],
            "metric_std": [0.3],
            "seeds": [[0, 1, 2]],
        }
    )
    out = reporting.summarize_with_ci(df, ["metric"], seeds_col="seeds")
    expected_ci = 1.96 * 0.3 / np.sqrt(3)
    assert np.isclose(out.loc[0, "metric_ci95"], expected_ci)


def test_build_full_report_creates_outputs(tmp_path):
    df = pd.DataFrame(
        {
            "method": ["jepa"],
            "hidden_dim": [64],
            "num_layers": [2],
            "mask_ratio": [0.5],
            "gnn_type": ["gcn"],
            "ema_decay": [0.99],
            "score": [0.8],
        }
    )
    reporting.build_full_report(df, metric="score", out_dir=str(tmp_path), top_n=1)
    files = {
        "ranked_top1.csv",
        "top1_bar.png",
        "heatmap_jepa_mask_gnn.png",
        "heatmap_jepa_ema_hidden.png",
    }
    produced = {p.name for p in Path(tmp_path).iterdir()}
    assert files <= produced
