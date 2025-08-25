import os
import sys
import types
from unittest.mock import patch

import pandas as pd
import pytest

try:  # pragma: no cover - torch may be unavailable
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = types.SimpleNamespace()
    sys.modules.setdefault("torch", torch)

pytest.importorskip("fastparquet")

from data import GraphDataset


def test_from_directory_respects_max_graphs(tmp_path):
    pd.options.io.parquet.engine = "fastparquet"
    for i in range(4):
        df = pd.DataFrame({"smiles": [f"C{i}", f"O{i}"]})
        df.to_parquet(tmp_path / f"part{i}.parquet")

    with patch.object(GraphDataset, "from_parquet", wraps=GraphDataset.from_parquet) as mock_fp:
        ds = GraphDataset.from_directory(
            str(tmp_path), max_graphs=5, n_rows_per_file=2
        )

    assert len(ds) <= 5
    touched = [os.path.basename(call.kwargs["filepath"]) for call in mock_fp.call_args_list]
    assert touched == [f"part{i}.parquet" for i in range(3)]
