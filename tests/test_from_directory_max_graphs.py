import os
import sys
import types
from unittest.mock import patch

import pandas as pd

try:  # pragma: no cover - torch may be unavailable
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = types.SimpleNamespace()
    sys.modules.setdefault("torch", torch)

from pathlib import Path
import importlib.util
def _load_real_graphdataset():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    mod_name = "data.mdataset"              # the real module name
    file_path = data_dir / "mdataset.py"

    # 1) Ensure 'data' package exists and points at your repo's data/ dir
    if "data" not in sys.modules:
        pkg = types.ModuleType("data")
        pkg.__path__ = [str(data_dir)]
        sys.modules["data"] = pkg
    else:
        # make sure its __path__ points to your repo
        sys.modules["data"].__path__ = [str(data_dir)]

    # 2) Build spec for the correct qualified name, create module, and
    #    register it in sys.modules BEFORE exec_module (needed for dataclasses)
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module.GraphDataset

GraphDataset = _load_real_graphdataset()


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
