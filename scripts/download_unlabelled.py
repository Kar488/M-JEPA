"""Download unlabelled molecules and convert them to graph format.

This small utility streams SMILES strings from the public ZINC15 and
PubChem REST APIs.  The SMILES are converted into the light-weight
:class:`~data.mdataset.GraphDataset` format used throughout the project
and written to a parquet file for later consumption.

By default a file named ``unlabelled.parquet`` is placed in
``data/raw/`` but both the dataset size and output directory can be
controlled via the command line.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from data.mdataset import GraphDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for fetching SMILES
# ---------------------------------------------------------------------------

def _stream_zinc(batch_size: int) -> Iterator[str]:
    """Yield SMILES strings from ZINC15."""

    page = 1
    while True:
        url = f"https://zinc15.docking.org/substances.txt?count={batch_size}&page={page}"
        try:
            res = requests.get(url, timeout=30)
        except Exception as exc:  # pragma: no cover - network hiccups
            logger.warning("ZINC request failed: %s", exc)
            continue
        if res.status_code != 200 or not res.text.strip():
            break
        for line in res.text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                yield parts[1]
        page += 1


def _stream_pubchem(batch_size: int) -> Iterator[str]:
    """Yield SMILES strings from PubChem."""

    cid = 1
    while True:
        ids = ",".join(str(i) for i in range(cid, cid + batch_size))
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{ids}/property/IsomericSMILES/TXT"
        )
        try:
            res = requests.get(url, timeout=30)
        except Exception as exc:  # pragma: no cover
            logger.warning("PubChem request failed: %s", exc)
            continue
        if res.status_code != 200 or not res.text.strip():
            break
        for line in res.text.splitlines():
            sm = line.strip()
            if sm:
                yield sm
        cid += batch_size


def _collect_smiles(total: int, batch_size: int = 100) -> list[str]:
    """Collect roughly ``total`` SMILES from the public APIs."""

    smiles: list[str] = []
    # Try ZINC first
    for sm in _stream_zinc(batch_size):
        smiles.append(sm)
        if len(smiles) >= total:
            return smiles[:total]
    # Fall back to PubChem for the remainder
    for sm in _stream_pubchem(batch_size):
        smiles.append(sm)
        if len(smiles) >= total:
            return smiles[:total]
    return smiles


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Total number of molecules to download",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "raw",
        help="Directory in which the parquet file will be written",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    smiles = _collect_smiles(args.size)
    if not smiles:
        logger.error("Failed to download any molecules")
        return

    ds = GraphDataset.from_smiles_list(smiles)
    rows = []
    for sm, g in zip(ds.smiles, ds.graphs):
        rows.append(
            {
                "smiles": sm,
                "x": g.x.tolist(),
                "edge_index": g.edge_index.tolist(),
                "edge_attr": None if g.edge_attr is None else g.edge_attr.tolist(),
            }
        )
    df = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "unlabelled.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Wrote %d molecules to %s", len(df), out_file)


if __name__ == "__main__":
    main()
