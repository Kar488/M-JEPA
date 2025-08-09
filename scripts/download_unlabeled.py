"""Stream unlabelled molecules from ZINC and PubChem and store as GraphDataset shards.

This utility queries the public ZINC15 and PubChem REST APIs to download raw
SMILES strings. Each SMILES string is converted to a :class:`GraphDataset`
entry and written to sharded parquet files. The resulting directory structure
looks as follows::

    data/unlabeled/
        train/0000.parquet
        val/0000.parquet
        test/0000.parquet

A ``progress.json`` file is maintained alongside the shards so interrupted
downloads can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import pandas as pd
import requests

from data.dataset import GraphDataset

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Streaming helpers
# ----------------------------------------------------------------------------

def stream_zinc(
    batch_size: int,
    start_page: int = 1,
    sleep: float = 0.5,
) -> Iterator[Tuple[int, List[str]]]:
    """Yield batches of SMILES strings from ZINC.

    Parameters
    ----------
    batch_size: int
        Number of molecules per request.
    start_page: int, optional
        Page index to start from (1-based).
    sleep: float
        Delay between requests to avoid hammering the API.

    Yields
    ------
    page: int
        The page number that produced the batch.
    smiles: List[str]
        SMILES strings returned for that page.
    """

    page = start_page
    while True:
        url = f"https://zinc15.docking.org/substances.txt?count={batch_size}&page={page}"
        try:
            res = requests.get(url, timeout=30)
        except Exception as exc:  # network hiccup
            logger.warning("ZINC request failed: %s", exc)
            time.sleep(sleep)
            continue
        if res.status_code != 200 or not res.text.strip():
            break
        smiles = []
        for line in res.text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles.append(parts[1])
        yield page, smiles
        page += 1
        time.sleep(sleep)


def stream_pubchem(
    batch_size: int,
    start_cid: int = 1,
    sleep: float = 0.5,
) -> Iterator[Tuple[int, List[str]]]:
    """Yield batches of SMILES strings from PubChem.

    Parameters
    ----------
    batch_size: int
        Number of sequential CIDs per request.
    start_cid: int, optional
        Starting CID.
    sleep: float
        Delay between requests.

    Yields
    ------
    cid: int
        First CID used in the batch.
    smiles: List[str]
        SMILES strings returned by the query.
    """

    cid = start_cid
    while True:
        ids = ",".join(str(i) for i in range(cid, cid + batch_size))
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{ids}/property/IsomericSMILES/TXT"
        )
        try:
            res = requests.get(url, timeout=30)
        except Exception as exc:
            logger.warning("PubChem request failed: %s", exc)
            time.sleep(sleep)
            continue
        if res.status_code != 200:
            logger.warning("PubChem HTTP %s", res.status_code)
            time.sleep(sleep)
            continue
        smiles = [ln.strip() for ln in res.text.splitlines() if ln.strip()]
        if not smiles:
            break
        yield cid, smiles
        cid += batch_size
        time.sleep(sleep)


# ----------------------------------------------------------------------------
# Saving helper
# ----------------------------------------------------------------------------

def save_shards(smiles: List[str], out_dir: Path, shard_size: int) -> None:
    """Convert SMILES to graphs and write sharded parquet files."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(0, len(smiles), shard_size):
        chunk = smiles[idx : idx + shard_size]
        ds = GraphDataset.from_smiles_list(chunk)
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
        shard_path = out_dir / f"{idx // shard_size:04d}.parquet"
        df.to_parquet(shard_path, index=False)


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data") / "unlabeled",
        help="Output directory root",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=1000,
        help="Total number of molecules to fetch from all sources",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of molecules per API request",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of graphs per parquet shard",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay (in seconds) between API calls",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Dataset split ratios",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing progress.json file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    progress_path = args.out_root / "progress.json"
    start_page = 1
    start_cid = 1
    if args.resume and progress_path.exists():
        state = json.loads(progress_path.read_text())
        start_page = int(state.get("zinc_page", 1))
        start_cid = int(state.get("pubchem_cid", 1))
        logger.info(
            "Resuming from ZINC page %s and PubChem CID %s", start_page, start_cid
        )

    need = args.total
    half = need // 2
    smiles: List[str] = []

    zinc_stream = stream_zinc(args.batch_size, start_page=start_page, sleep=args.sleep)
    last_page = start_page
    while len(smiles) < half:
        try:
            page, batch = next(zinc_stream)
        except StopIteration:
            break
        smiles.extend(batch)
        last_page = page + 1
    pubchem_stream = stream_pubchem(
        args.batch_size, start_cid=start_cid, sleep=args.sleep
    )
    last_cid = start_cid
    while len(smiles) < need:
        try:
            cid, batch = next(pubchem_stream)
        except StopIteration:
            break
        smiles.extend(batch)
        last_cid = cid + args.batch_size

    random.shuffle(smiles)
    n_train = int(len(smiles) * args.split[0])
    n_val = int(len(smiles) * args.split[1])
    splits = {
        "train": smiles[:n_train],
        "val": smiles[n_train : n_train + n_val],
        "test": smiles[n_train + n_val :],
    }
    for split, sm_list in splits.items():
        save_shards(sm_list, args.out_root / split, args.shard_size)

    progress = {"zinc_page": last_page, "pubchem_cid": last_cid}
    args.out_root.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress))
    logger.info("Finished. Wrote %d molecules", len(smiles))


if __name__ == "__main__":
    main()
