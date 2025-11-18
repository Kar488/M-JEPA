from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.dataset import load_directory_dataset


def _maybe_inject_repo_root() -> None:
    """Ensure the repository root is available on ``sys.path``."""

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _load_graph_module():
    from data import mdataset as _mdataset  # local import for PYTHONPATH hygiene

    return _mdataset


_maybe_inject_repo_root()

_mdataset = _load_graph_module()

from scripts.commands import dataset_cache

_PART_PATTERN = re.compile(r"part-(\d+)-(\d+)\.pkl$")
_DEFAULT_SHARD_SIZE = 5000
_SMILES_COLUMN = "smiles"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unlabeled-dir", required=True, help="Path to the unlabeled corpus")
    parser.add_argument("--labeled-dir", required=True, help="Path to the labeled corpus")
    parser.add_argument("--cache-dir", required=True, help="Base cache directory (same as sweeps)")
    parser.add_argument("--label-col", default=None, help="Optional label column for the labeled set")
    parser.add_argument("--sample-unlabeled", type=int, default=0, help="Optional max graphs for unlabeled set")
    parser.add_argument("--sample-labeled", type=int, default=0, help="Optional max graphs for labeled set")
    parser.add_argument("--num-workers", type=int, default=-1, help="RDKit worker pool size")
    parser.add_argument("--add-3d", action="store_true", help="Enable 3D featurisation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even when files already exist",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    return parser


def _normalized_payload(dirpath: str, add_3d: bool, sample: int, label_col: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": dirpath,
        "add_3d": bool(add_3d),
        "sample": int(sample or 0),
    }
    if label_col:
        payload["label_col"] = label_col
    return payload


def _ensure_dir(path: str, name: str) -> str:
    resolved = dataset_cache.resolve_env_path(path)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{name} directory not found: {resolved}")
    return resolved


def _log(msg: str) -> None:
    print(f"[cache-warm] {msg}", flush=True)


def _list_dataset_files(dirpath: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(dirpath)):
        path = os.path.join(dirpath, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in {".parquet", ".csv"}:
            files.append((path, ext))
    return files


def _read_dataframe(path: str, ext: str, label_col: Optional[str]) -> pd.DataFrame:
    cols = [_SMILES_COLUMN]
    if label_col:
        cols.append(label_col)
    if ext == ".parquet":
        return pd.read_parquet(path, columns=cols)
    if ext == ".csv":
        return pd.read_csv(path, usecols=lambda c: c in cols)
    raise ValueError(f"Unsupported dataset extension: {ext}")


def _discover_existing_shards(shard_dir: str, base_dir: str) -> Tuple[List[Dict[str, Any]], int]:
    if not os.path.isdir(shard_dir):
        return [], 0
    entries: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(shard_dir)):
        match = _PART_PATTERN.match(fname)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2))
        rel_path = os.path.relpath(os.path.join(shard_dir, fname), base_dir)
        entries.append({"path": rel_path, "start": start, "count": count})
    entries.sort(key=lambda entry: entry["start"])
    processed = 0
    contiguous: List[Dict[str, Any]] = []
    for entry in entries:
        if entry["start"] != processed:
            shutil.rmtree(shard_dir, ignore_errors=True)
            return [], 0
        processed += entry["count"]
        contiguous.append(entry)
    return contiguous, processed


class _ShardAccumulator:
    def __init__(
        self,
        *,
        shard_dir: str,
        base_dir: str,
        shard_size: int,
        existing: List[Dict[str, Any]],
        expect_labels: bool,
    ) -> None:
        self.shard_dir = shard_dir
        self.base_dir = base_dir
        self.shard_size = shard_size
        self.records: List[Dict[str, Any]] = list(existing)
        self.total_emitted = sum(entry["count"] for entry in existing)
        self.expect_labels = expect_labels
        self.labels_present = bool(existing) and expect_labels
        self._batch_graphs: List[Any] = []
        self._batch_smiles: List[str] = []
        self._batch_labels: Optional[List[Any]] = [] if expect_labels else None
        self._batch_start = self.total_emitted

    def add(self, graph_state: Dict[str, Any], smiles: str, label: Any) -> None:
        if not self._batch_graphs:
            self._batch_start = self.total_emitted
            if self.expect_labels and self._batch_labels is None:
                self._batch_labels = []
        self._batch_graphs.append(graph_state)
        self._batch_smiles.append(smiles)
        if self.expect_labels and self._batch_labels is not None:
            self._batch_labels.append(label)
            self.labels_present = True
        if len(self._batch_graphs) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._batch_graphs:
            return
        shard_name = f"part-{self._batch_start:09d}-{len(self._batch_graphs):05d}.pkl"
        shard_path = os.path.join(self.shard_dir, shard_name)
        payload = {
            "graphs": list(self._batch_graphs),
            "smiles": list(self._batch_smiles),
            "count": len(self._batch_graphs),
        }
        if self.expect_labels and self._batch_labels is not None:
            payload["labels"] = list(self._batch_labels)
        with open(shard_path, "wb") as fh:
            pickle.dump(payload, fh)
        rel_path = os.path.relpath(shard_path, self.base_dir)
        self.records.append(
            {"path": rel_path, "start": self._batch_start, "count": len(self._batch_graphs)}
        )
        self.total_emitted += len(self._batch_graphs)
        self._batch_graphs.clear()
        self._batch_smiles.clear()
        if self._batch_labels is not None:
            self._batch_labels.clear()

    def finalize(self) -> None:
        self.flush()


def _stream_directory_to_cache(
    *,
    dirpath: str,
    cache_path: str,
    label_col: Optional[str],
    add_3d: bool,
    sample: int,
    num_workers: int,
    force: bool,
    log: Callable[[str], None],
) -> None:
    base_dir = os.path.dirname(cache_path)
    shard_dir = f"{cache_path}.parts"
    if force:
        shutil.rmtree(shard_dir, ignore_errors=True)
        try:
            os.remove(cache_path)
        except OSError:
            pass
    os.makedirs(shard_dir, exist_ok=True)

    existing_shards, processed = _discover_existing_shards(shard_dir, base_dir)
    if processed:
        log(f"resuming from {processed} processed graphs ({len(existing_shards)} shards)")

    max_graphs = sample if sample > 0 else None
    expect_labels = bool(label_col)
    accumulator = _ShardAccumulator(
        shard_dir=shard_dir,
        base_dir=base_dir,
        shard_size=_DEFAULT_SHARD_SIZE,
        existing=existing_shards,
        expect_labels=expect_labels,
    )

    if max_graphs is not None and processed >= max_graphs:
        accumulator.finalize()
        _write_manifest(cache_path, accumulator, log)
        return

    worker_budget = _mdataset._resolve_worker_count(num_workers)
    cls_module = _mdataset.GraphDataset.__module__
    cls_qualname = _mdataset.GraphDataset.__qualname__

    resume_offset = processed
    graphs_emitted = processed
    files = _list_dataset_files(dirpath)
    if not files:
        raise FileNotFoundError(f"No supported dataset files found in {dirpath}")
    for path, ext in files:
        if max_graphs is not None and graphs_emitted >= max_graphs:
            break
        df = _read_dataframe(path, ext, label_col)
        smiles = df[_SMILES_COLUMN].astype(str).tolist()
        labels_seq: Optional[Sequence[Any]] = None
        if label_col and label_col in df.columns:
            labels_seq = df[label_col].tolist()
        if resume_offset:
            if resume_offset >= len(smiles):
                resume_offset -= len(smiles)
                continue
            smiles = smiles[resume_offset:]
            if labels_seq is not None:
                labels_seq = labels_seq[resume_offset:]
            resume_offset = 0
        if max_graphs is not None:
            remaining = max_graphs - graphs_emitted
            smiles = smiles[:remaining]
            if labels_seq is not None:
                labels_seq = labels_seq[:remaining]
        if not smiles:
            continue
        state_iter = _mdataset._iter_graph_states(
            smiles,
            add_3d=add_3d,
            random_seed=None,
            worker_budget=worker_budget,
            cls_module=cls_module,
            cls_qualname=cls_qualname,
        )
        for idx, g_state in state_iter:
            if g_state is None:
                continue
            label_value = None
            if labels_seq is not None and idx < len(labels_seq):
                label_value = labels_seq[idx]
            accumulator.add(g_state, smiles[idx], label_value)
            graphs_emitted += 1
            if max_graphs is not None and graphs_emitted >= max_graphs:
                break
    accumulator.finalize()
    _write_manifest(cache_path, accumulator, log)


def _write_manifest(cache_path: str, accumulator: _ShardAccumulator, log: Callable[[str], None]) -> None:
    manifest = {
        "__dataset_cache_format__": dataset_cache.SHARDED_CACHE_FORMAT,
        "version": dataset_cache.SHARDED_CACHE_VERSION,
        "shards": accumulator.records,
        "total_graphs": accumulator.total_emitted,
        "has_labels": bool(accumulator.labels_present),
    }
    with open(cache_path, "wb") as fh:
        pickle.dump(manifest, fh)
    log(
        f"persisted manifest for {accumulator.total_emitted} graphs across {len(accumulator.records)} shards"
    )


def _make_streaming_builder(
    *,
    kind: str,
    payload: Dict[str, Any],
    dirpath: str,
    cache_root: str,
    label_col: Optional[str],
    add_3d: bool,
    sample: int,
    num_workers: int,
    force: bool,
    log: Callable[[str], None],
) -> Callable[[], dataset_cache.DatasetBuilderResult]:
    def _builder() -> dataset_cache.DatasetBuilderResult | "_mdataset.GraphDataset":  # type: ignore[name-defined]
        cache_path = dataset_cache.dataset_cache_path(kind, payload, cache_root)
        if cache_path is None:
            raise RuntimeError(f"unable to resolve cache path for {kind}")
        try:
            _stream_directory_to_cache(
                dirpath=dirpath,
                cache_path=cache_path,
                label_col=label_col,
                add_3d=add_3d,
                sample=sample,
                num_workers=num_workers,
                force=force,
                log=log,
            )
        except FileNotFoundError:
            log(
                "directory lacks parquet/csv files; falling back to load_directory_dataset"
            )
            max_graphs = sample if sample > 0 else None
            return load_directory_dataset(
                dirpath,
                label_col=label_col,
                add_3d=add_3d,
                max_graphs=max_graphs,
                num_workers=num_workers,
            )
        return dataset_cache.DatasetBuilderResult(data=None, already_persisted=True)

    return _builder


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    unlabeled_dir = _ensure_dir(args.unlabeled_dir, "unlabeled")
    labeled_dir = _ensure_dir(args.labeled_dir, "labeled")
    cache_dir = dataset_cache.resolve_env_path(args.cache_dir)

    cache_root = dataset_cache.prepare_cache_root(cache_dir, enabled=True)
    if not cache_root:
        raise RuntimeError("cache directory could not be prepared")

    _log(
        f"warming dataset caches at {cache_root} (add_3d={int(bool(args.add_3d))}, force={int(bool(args.force))})"
    )

    unlabeled_payload = _normalized_payload(unlabeled_dir, args.add_3d, args.sample_unlabeled, None)
    unlabeled_builder = _make_streaming_builder(
        kind="unlabeled",
        payload=unlabeled_payload,
        dirpath=unlabeled_dir,
        cache_root=cache_root,
        label_col=None,
        add_3d=bool(args.add_3d),
        sample=args.sample_unlabeled,
        num_workers=args.num_workers,
        force=args.force,
        log=_log,
    )
    dataset_cache.ensure_dataset_cache(
        "unlabeled",
        unlabeled_payload,
        unlabeled_builder,
        cache_root,
        force=args.force,
        log=_log,
    )

    labeled_payload = _normalized_payload(
        labeled_dir, args.add_3d, args.sample_labeled, args.label_col
    )
    labeled_builder = _make_streaming_builder(
        kind="labeled",
        payload=labeled_payload,
        dirpath=labeled_dir,
        cache_root=cache_root,
        label_col=args.label_col,
        add_3d=bool(args.add_3d),
        sample=args.sample_labeled,
        num_workers=args.num_workers,
        force=args.force,
        log=_log,
    )
    dataset_cache.ensure_dataset_cache(
        "labeled",
        labeled_payload,
        labeled_builder,
        cache_root,
        force=args.force,
        log=_log,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
