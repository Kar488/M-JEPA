from __future__ import annotations

import argparse
import math
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import log_effective_gnn

try:  # pragma: no cover - optional relative import depending on entry point
    from ..bench import BenchmarkRule, resolve_metric_threshold
except ImportError:  # pragma: no cover - fallback when executed as a script
    from scripts.bench import BenchmarkRule, resolve_metric_threshold


HIGHER_IS_BETTER = {"roc_auc", "pr_auc", "acc", "accuracy"}
def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare JEPA and contrastive encoders on the same labelled dataset  with flexible loading + report.

    Runs training across seeds and reports which method yields better
    performance based on ROC‑AUC (classification) or RMSE (regression).
    """

    logger.info("Starting benchmark with args: %s", args)
    if (
        load_directory_dataset is None
        or build_encoder is None
        or train_linear_head is None
    ):
        logger.warning("Benchmark modules are unavailable.")
        sys.exit(6)

    dataset_name = getattr(args, "dataset", None)
    task_name = getattr(args, "task", None)
    threshold_rule: Optional[BenchmarkRule] = None
    if dataset_name:
        try:
            threshold_rule = resolve_metric_threshold(dataset_name, task_name)
        except KeyError:
            threshold_rule = None

    threshold_payload: Dict[str, Any] = {}
    if threshold_rule is not None:
        threshold_payload = {
            "benchmark_metric": threshold_rule.metric,
            "benchmark_threshold": threshold_rule.threshold,
        }

    seeds: List[int]
    arg_seeds = getattr(args, "seeds", None)
    if arg_seeds is not None and len(arg_seeds) > 0:
        seeds = arg_seeds
    else:
        seeds = CONFIG.get("benchmark", {}).get("seeds", [0])  # type: ignore[assignment]

    eval_finetuned = bool(getattr(args, "eval_finetuned", False))
    config_payload: Dict[str, Any] = {
        "labeled_dir": args.labeled_dir,
        "test_dir": getattr(args, "test_dir", None),
        "task_type": args.task_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seeds": seeds,
        "eval_finetuned": eval_finetuned,
        "gnn_type": getattr(args, "gnn_type", None),
        "hidden_dim": getattr(args, "hidden_dim", None),
        "num_layers": getattr(args, "num_layers", None),
        "add_3d": bool(getattr(args, "add_3d", False)),
        "num_workers": getattr(args, "num_workers", None),
        "pin_memory": getattr(args, "pin_memory", None),
        "persistent_workers": getattr(args, "persistent_workers", None),
        "prefetch_factor": getattr(args, "prefetch_factor", None),
    }
    if dataset_name:
        config_payload["dataset"] = dataset_name
    if task_name:
        config_payload["task"] = task_name
    config_payload.update(threshold_payload)

    run_id = f"benchmark-{uuid.uuid4().hex[:8]}"
    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config=config_payload,
        id=run_id,
        resume="never",
    )
    log_effective_gnn(args, logger, wb)

    import json
    import os
    import time

    import numpy as np
    import torch

    from utils.checkpoint import load_checkpoint  # for fine-tuned ckpt (encoder+head)

    try:
        from ..utils.checkpoint  import safe_load_checkpoint as _safe_load_checkpoint        # type: ignore[import-not-found]
        from ..utils.checkpoint  import load_state_dict_forgiving as _load_state_dict_forgiving      # type: ignore[import-not-found]
        from ..utils.checkpoint  import resolve_ckpt_path   # type: ignore[import-not-found]
    except ImportError:
        # Fallback: absolute imports when run from repo root with PYTHONPATH set
        from utils.checkpoint import safe_load_checkpoint  as _safe_load_checkpoint        # type: ignore[import-not-found]
        from utils.checkpoint import load_state_dict_forgiving as _load_state_dict_forgiving        # type: ignore[import-not-found]
        from utils.checkpoint import resolve_ckpt_path  # type: ignore[import-not-found]

    from pathlib import Path

    # --- paths / report ---
    args.report_dir = getattr(args, "report_dir", "reports")
    os.makedirs(args.report_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_stem = getattr(args, "report_stem", f"benchmark_{timestamp}")
    report_json = os.path.join(args.report_dir, report_stem + ".json")
    report_csv = os.path.join(args.report_dir, report_stem + ".csv")

    # safe W&B helpers
    # safe W&B helpers: prefer wb.log / wb.finish if present; else try wb.run.*
    def _wb_log(payload):
        if wb is None:
            return
        try:
            if hasattr(wb, "log"):
                wb.log(payload)
            elif hasattr(wb, "run") and hasattr(wb.run, "log"):
                wb.run.log(payload)
        except Exception:
            pass
    def _wb_finish():
        if wb is None:
            return
        try:
            if hasattr(wb, "finish"):
                wb.finish()
            elif hasattr(wb, "run") and hasattr(wb.run, "finish"):
                wb.run.finish()
        except Exception:
            pass

    split_aliases = {
        "train": ("train",),
        "val": ("val", "valid", "validation"),
        "test": ("test",),
    }

    def _canonical_split_name(name: str) -> Optional[str]:
        lower = name.lower()
        for canon, aliases in split_aliases.items():
            if lower == canon or lower in aliases:
                return canon
            for sep in ("-", "_"):
                if lower.endswith(f"{sep}{canon}") or any(
                    lower.endswith(f"{sep}{alias}") for alias in aliases
                ):
                    return canon
        return None

    def _discover_sibling_splits(dirpath: Path) -> Tuple[Optional[str], Dict[str, Path]]:
        split_hint = _canonical_split_name(dirpath.name)
        found: Dict[str, Path] = {}
        if split_hint is None:
            return None, found
        parent = dirpath.parent
        try:
            siblings = list(parent.iterdir())
        except Exception:
            siblings = []
        for canon, aliases in split_aliases.items():
            for name in aliases + (canon,):
                candidate = parent / name
                if candidate.is_dir():
                    found.setdefault(canon, candidate)
                    break
        for sib in siblings:
            if not sib.is_dir():
                continue
            canon = _canonical_split_name(sib.name)
            if canon is not None and canon not in found:
                found[canon] = sib
        if dirpath.is_dir():
            found.setdefault(split_hint, dirpath)
        return split_hint, found

    def _merge_split_datasets(split_map: Dict[str, Any]) -> Tuple[Any, Dict[str, List[int]]]:
        graphs_all: List[Any] = []
        labels_all: List[Any] = []
        smiles_all: List[str] = []
        labels_present = False
        indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        offset = 0
        ordered: Sequence[Tuple[str, Any]] = [
            (name, split_map[name])
            for name in ("train", "val", "test")
            if name in split_map and split_map[name] is not None
        ]
        for split_name, dataset in ordered:
            size = len(dataset) if hasattr(dataset, "__len__") else 0
            indices[split_name] = list(range(offset, offset + size))
            offset += size
            graphs_all.extend(getattr(dataset, "graphs", []))
            smiles_all.extend(getattr(dataset, "smiles", []) or [])
            labels_attr = getattr(dataset, "labels", None)
            if labels_attr is not None:
                labels_present = True
                labels_all.extend(np.asarray(labels_attr).tolist())
        labels_arr = np.asarray(labels_all) if labels_present else None

        dataset_cls = ordered[0][1].__class__ if ordered else None

        def _fallback_dataset(graphs, labels, smiles):
            class _CombinedDataset:
                def __init__(self, g, l, s):
                    self.graphs = g
                    self.labels = l
                    self.smiles = s

                def __len__(self):
                    return len(self.graphs)

            return _CombinedDataset(graphs, labels, smiles)

        try:
            combined = dataset_cls(graphs_all, labels_arr, smiles_all or None)  # type: ignore[operator]
        except Exception:
            combined = _fallback_dataset(graphs_all, labels_arr, smiles_all or None)
        return combined, indices
    
    def _split_indices_for_training(dataset: Any) -> Tuple[List[int], List[int], List[int]]:
        total = len(dataset) if hasattr(dataset, "__len__") else 0
        indices = list(range(total))
        if getattr(args, "task_type", None) == "classification":
            labels_attr = getattr(dataset, "labels", None)
            if labels_attr is not None:
                labels_arr = np.asarray(labels_attr)
                train_idx: List[int] = []
                val_idx: List[int] = []
                test_idx: List[int] = []
                for label in np.unique(labels_arr):
                    label_idx = np.where(labels_arr == label)[0].tolist()
                    random.shuffle(label_idx)
                    n_train = int(math.floor(0.8 * len(label_idx)))
                    n_val = int(math.floor(0.1 * len(label_idx)))
                    train_idx.extend(label_idx[:n_train])
                    val_idx.extend(label_idx[n_train : n_train + n_val])
                    test_idx.extend(label_idx[n_train + n_val :])
                return train_idx, val_idx, test_idx
        random.shuffle(indices)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    def _estimate_split_sizes(
        dataset: Any,
        train_indices: Optional[Sequence[int]],
        val_indices: Optional[Sequence[int]],
        test_indices: Optional[Sequence[int]],
    ) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        total = None
        try:
            total = len(dataset)  # type: ignore[arg-type]
        except Exception:
            total = None
        if total is None:
            return stats
        stats["n_total"] = int(total)
        if any(idx is not None for idx in (train_indices, val_indices, test_indices)):
            stats["n_train"] = len(train_indices or [])
            stats["n_val"] = len(val_indices or [])
            stats["n_test"] = len(test_indices or [])
            return stats
        if getattr(args, "task_type", None) == "classification":
            labels_attr = getattr(dataset, "labels", None)
            if labels_attr is not None:
                labels_arr = np.asarray(labels_attr)
                _, counts = np.unique(labels_arr, return_counts=True)
                stats["n_train"] = int(sum(math.floor(0.8 * c) for c in counts))
                stats["n_val"] = int(sum(math.floor(0.1 * c) for c in counts))
                stats["n_test"] = max(0, int(total) - stats["n_train"] - stats["n_val"])
                return stats
        stats["n_train"] = int(math.floor(0.8 * total))
        stats["n_val"] = int(math.floor(0.1 * total))
        stats["n_test"] = max(0, int(total) - stats["n_train"] - stats["n_val"])
        return stats

    def _scaled_batch_size(requested: Any, dataset_size: int) -> int:
        try:
            req = int(requested)
        except Exception:
            req = 1
        if req <= 0:
            req = 1
        target = max(8, dataset_size // 8)
        scaled = min(req, max(1, target))
        if scaled != req:
            expected_batches = max(1, math.ceil(dataset_size / max(1, scaled)))
            logger.warning(
                "Detected split-only dataset at %s (N=%d); adjusting batch_size from %d to %d to target ~%d batches.",
                args.labeled_dir,
                dataset_size,
                req,
                scaled,
                expected_batches,
            )
        return scaled

    loader_kwargs = {
        "label_col": args.label_col,
        "add_3d": args.add_3d,
        "num_workers": getattr(args, "num_workers", -1),
        "cache_dir": getattr(args, "cache_dir", None),
    }

    train_indices: Optional[List[int]] = None
    val_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None
    dataset_stats: Dict[str, int] = {}
    effective_batch_size = args.batch_size
    dataset_strategy = "single_dir"

    split_hint, sibling_dirs = _discover_sibling_splits(Path(args.labeled_dir))

    try:
        if getattr(args, "test_dir", None) and eval_finetuned:
            labeled = load_directory_dataset(
                args.test_dir,
                **loader_kwargs,  # type: ignore[arg-type]
            )
            dataset_strategy = "eval_only_test_dir"
            dataset_stats = {
                "n_total": len(labeled) if hasattr(labeled, "__len__") else 0,
                "n_train": 0,
                "n_val": 0,
                "n_test": len(labeled) if hasattr(labeled, "__len__") else 0,
            }
            _wb_log({"phase": "data_load", "test_graphs": dataset_stats["n_test"]})
        elif "train" in sibling_dirs:
            split_datasets: Dict[str, Any] = {}
            split_counts: Dict[str, int] = {}
            for name in ("train", "val"):
                path = sibling_dirs.get(name)
                if path is None:
                    continue
                ds = load_directory_dataset(str(path), **loader_kwargs)  # type: ignore[arg-type]
                split_datasets[name] = ds
                split_counts[name] = len(ds) if hasattr(ds, "__len__") else 0
            if getattr(args, "test_dir", None):
                test_ds = load_directory_dataset(args.test_dir, **loader_kwargs)  # type: ignore[arg-type]
                split_datasets["test"] = test_ds
                split_counts["test"] = len(test_ds) if hasattr(test_ds, "__len__") else 0
                dataset_strategy = "explicit_splits_with_test_dir"
            else:
                path = sibling_dirs.get("test")
                if path is not None:
                    ds = load_directory_dataset(str(path), **loader_kwargs)  # type: ignore[arg-type]
                    split_datasets["test"] = ds
                    split_counts["test"] = len(ds) if hasattr(ds, "__len__") else 0
                dataset_strategy = "explicit_splits"
            labeled, indices = _merge_split_datasets(split_datasets)
            train_indices = indices.get("train")
            val_indices = indices.get("val")
            test_indices = indices.get("test")
            logger.info(
                "Detected split directory input under %s; using train=%s (N=%d) val=%s (N=%d) test=%s (N=%d)",
                args.labeled_dir,
                sibling_dirs.get("train"),
                split_counts.get("train", 0),
                sibling_dirs.get("val"),
                split_counts.get("val", 0),
                sibling_dirs.get("test"),
                split_counts.get("test", 0),
            )
            _wb_log(
                {
                    "phase": "data_load",
                    "train_graphs": split_counts.get("train", 0),
                    "val_graphs": split_counts.get("val", 0),
                    "test_graphs": split_counts.get("test", 0),
                    "dataset/strategy": dataset_strategy,
                }
            )
        else:
            if getattr(args, "test_dir", None):
                train_ds = load_directory_dataset(
                    args.labeled_dir,
                    **loader_kwargs,  # type: ignore[arg-type]
                )
                test_ds = load_directory_dataset(
                    args.test_dir,
                    **loader_kwargs,  # type: ignore[arg-type]
                )
                labeled, indices = _merge_split_datasets({"train": train_ds, "test": test_ds})
                train_indices, val_indices, _ = _split_indices_for_training(train_ds)
                test_indices = indices.get("test")
                dataset_strategy = "train_val_plus_test_dir"
                _wb_log(
                    {
                        "phase": "data_load",
                        "train_graphs": len(train_ds) if hasattr(train_ds, "__len__") else 0,
                        "test_graphs": len(test_ds) if hasattr(test_ds, "__len__") else 0,
                        "dataset/strategy": dataset_strategy,
                    }
                )
            else:
                data_dir = getattr(args, "test_dir", None) or args.labeled_dir
                labeled = load_directory_dataset(
                    data_dir,
                    **loader_kwargs,  # type: ignore[arg-type]
                )
                dataset_strategy = "single_split_dir" if split_hint is not None else "single_dir"
                if split_hint is not None:
                    logger.info(
                        "Input directory %s appears to be a '%s' split without sibling train data; reusing it for training/validation.",
                        data_dir,
                        split_hint,
                    )
                    dataset_len = len(labeled) if hasattr(labeled, "__len__") else 0
                    effective_batch_size = _scaled_batch_size(args.batch_size, dataset_len)
                    if effective_batch_size != args.batch_size:
                        _wb_log(
                            {
                                "dataset/batch_size_requested": args.batch_size,
                                "dataset/batch_size_effective": effective_batch_size,
                                "dataset/strategy": "single_split_dir",
                                "dataset/size": dataset_len,
                            }
                        )
                _wb_log({"phase": "data_load", "labeled_graphs": len(labeled)})
    except Exception:
        logger.exception("Failed to load labelled dataset for benchmarking")
        _wb_log({"phase": "data_load", "status": "error"})
        sys.exit(1)

    dataset_stats = _estimate_split_sizes(
        labeled,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    if dataset_stats:
        payload = {f"dataset/{k}": v for k, v in dataset_stats.items()}
        payload["dataset/strategy"] = dataset_strategy
        _wb_log(payload)

    input_dim = labeled.graphs[0].x.shape[1]
    edge_dim = (
        None
        if labeled.graphs[0].edge_attr is None
        else labeled.graphs[0].edge_attr.shape[1]
    )
    device = resolve_device(args.device)

    if getattr(args, "ft_ckpt", None) and not eval_finetuned:
        logger.warning(
            "Fine-tuned checkpoint provided (%s) but benchmark compares encoders only; ignoring (use --eval-finetuned to enable).",
            args.ft_ckpt,
        )

    # Prepare results dict
    all_results: Dict[str, Dict[str, float]] = {}
    from typing import Any, Dict

    # If eval-finetuned is requested, run in eval-only mode using the
    # fine-tuned checkpoint and return early.
    if eval_finetuned:
        start_payload = {"phase": "benchmark", "status": "start"}
        start_payload.update(threshold_payload)
        _wb_log(start_payload)
        if not getattr(args, "ft_ckpt", None):
            _wb_log({"phase": "benchmark", "status": "error", "error": "missing_ft_ckpt"})
            _wb_finish()
            logger.error("Fine-tuned checkpoint required for --eval-finetuned mode.")
            raise SystemExit(1)
        try:
            agg_ft = evaluate_finetuned_head(args.ft_ckpt, labeled, args, device)
        except FileNotFoundError:
            _wb_log({"phase": "benchmark", "status": "error", "error": "missing_ft_ckpt"})
            _wb_finish()
            logger.exception("Failed to resolve fine-tuned checkpoint")
            raise SystemExit(1)
        if agg_ft:
            all_results["finetuned"] = agg_ft
            for k, v in agg_ft.items():
                _wb_log({f"finetuned/{k}": v})
        verdict = "finetuned"
        success_payload = {"phase": "benchmark", "status": "success", "best_method": verdict}
        success_payload.update(threshold_payload)
        _wb_log(success_payload)
        logger.info(f"Benchmark completed. Best method: {verdict}")

        try:
            payload = {"results": all_results, "best_method": verdict}
            if dataset_stats:
                payload["dataset_stats"] = dataset_stats
                payload["dataset_strategy"] = dataset_strategy
            if threshold_rule is not None:
                payload["threshold"] = {
                    "dataset": dataset_name,
                    "task": task_name,
                    **threshold_payload,
                }
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            import csv

            with open(report_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "metric", "value"])
                for k, v in agg_ft.items():
                    w.writerow(["finetuned", k, v])
                if threshold_rule is not None:
                    w.writerow(["threshold/metric", threshold_rule.metric])
                    w.writerow(["threshold/value", float(threshold_rule.threshold)])
                if dataset_stats:
                    for key in ("n_total", "n_train", "n_val", "n_test"):
                        if key in dataset_stats:
                            w.writerow(["dataset_stats", key, dataset_stats[key]])
                    w.writerow(["dataset_stats", "strategy", dataset_strategy])
            logger.info("Wrote reports: %s , %s", report_json, report_csv)
        except Exception:
            logger.warning("Failed to write reports", exc_info=True)
        finally:
            _wb_finish()
        return

    def _ensure_cublas_determinism() -> None:
        if not torch.cuda.is_available():
            return
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            return
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.warning(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 to satisfy deterministic "
            "algorithms. For full reproducibility, export this before launch."
        )

    def evaluate_state(
        state_obj: Dict[str, Any] | Any, method_name: str
    ) -> Dict[str, float]:
        """
        Evaluate an already-loaded state object (either a raw encoder state_dict or a
        dict with key 'encoder'). Always trains a fresh linear head for fairness.
        """
        metrics_runs: List[Dict[str, float]] = []
        prev_det = None
        try:
            _ensure_cublas_determinism()
            prev_det = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
        except Exception:
            prev_det = None

        for seed in seeds:
            # Repro
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            try:
                if prev_det is not None:
                    _ensure_cublas_determinism()
                    torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            # Build & load encoder
            enc = build_encoder(
                gnn_type=args.gnn_type,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                edge_dim=edge_dim,
            )
            if isinstance(state_obj, dict) and "encoder" in state_obj:
                _load_state_dict_forgiving(enc, state_obj["encoder"])
            else:
                _load_state_dict_forgiving(enc, state_obj)
            _maybe_to(enc, device)

            # Train fresh head and log metrics
            mets = train_linear_head(
                dataset=labeled,
                encoder=enc,
                task_type=args.task_type,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=effective_batch_size,
                device=device,
                patience=args.patience,
                devices=args.devices,
                num_workers=getattr(args, "num_workers", -1),
                pin_memory=getattr(args, "pin_memory", True),
                persistent_workers=getattr(args, "persistent_workers", True),
                prefetch_factor=getattr(args, "prefetch_factor", 4),
                bf16=getattr(args, "bf16", False),
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
            metrics_runs.append({k: v for k, v in mets.items() if k != "head"})

        agg = aggregate_metrics(metrics_runs)
        for k, v in agg.items():
            _wb_log({f"{method_name}/{k}": v})
        return agg

    # Thin wrappers that load, then call evaluate_state
    def evaluate_encoder(ckpt_path: str, method_name: str) -> Dict[str, float]:
        state, loaded_path = _safe_load_checkpoint(
            primary=ckpt_path,
            ckpt_dir=None,
            default_name="encoder.pt",
            map_location=device,
            allow_missing=True,
        )
        if not isinstance(state, dict):
            state = {}
        if "encoder" not in state or not state["encoder"]:
            logger.warning("No encoder weights; using random init (path=%r).", loaded_path or getattr(args, "ft_ckpt", None))
            
        return evaluate_state(state, method_name)

    main_start_payload = {"phase": "benchmark", "status": "start"}
    main_start_payload.update(threshold_payload)
    _wb_log(main_start_payload)
    # Evaluate JEPA
    agg_jepa = evaluate_encoder(args.jepa_encoder, "jepa")
    all_results["jepa"] = agg_jepa

    # Evaluate contrastive
    agg_cont: Dict[str, float] = {}
    if args.contrastive_encoder:
        agg_cont = evaluate_encoder(args.contrastive_encoder, "contrastive")
        all_results["contrastive"] = agg_cont

    # Decide which is better
    verdict = "jepa"
    if agg_cont:
        # Choose metric based on task
        if args.task_type == "classification":
            # Higher AUC/ACC is better
            key = (
                "roc_auc_mean"
                if "roc_auc_mean" in agg_jepa
                else ("acc_mean" if "acc_mean" in agg_jepa else None)
            )
            if key and agg_cont.get(key, float("-inf")) > agg_jepa.get(
                key, float("-inf")
            ):
                verdict = "contrastive"
        else:
            # Lower RMSE/MAE is better
            key = (
                "rmse_mean"
                if "rmse_mean" in agg_jepa
                else ("mae_mean" if "mae_mean" in agg_jepa else None)
            )
            if key and agg_cont.get(key, float("inf")) < agg_jepa.get(
                key, float("inf")
            ):
                verdict = "contrastive"

    metric_value_key: Optional[str] = None
    threshold_report: Optional[Dict[str, Any]] = None
    metric_pass: Optional[bool] = None
    best_metric_value: Optional[float] = None
    if threshold_rule is not None:
        for candidate in (f"{threshold_rule.metric}_mean", threshold_rule.metric):
            if any(candidate in mets for mets in all_results.values()):
                metric_value_key = candidate
                break
        if metric_value_key is not None:
            higher_is_better = threshold_rule.metric in HIGHER_IS_BETTER
            threshold_report = {
                "dataset": dataset_name,
                "task": task_name,
                "metric": threshold_rule.metric,
                "threshold": threshold_rule.threshold,
                "metric_key": metric_value_key,
                "orientation": "higher" if higher_is_better else "lower",
                "results": {},
            }
            for method, metrics in all_results.items():
                if metric_value_key in metrics:
                    value = metrics[metric_value_key]
                    passed = bool(
                        value >= threshold_rule.threshold
                        if higher_is_better
                        else value <= threshold_rule.threshold
                    )
                    threshold_report["results"][method] = {
                        "value": value,
                        "passed": passed,
                    }
                    if method == verdict:
                        best_metric_value = value
                        metric_pass = passed

    success_payload = {"phase": "benchmark", "status": "success", "best_method": verdict}
    success_payload.update(threshold_payload)
    if best_metric_value is not None:
        success_payload["benchmark_metric_value"] = best_metric_value
    if metric_pass is not None:
        success_payload["benchmark_pass"] = metric_pass
    logger.info(f"Benchmark completed. Best method: {verdict}")
    _wb_log(success_payload)

    # --- Write JSON/CSV report with all results + verdict ---
    try:
        payload = {"results": all_results, "best_method": verdict}
        if dataset_stats:
            payload["dataset_stats"] = dataset_stats
            payload["dataset_strategy"] = dataset_strategy
        if threshold_report is not None:
            payload["threshold"] = threshold_report
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        # CSV: method,metric,value
        import csv

        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "metric", "value"])
            for method, mets in all_results.items():
                for k, v in mets.items():
                    w.writerow([method, k, v])
            if threshold_rule is not None:
                w.writerow(["threshold/info", "metric", threshold_rule.metric])
                w.writerow(["threshold/info", "threshold", float(threshold_rule.threshold)])
                if metric_value_key is not None:
                    for method, info in (threshold_report or {}).get("results", {}).items():
                        value = info.get("value")
                        if value is not None:
                            w.writerow([f"threshold/{method}", metric_value_key, float(value)])
                        w.writerow([f"threshold/{method}", "passed", info.get("passed")])
                if best_metric_value is not None:
                    w.writerow(["threshold/best_method", "name", verdict])
                    w.writerow(["threshold/best_method", metric_value_key or "value", float(best_metric_value)])
                    if metric_pass is not None:
                        w.writerow(["threshold/best_method", "passed", metric_pass])
            if dataset_stats:
                for key in ("n_total", "n_train", "n_val", "n_test"):
                    if key in dataset_stats:
                        w.writerow(["dataset_stats", key, dataset_stats[key]])
                w.writerow(["dataset_stats", "strategy", dataset_strategy])
        logger.info("Wrote reports: %s , %s", report_json, report_csv)
    except Exception:
        logger.warning("Failed to write reports", exc_info=True)
    finally:
        _wb_finish()
