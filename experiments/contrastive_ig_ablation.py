"""Contrastive-vs-JEPA Integrated-Gradients ablation (standalone).

This driver answers a single reviewer question: is the chemical coherence of
M-JEPA attributions attributable to connected-subgraph masking specifically, or
to the predictive objective broadly?

It runs the existing Tox21 case study (``experiments.case_study``) twice with
*identical* settings, varying only the pretraining objective:

  * JEPA arm:        ``contrastive=False`` (connected-subgraph masking)
  * Contrastive arm: ``contrastive=True``  (InfoNCE)

Both arms pretrain on the same Tox21 molecules at the same budget, use the same
deterministic Bemis-Murcko scaffold split, fine-tune through the same hybrid
schedule, and emit Integrated-Gradients (and IG-motif) attributions. Only the
objective differs, so any difference in attribution quality is isolated to it.

This is deliberately run OUTSIDE the locked CI lineage / frozen-encoder pipeline
(it calls the engine directly, bypassing ``scripts/ci/common.sh`` lineage
gating). It does not write any ``encoder_frozen.ok`` marker and must not be
reported as a graded headline number -- it is a reduced-budget controlled
ablation only.

Usage:
    PYTHONPATH=. python experiments/contrastive_ig_ablation.py \
        --csv data/tox21/data.csv --task NR-AR \
        --pretrain-epochs 30 --finetune-epochs 30 \
        --out outputs/ig_ablation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from experiments.case_study import run_tox21_case_study

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("contrastive_ig_ablation")


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Best-effort JSON-serialisable summary of a CaseStudyResult."""
    if dataclasses.is_dataclass(result):
        payload = dataclasses.asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"repr": repr(result)}

    def _coerce(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # numpy scalars / arrays / misc -> string fallback
        try:
            return float(obj)  # type: ignore[arg-type]
        except Exception:
            return repr(obj)

    return _coerce(payload)


def _run_arm(arm: str, contrastive: bool, args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out) / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    explain_config = {"task_name": args.task, "output_dir": str(out_dir)}
    if args.explain_steps is not None:
        explain_config["steps"] = int(args.explain_steps)

    logger.info(
        "=== %s arm: contrastive=%s pretrain=%d finetune=%d -> %s ===",
        arm,
        contrastive,
        args.pretrain_epochs,
        args.finetune_epochs,
        out_dir,
    )

    result = run_tox21_case_study(
        csv_path=args.csv,
        task_name=args.task,
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        mask_ratio=args.mask_ratio,
        contiguous=True,  # connected-subgraph masking (irrelevant to contrastive arm)
        contrastive=contrastive,
        # end_to_end with no external checkpoint => self-pretrains on the Tox21
        # molecules then fine-tunes the whole model. This is what makes the run
        # self-contained (no frozen lineage) and matched across both arms.
        evaluation_mode="end_to_end",
        full_finetune=True,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        layerwise_decay=args.layerwise_decay,
        device="cpu",
        explain_mode=args.explain_mode,
        explain_config=explain_config,
        # mirror the locked Tox21 fine-tune recipe (method-agnostic)
        dynamic_pos_weight=True,
        calibrate=True,
        calibration_method="temperature",
        threshold_metric="pr_auc",
    )

    summary = _result_to_dict(result)
    (out_dir / "result_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("[%s] wrote summary + IG artifacts under %s", arm, out_dir)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="data/tox21/data.csv")
    p.add_argument("--task", default="NR-AR")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--gnn-type", default="mpnn")
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--layerwise-decay", type=float, default=0.8)
    p.add_argument("--explain-mode", default="ig,ig_motif")
    p.add_argument("--explain-steps", type=int, default=None)
    p.add_argument("--out", default="outputs/ig_ablation")
    p.add_argument(
        "--arms",
        default="jepa,contrastive",
        help="Comma-separated arms to run (jepa, contrastive).",
    )
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    combined: Dict[str, Any] = {}
    for arm in arms:
        combined[arm] = _run_arm(arm, contrastive=(arm == "contrastive"), args=args)

    (Path(args.out) / "ablation_summary.json").write_text(json.dumps(combined, indent=2))
    logger.info("Done. Combined summary at %s/ablation_summary.json", args.out)


if __name__ == "__main__":
    main()
