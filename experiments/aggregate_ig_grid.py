"""Paired aggregation of the IG coherence grid (manuscript-grade statistics).

Walks ``<root>/<assay>/seed<seed>/<arm>/`` produced by ``run_ig_grid.sh`` and,
for every molecule, computes per-molecule attribution-coherence metrics for each
arm. Because the capped test order is seed-deterministic, the same molecules
appear in every arm of a cell, so we can form *paired* per-molecule differences.

Two contrasts are reported, each answering a distinct question:

  * jepa - contrastive       : objective family (predictive vs InfoNCE)
  * jepa - jepa_dispersed    : connected-subgraph masking *specifically*
                               (the reviewer's literal question)

For each contrast and metric we report the paired mean difference, a Wilcoxon
signed-rank p-value, and a bootstrap 95% CI over the pooled molecule-level
differences (pooled across assays/seeds), plus a per-assay breakdown.

Usage:
    PYTHONPATH=. python experiments/aggregate_ig_grid.py --root outputs/ig_grid
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from experiments.analyze_ig_coherence import (
    _gini,
    _largest_cc_fraction,
    _load_atoms,
    _load_edges,
    _topk_mass,
)

ARMS = ["jepa", "jepa_dispersed", "contrastive"]
CONTRASTS = [("jepa", "contrastive"), ("jepa", "jepa_dispersed")]
GRAPH_RE = re.compile(r"(graph_\d+)")

# per-molecule metrics: name -> (higher_is_more_coherent)
METRICS = ["atom_gini", "atom_top20pct_mass", "salient_largest_cc_frac", "motif_top1_share"]


def _mol_key(path: Path) -> Optional[str]:
    m = GRAPH_RE.search(path.name)
    return m.group(1) if m else None


def _per_molecule_metrics(
    arm_dir: Path, task: str, k_frac: float = 0.20, q_frac: float = 0.20
) -> Dict[str, Dict[str, float]]:
    """Return {mol_key: {metric: value}} for one arm."""
    out: Dict[str, Dict[str, float]] = {}
    ig_dir = arm_dir / "ig_explanations" / task
    for af in sorted(ig_dir.glob("*_atoms.csv")):
        key = _mol_key(af)
        if key is None:
            continue
        atoms = _load_atoms(af)
        if not atoms:
            continue
        rec: Dict[str, float] = {
            "atom_gini": _gini(list(atoms.values())),
            "atom_top20pct_mass": _topk_mass(list(atoms.values()), k_frac),
        }
        bf = af.with_name(af.name.replace("_atoms.csv", "_bonds.csv"))
        if bf.exists():
            cc = _largest_cc_fraction(atoms, _load_edges(bf), q_frac)
            if cc is not None:
                rec["salient_largest_cc_frac"] = cc
        out[key] = rec

    # motif top-1 share, keyed by the same graph id (motif dir uses graph_* subdirs)
    motif_dir = arm_dir / "ig_motif_explanations" / task
    if motif_dir.is_dir():
        for summ in motif_dir.glob("*/motif_summary.csv"):
            key = _mol_key(summ.parent)
            if key is None or key not in out:
                continue
            try:
                import csv

                with open(summ, newline="") as fh:
                    norms = [
                        float(r["norm_score"])
                        for r in csv.DictReader(fh)
                        if r.get("norm_score")
                    ]
                s = sum(abs(x) for x in norms)
                if s > 0:
                    out[key]["motif_top1_share"] = max(abs(x) for x in norms) / s
            except (ValueError, KeyError, OSError):
                continue
    return out


def _boot_ci(diffs: np.ndarray, n: int = 5000, seed: int = 0) -> Tuple[float, float]:
    if diffs.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, diffs.size, size=(n, diffs.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/ig_grid")
    p.add_argument("--out", default=None, help="JSON output (default <root>/grid_paired_stats.json)")
    args = p.parse_args()
    root = Path(args.root)

    # cell metrics: per (assay, arm) -> list of {mol_key: {metric: val}} across seeds,
    # but we pair within each (assay, seed) cell then pool the diffs.
    # pooled diffs: contrast -> metric -> list of per-molecule diffs (all cells)
    pooled: Dict[str, Dict[str, List[float]]] = {
        f"{a}_minus_{b}": defaultdict(list) for a, b in CONTRASTS
    }
    per_assay: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: {f"{a}_minus_{b}": defaultdict(list) for a, b in CONTRASTS}
    )
    arm_means: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cells = 0

    for assay_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        assay = assay_dir.name
        for seed_dir in sorted(assay_dir.glob("seed*")):
            arm_metrics = {}
            ok = True
            for arm in ARMS:
                ad = seed_dir / arm
                if not (ad / "ig_explanations" / assay).is_dir():
                    ok = False
                    break
                arm_metrics[arm] = _per_molecule_metrics(ad, assay)
            if not ok or any(not arm_metrics[a] for a in ARMS):
                continue
            cells += 1
            # arm-level means (for descriptive table)
            for arm in ARMS:
                for metric in METRICS:
                    vals = [r[metric] for r in arm_metrics[arm].values() if metric in r]
                    if vals:
                        arm_means[arm][metric].append(float(np.mean(vals)))
            # paired diffs per contrast
            for a, b in CONTRASTS:
                cname = f"{a}_minus_{b}"
                shared = set(arm_metrics[a]) & set(arm_metrics[b])
                for metric in METRICS:
                    for k in shared:
                        if metric in arm_metrics[a][k] and metric in arm_metrics[b][k]:
                            d = arm_metrics[a][k][metric] - arm_metrics[b][k][metric]
                            pooled[cname][metric].append(d)
                            per_assay[assay][cname][metric].append(d)

    def _summ(diffs: List[float]) -> Dict[str, float]:
        arr = np.asarray(diffs, dtype=float)
        if arr.size < 2:
            return {"n": int(arr.size), "mean_diff": float("nan")}
        try:
            w_p = float(stats.wilcoxon(arr, zero_method="wilcox").pvalue)
        except ValueError:
            w_p = float("nan")
        lo, hi = _boot_ci(arr)
        return {
            "n": int(arr.size),
            "mean_diff": float(arr.mean()),
            "median_diff": float(np.median(arr)),
            "wilcoxon_p": w_p,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "frac_positive": float((arr > 0).mean()),
        }

    result = {
        "cells_aggregated": cells,
        "arm_means": {
            arm: {m: float(np.mean(v)) for m, v in mm.items() if v}
            for arm, mm in arm_means.items()
        },
        "pooled": {
            c: {m: _summ(d) for m, d in md.items()} for c, md in pooled.items()
        },
        "per_assay": {
            assay: {
                c: {m: _summ(d) for m, d in md.items()}
                for c, md in contrasts.items()
            }
            for assay, contrasts in per_assay.items()
        },
    }

    out_path = Path(args.out) if args.out else root / "grid_paired_stats.json"
    out_path.write_text(json.dumps(result, indent=2))

    # console table
    print(f"\ncells aggregated: {cells}\n")
    print("ARM MEANS (pooled across cells)")
    print("metric".ljust(28) + "".join(a.rjust(16) for a in ARMS))
    for m in METRICS:
        row = "  " + m.ljust(26)
        for a in ARMS:
            v = result["arm_means"].get(a, {}).get(m, float("nan"))
            row += f"{v:16.4f}"
        print(row)
    for cname, md in result["pooled"].items():
        print(f"\nPAIRED CONTRAST: {cname}  (>0 favours {cname.split('_minus_')[0]})")
        print("metric".ljust(28) + "mean_diff".rjust(11) + "wilcoxon_p".rjust(12)
              + "ci95".rjust(20) + "frac>0".rjust(9) + "n".rjust(7))
        for m in METRICS:
            s = md.get(m, {})
            if "mean_diff" not in s or s.get("n", 0) < 2:
                continue
            ci = f"[{s['ci95_lo']:+.3f},{s['ci95_hi']:+.3f}]"
            print("  " + m.ljust(26) + f"{s['mean_diff']:+11.4f}{s['wilcoxon_p']:12.2e}"
                  + ci.rjust(20) + f"{s['frac_positive']:9.2f}{s['n']:7d}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
