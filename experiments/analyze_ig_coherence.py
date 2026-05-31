"""Quantify attribution coherence for the contrastive-vs-JEPA IG ablation.

The reviewer question is whether M-JEPA's attribution quality is attributable to
connected-subgraph masking specifically rather than to the predictive objective
broadly. "Quality" for molecular attributions has two operational meanings that
we can measure directly from the exported IG artifacts:

1. Concentration / sparsity -- coherent attributions concentrate on a few atoms
   rather than smearing over the whole molecule. We measure the Gini coefficient
   and top-k mass of |IG| over atoms, averaged across the matched test set.

2. Spatial connectivity -- coherent attributions highlight *connected*
   substructures (rings/functional groups), not scattered atoms. Using the bond
   list, we take the top-q fraction of atoms by |IG| and measure the size of the
   largest connected component among them (as a fraction of the selected set).
   Higher => the salient atoms form a connected subgraph.

Both arms are scored on the SAME molecules (same scaffold split, same atoms), so
the comparison isolates the pretraining objective. This is a reduced-budget
controlled ablation run outside the locked pipeline; absolute numbers are not
comparable to the headline frozen-encoder result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _gini(values: List[float]) -> float:
    xs = sorted(abs(v) for v in values)
    n = len(xs)
    if n == 0:
        return float("nan")
    total = sum(xs)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _topk_mass(values: List[float], k_frac: float) -> float:
    xs = sorted((abs(v) for v in values), reverse=True)
    n = len(xs)
    if n == 0:
        return float("nan")
    total = sum(xs)
    if total <= 0:
        return 0.0
    k = max(1, int(math.ceil(k_frac * n)))
    return sum(xs[:k]) / total


def _load_atoms(path: Path) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            # The atom index is the stable node id in the "id" field
            # (<graph_idx>_<node_idx>). The "type" column is NOT reliable: the
            # RDKit-backed exporter writes element symbols (C/N/O) there when
            # SMILES parsing succeeds, and only falls back to "atom_<idx>"
            # otherwise, so filtering on type would drop element-symbol rows.
            rid = row.get("id", "")
            node_part = rid.rsplit("_", 1)[-1]
            try:
                idx = int(node_part)
                scores[idx] = float(row["ig_score"])
            except (ValueError, KeyError):
                continue
    return scores


def _load_edges(path: Path) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rid = row.get("id", "")
            if "-" not in rid:
                continue
            a, _, b = rid.partition("-")
            try:
                edges.append((int(a), int(b)))
            except ValueError:
                continue
    return edges


def _largest_cc_fraction(
    atom_scores: Dict[int, float], edges: List[Tuple[int, int]], q_frac: float
) -> Optional[float]:
    """Fraction of the top-q salient atoms that lie in their largest connected
    component (using molecular bonds restricted to the salient atom set)."""
    n = len(atom_scores)
    if n < 3 or not edges:
        return None
    ordered = sorted(atom_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    k = max(2, int(math.ceil(q_frac * n)))
    selected = {idx for idx, _ in ordered[:k]}
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        if a in selected and b in selected:
            adj[a].append(b)
            adj[b].append(a)
    seen: set = set()
    best = 0
    for node in selected:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp = 0
        while stack:
            cur = stack.pop()
            comp += 1
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        best = max(best, comp)
    return best / len(selected)


def _analyze_arm(arm_dir: Path, task: str, k_frac: float, q_frac: float) -> Dict[str, float]:
    ig_dir = arm_dir / "ig_explanations" / task
    atom_files = sorted(ig_dir.glob("*_atoms.csv"))
    ginis: List[float] = []
    masses: List[float] = []
    cc_fracs: List[float] = []
    motif_top1: List[float] = []
    n_mols = 0

    for af in atom_files:
        atoms = _load_atoms(af)
        if not atoms:
            continue
        n_mols += 1
        vals = list(atoms.values())
        ginis.append(_gini(vals))
        masses.append(_topk_mass(vals, k_frac))
        bf = af.with_name(af.name.replace("_atoms.csv", "_bonds.csv"))
        if bf.exists():
            cc = _largest_cc_fraction(atoms, _load_edges(bf), q_frac)
            if cc is not None:
                cc_fracs.append(cc)

    # motif concentration: how dominant is the single most important motif?
    motif_dir = arm_dir / "ig_motif_explanations" / task
    if motif_dir.is_dir():
        for summ in motif_dir.glob("*/motif_summary.csv"):
            try:
                with open(summ, newline="") as fh:
                    norms = [
                        float(r["norm_score"])
                        for r in csv.DictReader(fh)
                        if r.get("norm_score")
                    ]
                if norms:
                    s = sum(abs(x) for x in norms)
                    if s > 0:
                        motif_top1.append(max(abs(x) for x in norms) / s)
            except (ValueError, KeyError, OSError):
                continue

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "n_molecules": n_mols,
        "atom_gini_mean": _mean(ginis),
        f"atom_top{int(k_frac*100)}pct_mass_mean": _mean(masses),
        f"salient_top{int(q_frac*100)}pct_largest_cc_frac_mean": _mean(cc_fracs),
        "motif_top1_share_mean": _mean(motif_top1),
        "n_with_motif": len(motif_top1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/ig_ablation")
    p.add_argument("--task", default="NR-AR")
    p.add_argument("--k-frac", type=float, default=0.2, help="top-k atom mass fraction")
    p.add_argument("--q-frac", type=float, default=0.2, help="salient-atom fraction for CC")
    p.add_argument("--arms", default="jepa,contrastive")
    args = p.parse_args()

    root = Path(args.root)
    out: Dict[str, Dict[str, float]] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arm_dir = root / arm
        if not arm_dir.is_dir():
            print(f"[skip] {arm}: {arm_dir} not found")
            continue
        out[arm] = _analyze_arm(arm_dir, args.task, args.k_frac, args.q_frac)

    (root / "coherence_metrics.json").write_text(json.dumps(out, indent=2))

    # pretty table
    if out:
        metrics = sorted({k for arm in out.values() for k in arm})
        arms = list(out)
        w = max(len(m) for m in metrics) + 2
        header = "metric".ljust(w) + "".join(a.rjust(16) for a in arms)
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = m.ljust(w)
            for a in arms:
                v = out[a].get(m, float("nan"))
                row += (f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(16)
            print(row)
    print(f"\nWrote {root/'coherence_metrics.json'}")


if __name__ == "__main__":
    main()
