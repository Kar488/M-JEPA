#!/usr/bin/env bash
# Orchestrate the contrastive-vs-JEPA IG coherence grid.
#   arms (in driver): jepa (connected), jepa_dispersed (contiguity=0), contrastive
#   grid: <assays> x <seeds>, one driver call per (assay, seed) cell.
# Reduced-budget controlled ablation (CPU). NOT a graded headline number.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export MJEPA_ALLOW_DATA_FALLBACKS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

CSV="data/tox21/data_rdkit_clean.csv"
ASSAYS="${ASSAYS:?set ASSAYS, e.g. 'NR-AR SR-MMP'}"
SEEDS="${SEEDS:-42 1 2}"
ROOT="${ROOT:-outputs/ig_grid}"
PRE="${PRE:-20}"; FT="${FT:-20}"; STEPS="${STEPS:-24}"; MAXMOL="${MAXMOL:-250}"

mkdir -p "$ROOT"
for assay in $ASSAYS; do
  for seed in $SEEDS; do
    out="$ROOT/$assay/seed$seed"
    if [ -f "$out/ablation_summary.json" ]; then
      echo "[skip] $assay seed$seed already done"; continue
    fi
    mkdir -p "$out"
    echo "[run] $assay seed$seed -> $out ($(date -u +%T))"
    python experiments/contrastive_ig_ablation.py \
      --csv "$CSV" --task "$assay" --seed "$seed" \
      --eval-mode hybrid --pretrain-epochs "$PRE" --finetune-epochs "$FT" \
      --explain-mode ig,ig_motif --explain-steps "$STEPS" --max-molecules "$MAXMOL" \
      --arms jepa,jepa_dispersed,contrastive \
      --out "$out" > "$out/run.log" 2>&1 \
      && echo "[ok]   $assay seed$seed ($(date -u +%T))" \
      || echo "[FAIL] $assay seed$seed (see $out/run.log)"
  done
done
echo "[done] grid complete for ASSAYS='$ASSAYS' ($(date -u +%T))"
