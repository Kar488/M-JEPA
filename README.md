# M-JEPA

> **Summary**
> - Frozen encoder lineages are immutable.
> - Dependent runs must set `PRETRAIN_EXP_ID` to reuse them.
> - To rebuild, remove or override the freeze marker explicitly.

Joint Embedding Predictive Architectures (JEPA) for molecular graphs. The
project includes data download scripts, self-supervised pretraining, and
utilities for downstream evaluation on MoleculeNet benchmarks.

For a reviewer-focused, end-to-end reproduction path, see [`REPRODUCE.md`](REPRODUCE.md).

## 🔒 Frozen Encoder Lineages

CI discovers encoders that carry the `bench/encoder_frozen.ok` marker and
reuses them for downstream stages. Phase‑1 sweeps now allocate their own
`EXP_ID` and set `GRID_EXP_ID` to that value so fresh grid logs land under a
dedicated directory. When a freeze marker is present, automation launches new
runs under a fresh `EXP_ID` while binding `PRETRAIN_EXP_ID` (and, when reading
existing sweeps, `GRID_EXP_ID`) to the frozen lineage.

```bash
# Run tox21 grading on a frozen encoder
export PRETRAIN_EXP_ID=1759825317
bash scripts/ci/run-tox21.sh
# Output -> /data/mjepa/experiments/$RUN_ID/tox21/
```

See `docs/frozen_lineage_policy.rst` for override flags and lineage semantics,
including `FORCE_UNFREEZE_GRID=1` (rebuild a frozen lineage) and
`FORCE_RERUN=stage1,stage2` to selectively invalidate caches.

## Paper Reproduction Map

This section maps the manuscript's main-text figures and tables to the repository stages and the existing reproduction entry points documented in [`REPRODUCE.md`](REPRODUCE.md). Where a manuscript panel was assembled from multiple sweep runs or W&B exports rather than one deterministic local file, the outputs are described generically rather than as fixed filenames.

Unless otherwise stated, experiments follow the seed configurations described in Table 1 (seeds `{1,2,3,4,5}` for Phase-1). Downstream runs use multiple seeds for averaging where applicable.

- **Figure 1: Overview of predictive self-supervised learning on molecular graphs**
  - Stage: Cross-phase overview (Phase 1 → Phase 3).
  - Generated from: Repository workflow summarized in `pretrain`, `finetune` / `evaluate`, `benchmark`, and `tox21`.
  - REPRODUCE.md commands: see **Full pretraining**, **Fine-tuning / evaluation**, **Benchmarking**, and **Tox21 case study**.
  - Output: conceptual workflow figure; no single deterministic output artifact in the repo.
- **Figure 2: Leakage-resistant evaluation pipeline (3-phase)**
  - Stage: Cross-phase overview (Phase 1, Phase 2, Phase 3).
  - Generated from: the staged workflow reproduced through the documented local commands and the remote `ci-vast` workflow path.
  - REPRODUCE.md commands: see **Full pretraining**, **Fine-tuning / evaluation**, **Benchmarking**, **Tox21 case study**, and the note that remote automation uses the `ci-vast` workflow.
  - Output: conceptual pipeline summary; no single deterministic output artifact in the repo.
- **Figure 3: Phase-1 objective comparison (JEPA vs InfoNCE RMSE distributions)**
  - Stage: Phase 1.
  - Generated from: repeated Phase-1 objective-screening runs that combine pretrained encoders with downstream evaluation.
  - REPRODUCE.md commands: `python scripts/train_jepa.py pretrain ...` plus downstream `finetune` / `evaluate` runs; for full automation, the equivalent remote path is the `ci-vast` workflow.
  - Output: aggregated RMSE metrics from multiple runs, typically surfaced through experiment directories, benchmark JSON/CSV reports, and/or W&B logs.
- **Figure 4: Phase-2 sweep structure (parallel coordinates)**
  - Stage: Phase 2.
  - Generated from: Phase-2 hyperparameter sweep / selection runs in the documented remote workflow.
  - REPRODUCE.md commands: the documented `ci-vast` remote workflow path together with the same `pretrain` and downstream evaluation entry points used in local reproduction.
  - Output: sweep summaries and exported configuration/metric artifacts under CI experiment directories, reports, and/or W&B logs.
- **Figure 5: ESOL RMSE by augmentation profile and backbone**
  - Stage: Phase 2.
  - Generated from: Phase-2 augmentation/backbone comparisons evaluated through pretraining plus downstream regression benchmarking.
  - REPRODUCE.md commands: `pretrain` plus downstream `finetune` / `evaluate` or `benchmark`, depending on whether evaluation is performed from a labeled dataset source or explicit split directories.
  - Output: aggregated regression metrics from multiple runs in reports, experiment directories, and/or W&B logs.
- **Figure 6: Augmentation and masking effects**
  - Stage: Phase 2.
  - Generated from: ablation-style sweep runs varying augmentation and masking settings.
  - REPRODUCE.md commands: `pretrain` plus downstream `finetune` / `evaluate`; for the automated manuscript-style sweep path, use the documented `ci-vast` workflow.
  - Output: aggregated comparison metrics and sweep exports under reports, CI experiment directories, and/or W&B logs.
- **Figure 7: Reliability diagrams (Tox21)**
  - Stage: Phase 3.
  - Generated from: Tox21 evaluation runs.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 ...`.
  - Output: Tox21 report files under the chosen `--report-dir`, including aggregate and per-task JSON/CSV outputs; reliability plots may also be regenerated from those outputs or W&B logs.
- **Figure 8: Discrimination vs calibration trade-off (ΔPR-AUC vs ΔECE)**
  - Stage: Phase 3.
  - Generated from: aggregated Tox21 evaluation results across assays / model variants.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 ...`.
  - Output: aggregated Tox21 metrics CSV/JSON outputs and related report artifacts under the chosen `--report-dir`.
- **Figure 9: IG attribution patterns across assays**
  - Stage: Phase 3.
  - Generated from: Tox21 assay-level interpretation outputs.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 --explain-mode ig ...` (the Tox21 command only emits attribution artifacts when `--explain-mode` is set).
  - Output: per-task Tox21 JSON/CSV outputs, prediction files, attribution artifacts, and report artifacts under the chosen `--report-dir`; TODO: confirm exact attribution export path used for the manuscript figure.
- **Figure 10: NR-AR attribution case study**
  - Stage: Phase 3.
  - Generated from: Tox21 evaluation / interpretation focused on the `NR-AR` assay.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 --tasks NR-AR --explain-mode ig ...`.
  - Output: assay-specific Tox21 JSON/CSV outputs, attribution artifacts, and related report artifacts under the chosen `--report-dir`; TODO: confirm exact attribution export path used for the manuscript figure.
- **Figure 11: SR-HSE attribution case study**
  - Stage: Phase 3.
  - Generated from: Tox21 evaluation / interpretation focused on the `SR-HSE` assay.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 --tasks SR-HSE --explain-mode ig ...`.
  - Output: assay-specific Tox21 JSON/CSV outputs, attribution artifacts, and related report artifacts under the chosen `--report-dir`; TODO: confirm exact attribution export path used for the manuscript figure.
- **Figure 12: NR-AR-LBD attribution case study**
  - Stage: Phase 3.
  - Generated from: Tox21 evaluation / interpretation focused on the `NR-AR-LBD` assay.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 --tasks NR-AR-LBD --explain-mode ig ...`.
  - Output: assay-specific Tox21 JSON/CSV outputs, attribution artifacts, and related report artifacts under the chosen `--report-dir`; TODO: confirm exact attribution export path used for the manuscript figure.
- **Table 1: Phase-1 sweep configuration**
  - Stage: Phase 1.
  - Generated from: Phase-1 sweep configuration and seed settings.
  - REPRODUCE.md commands: Phase-1 reproduction uses the documented `pretrain` plus downstream evaluation commands; automated sweep execution follows the documented `ci-vast` workflow path.
  - Output: sweep configuration records, experiment metadata, and/or W&B configs rather than a single fixed repository file.
- **Table 2: ESOL RMSE distributions (Phase-1)**
  - Stage: Phase 1.
  - Generated from: Phase-1 objective-screening runs evaluated on downstream regression metrics.
  - REPRODUCE.md commands: `pretrain` plus downstream `finetune` / `evaluate`; automated multi-run execution follows the documented `ci-vast` workflow path.
  - Output: aggregated RMSE metrics in reports, experiment directories, and/or W&B logs.
- **Table 3: Phase-2 configuration selection (top candidates)**
  - Stage: Phase 2.
  - Generated from: Phase-2 sweep, recheck, and configuration export steps.
  - REPRODUCE.md commands: the documented `ci-vast` workflow path together with the same `pretrain` and downstream evaluation entry points.
  - Output: exported best-configuration artifacts, sweep summaries, and associated JSON/CSV or W&B records.
- **Table 4: Tox21 assay-level results (ROC-AUC, PR-AUC, Brier, ECE)**
  - Stage: Phase 3.
  - Generated from: assay-level Tox21 evaluation.
  - REPRODUCE.md commands: `python scripts/train_jepa.py tox21 ...`.
  - Output: aggregated `tox21_<mode>_metrics.csv`, per-task JSON/CSV outputs, score files, and run manifests under the chosen `--report-dir`.

## Dataset Provenance and Role

This section records what the checked-in dataset folders appear to represent from repository evidence alone. It intentionally stays conservative: where an exact public source snapshot is not documented in code or docs, it is marked as a TODO rather than guessed.

- **`data/tox21/`**
  - Role in pipeline: downstream evaluation and Tox21 case-study input for `finetune`, `evaluate`, and `tox21`.
  - What is present: `data/tox21/data.csv`, a single labeled dataset file.
  - Dataset type in this repo: bundled labeled benchmark CSV rather than a generated split directory.
  - Likely public origin from repo evidence: treated throughout the repo as a MoleculeNet-style Tox21 dataset. TODO: confirm exact public source and snapshot.
- **`data/ZINC-canonicalized/`**
  - Role in pipeline: unlabeled pretraining / proxy-task corpus for `pretrain` and CI cache warming.
  - What is present: flat parquet shard files suitable for `--unlabeled-dir`.
  - Dataset type in this repo: preprocessed unlabeled shard dataset.
  - Likely public origin from repo evidence: ZINC-family unlabeled molecules. TODO: confirm exact public source and snapshot, including the canonicalization procedure.
- **`data/katielinkmoleculenet_benchmark/`**
  - Role in pipeline: benchmark fixture for downstream evaluation when explicit split folders are required.
  - What is present: `train/`, `val/`, and `test/` directories.
  - Dataset type in this repo: pre-generated split dataset / benchmark artifact.
  - Likely public origin from repo evidence: MoleculeNet-related benchmark fixture. TODO: confirm exact public source and snapshot.

Clarifications that are easy to confuse during review:

- **`data/tox21/data.csv`** is one labeled table. It is not a pre-materialized `train/val/test` split.
- **`data/katielinkmoleculenet_benchmark/train|val|test`** is already split on disk and is consumed as a benchmark fixture.
- **Runtime scaffold splits** are generated by code, either explicitly with `scripts/make_scaffold_splits.py` or implicitly inside `finetune` / `tox21` when scaffold splitting is enabled. They are execution-time artifacts, not the same thing as the checked-in benchmark directories.

## Phase 3: Tox21 Scaffold-Split Evaluation

Reviewer-facing Phase-3 command path:

```bash
python scripts/train_jepa.py tox21 \
  --csv data/tox21/data.csv \
  --tasks NR-AR NR-AhR SR-p53 \
  --encoder-checkpoint ckpts/pretrain/encoder.pt \
  --evaluation-mode hybrid \
  --report-dir reports/tox21 \
  --device cuda
```

This `tox21` path performs Tox21 assay evaluation from `data/tox21/data.csv` and, when RDKit scaffold splitting is available, it generates Bemis–Murcko scaffold splits at runtime rather than reading a pre-existing `train/val/test` split folder. If the runtime scaffold split cannot satisfy the command's validation/test diversity and positive-count checks, the code falls back to stratified splitting and records the chosen split strategy in the emitted diagnostics.

For the manuscript mapping, this is the repository path that supports the Phase-3 downstream evaluation artifacts referenced by **Table 4** and the report files used for **Figures 7–12**. The command writes per-task JSON/CSV summaries plus score and calibration artifacts under the chosen report directory, including `tox21_<task>.json`, `tox21_<task>.csv`, `tox21_<task>_scores.csv`, `tox21_<task>_reliability_bins.json`, `run_manifest_<task>.json`, `tox21_summary.json`, `run_manifest.json`, `stage-outputs/`, and, when per-task CSVs are present, `tox21_<evaluation_mode>_metrics.csv`. The per-task/aggregate metric exports include the Tox21 evaluation metrics surfaced by the command path, including ROC-AUC, PR-AUC, Brier score, and ECE.

`cache/pretrain/` and `cache/finetune/` hold model/cache artifacts for pretraining and fine-tuning runs; they are not the scaffold split definitions themselves. Exact numeric values may vary with seeds, hardware, and runtime configuration, especially because the Tox21 path can retry split seeds and can fall back from scaffold to stratified splitting when needed.

## Reviewer Quick Start

These commands use data that is already checked into this repository, so they can be run without downloading any additional datasets. They are intentionally small reviewer-facing smoke tests rather than full manuscript-scale jobs.

```bash
# 1) Minimal pretraining smoke test on bundled unlabeled shards
python scripts/train_jepa.py pretrain \
  --unlabeled-dir data/ZINC-canonicalized \
  --output encoder.pt \
  --ckpt-dir ckpts/pretrain_smoke \
  --epochs 1 \
  --sample-unlabeled 128 \
  --batch-size 32 \
  --device cpu

# 2) Minimal downstream evaluation on bundled Tox21 labels
python scripts/train_jepa.py evaluate \
  --labeled-dir data/tox21 \
  --encoder ckpts/pretrain_smoke/encoder.pt \
  --label-col NR-AR \
  --task-type classification \
  --epochs 1 \
  --batch-size 32 \
  --device cpu

# 3) Optional Tox21 smoke path on bundled assay data
python scripts/train_jepa.py tox21 \
  --csv data/tox21/data.csv \
  --tasks NR-AR \
  --encoder-checkpoint ckpts/pretrain_smoke/encoder.pt \
  --evaluation-mode hybrid \
  --report-dir reports/tox21_smoke \
  --device cpu
```

Expected smoke-test outputs are a small encoder checkpoint under `ckpts/pretrain_smoke/` plus evaluation or Tox21 JSON/CSV reports under the chosen checkpoint / report directories. For a fuller explanation of outputs and split behavior, see [`REPRODUCE.md`](REPRODUCE.md).

## Local development

1. **Install dependencies**

   ```bash
   git clone https://github.com/.../M-JEPA.git
   cd M-JEPA
   python -m pip install --upgrade pip setuptools wheel "numpy<2"
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1
   pip install --no-cache-dir -f https://data.pyg.org/whl/torch-2.2.1+cpu.html torch-scatter==2.1.2
   pip install torch-geometric==2.5.3
   pip install -r requirements.txt
   pre-commit install
   ```

   Turn on developer mode in developer settings for symlinks to work when running tests

   Install torch and deepchem through conda if you prefer GPU builds.
   RDKit is bundled via PyPI (`rdkit`) in `requirements.txt`; conda/micromamba
   installs remain supported for environments that prefer the conda-forge build.
   For Parquet support, install either `pyarrow` or `fastparquet`; tests will
   skip gracefully if neither is available.

2. **Authenticate with Weights & Biases**

   ```bash
   wandb login
   ```
   or set `WANDB_API_KEY` in the environment (e.g., `export WANDB_API_KEY=...` $env:WANDB_API_KEY = "your_api_key_here").
   This variable is used by `main.py`, plotting helpers, and the tests to
   establish a connection to W&B. Logging is disabled by default when the
   variable is absent.

   To publish full project reports with `reports/build_wandb_report.py`, install
   the optional `wandb-workspaces` package (provides the W&B Reports API) in
   addition to the core `wandb` client. Without it, the script falls back to
   writing `reports/FIGURE_MANIFEST.md` locally.

   In Repository → Settings → Secrets and variables → Actions → New repository secret add:

   WANDB_API_KEY = your W&B key

3. **Datasets**

   ### Repository data contents

   **Included public benchmark / example assets already in this repository**
   - `data/tox21/data.csv`: labeled Tox21-style benchmark CSV used by `finetune`,
     `evaluate`, and `tox21` examples.
   - `data/ZINC-canonicalized/`: unlabeled parquet shards used by pretraining and
     CI cache-warming paths.
   - `data/katielinkmoleculenet_benchmark/train|val|test`: pre-generated benchmark
     split directories used by the benchmark/CI paths.
   - `data/BASF_AIPubChem_v4/`: additional unlabeled parquet shards bundled as an
     offline example dataset.

   **Optional externally obtained data**
   - Any larger or alternate unlabeled corpus you want to pass to
     `scripts/train_jepa.py pretrain --unlabeled-dir ...`.
   - Any alternate labeled benchmark dataset you want to pass to
     `finetune` / `evaluate` / `benchmark`.

   **Generated caches / outputs (not source data)**
   - Runtime scaffold split folders produced by `scripts/make_scaffold_splits.py`.
   - Graph caches under `--cache-dir` (including `prebuilt_datasets/`).
   - Checkpoints, reports, and CI experiment artifacts under `ckpts/`, `reports/`,
     or `/data/mjepa/experiments/<EXP_ID>/...`.

   Notes:
   - `pretrain` still expects `--unlabeled-dir` to point to a **flat directory of
     `.parquet` or `.csv` shards** such as `data/ZINC-canonicalized`.
   - `scripts/download_unlabeled.py` creates `data/unlabeled/train|val|test`; when
     using that helper, pass one shard directory such as `data/unlabeled/train`,
     **not** the `data/unlabeled` parent.
   - `finetune` / `evaluate` no longer require reviewers to manually place a Tox21
     CSV if they use the repository-bundled `data/tox21/data.csv`.
   - `--cache-dir` stores **featurized graph caches only**. Caches speed up repeat
     runs but do not replace raw data, benchmark fixtures, or split metadata.

   Minimal local examples:

   ```bash
   # Pretraining smoke path using bundled unlabeled shards
   python scripts/train_jepa.py pretrain --unlabeled-dir data/ZINC-canonicalized

   # Fine-tuning / evaluation using the bundled Tox21 CSV
   python scripts/train_jepa.py finetune --labeled-dir data/tox21 --labeled-csv data/tox21/data.csv --encoder encoder.pt --label-col NR-AR
   python scripts/train_jepa.py evaluate --labeled-dir data/tox21 --encoder encoder.pt --label-col NR-AR
   ```

   Keep the README brief and use [`REPRODUCE.md`](REPRODUCE.md) for the detailed
   repo-specific data layout, provenance notes, split behavior, smoke tests, and
   output verification guidance.

4. **Run tests**

   Install the Python dependencies first so optional test helpers such as
   ``pandas`` and ``pyyaml`` are available:

   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   cd 'C:\\Users\\karth\\Dropbox\\Documents\\synched folder\\my.certifications\\La trobe\\research\\coding\\M-JEPA>'
   pytest --cache-clear tests -v -q -s -o log_cli=true -W ignore
   # or single one
   pytest --cache-clear tests/test_plot_small.py -q -s -o log_cli=true -W ignore
   ```

5. **Running on a server (GitHub Actions ➜ Vast.ai via SSH)**
  See vast.md for details
