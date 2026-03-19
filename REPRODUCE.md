# Reproducibility Guide

## Purpose

This repository supports reproducing the documented M-JEPA workflow for molecular graphs:

1. unlabeled-graph pretraining with `scripts/train_jepa.py pretrain`,
2. downstream linear-probe or fine-tuning evaluation with `finetune` / `evaluate`,
3. encoder comparison with `benchmark`, and
4. the Tox21 case study with `tox21`.

A successful run should produce:

- a pretrained encoder checkpoint,
- per-stage checkpoints and logs,
- an encoder manifest describing the pretrained artifact,
- downstream metrics/reports, and
- for Tox21, per-task summaries and prediction files.

## Supported environments

The code path is written to run in each of the following environments.

- **Local desktop CPU:** supported for smoke tests, debugging, and small dataset slices.
- **Local desktop GPU:** supported for practical single-machine training.
- **Remote GPU / cluster:** supported through the repository's CI wrappers and the `ci-vast` workflow; this is the main automated execution path for full runs.

## Minimal setup

### Python

- Use **Python 3.10** when possible. This is the version used by the repository's GitHub Actions quick-test environment and the documented micromamba setup in `.github/workflows/ci-vast.yml`.
- Python 3.12+ is not a good default here because the pinned `rdkit-pypi` wheel in `requirements.txt` is only installed for Python `<3.12`.

### Python dependencies

CPU-only local setup:

```bash
python -m pip install --upgrade pip setuptools wheel "numpy<2"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1
pip install --no-cache-dir -f https://data.pyg.org/whl/torch-2.2.1+cpu.html torch-scatter==2.1.2
pip install torch-geometric==2.5.3
pip install -r requirements.txt
```

GPU local setup follows the same pattern but should use the appropriate PyTorch wheel index for the target CUDA runtime. The repository's automated GPU environment is prepared by `scripts/ci/prepare_env.sh`.

### System/runtime dependencies

Only document what the repository actually relies on:

- **RDKit** is required for molecular featurization and scaffold splitting.
- **PyTorch / PyTorch Geometric / torch-scatter** are required for training.
- **Parquet support** requires either `pyarrow` or `fastparquet`; both are already listed in `requirements.txt`.
- **DeepChem** is only needed for helper paths that download MoleculeNet-style datasets; it is not required if datasets are already present on disk.

### Environment variables

No environment variable is strictly required for a local smoke test when datasets are provided explicitly on disk.

Optional variables:

- `WANDB_API_KEY` if you want Weights & Biases logging.
- `CUDA_VISIBLE_DEVICES` if you want to restrict visible GPUs.

For automated remote runs, the repository expects environment variables such as `EXP_ID`, `PRETRAIN_EXP_ID`, `EXPERIMENTS_ROOT`, and `ARTIFACTS_DIR` to be set by the CI wrappers rather than by hand.

## Repository and data layout

The code uses a few distinct path conventions. Keeping them separate is important for reproducibility.

### Repository-bundled `data/` directory in this public repo

This public repository already contains several benchmark/example datasets under `data/`. Reviewers do **not** need to manually place these specific folders on disk before running the documented smoke paths.

| Data folder | What it is in this repo | Data status | Source / provenance from repo evidence | Used by command(s) |
| --- | --- | --- | --- | --- |
| `data/tox21/data.csv` | Labeled Tox21-style assay table with `smiles` plus the standard Tox21 task columns used by the repository's Tox21 workflows. | Bundled labeled benchmark CSV. | Repository code and docs consistently treat Tox21 as a MoleculeNet-style benchmark task, but the exact public source snapshot for this checked-in CSV is not stated. TODO: confirm exact public source / snapshot for this dataset. | `scripts/train_jepa.py finetune`, `evaluate`, and `tox21`; also the default pretrain probe dataset. |
| `data/ZINC-canonicalized/` | Unlabeled parquet shard directory used as the pretraining corpus path in CI and cache warming. | Bundled curated/preprocessed unlabeled shard dataset. | `scripts/download_unlabeled.py` documents ZINC as a public source family and multiple CI paths explicitly point to `data/ZINC-canonicalized`, but the exact canonicalization procedure and snapshot are not documented here. TODO: confirm exact public source / snapshot for this dataset. | `scripts/train_jepa.py pretrain`; CI pretrain/cache-warm paths. |
| `data/katielinkmoleculenet_benchmark/train`, `val`, `test` | Pre-generated benchmark split directories consumed directly by benchmark/CI evaluation paths. | Bundled benchmark-ready split artifacts. | Folder naming and workflow usage indicate a MoleculeNet benchmark fixture, but the exact upstream public source and split-generation history are not documented in the repository. TODO: confirm exact public source / snapshot for this dataset. | `scripts/train_jepa.py benchmark`; CI benchmark/cache-warm paths. |
| `data/BASF_AIPubChem_v4/` | Additional unlabeled parquet shard directory bundled as an offline example corpus. | Bundled curated/preprocessed unlabeled shard dataset. | Repo docs mention this folder as an example shard set for offline experiments; no exact upstream provenance is documented. TODO: confirm exact public source / snapshot for this dataset. | Not part of the main reviewer smoke path; can be supplied to `pretrain --unlabeled-dir` as an alternate unlabeled corpus. |

#### Distinguishing raw public data, curated fixtures, runtime splits, and caches

- **`data/tox21/data.csv`** is a single labeled CSV table. `finetune`, `evaluate`, and `tox21` read it as a labeled dataset source and can generate train/validation/test partitions from it at runtime.
- **`data/ZINC-canonicalized/...`** is an unlabeled parquet shard directory. `pretrain` expects exactly this style of flat shard directory for `--unlabeled-dir`.
- **`data/katielinkmoleculenet_benchmark/train|val|test`** are already-split benchmark fixtures on disk. These are different from runtime-generated splits because the split membership is materialized as separate folders before execution.
- **Runtime-generated scaffold splits** are created either by `scripts/make_scaffold_splits.py` or internally by `finetune` / `tox21` when scaffold splitting is enabled and SMILES are available. Those runtime splits are execution-time artifacts, not the same thing as the checked-in benchmark fixture directories.
- **Cache artifacts** under `--cache-dir` store featurized graphs and dataset caches for speed. They are neither the source data nor an authoritative record of split membership.

#### Important note on the pre-existing `train/val/test` folders

The checked-in `data/katielinkmoleculenet_benchmark/train|val|test` folders should be read as **benchmark-ready split artifacts / fixtures** that the repository can consume immediately. They are distinct from scaffold splits generated at runtime by `finetune` or `tox21`. From repository evidence alone, it is **not** possible to state that these folders are the exact manuscript splits; this guide therefore treats them neutrally as archival benchmark fixtures unless and until a more specific provenance record is added.

### Source data layouts

#### Unlabeled corpora used by `pretrain`

`pretrain` expects `--unlabeled-dir` to point to a **flat directory of `.parquet` or `.csv` shards**.

Example:

```text
data/
  zinc_pretrain/
    0000.parquet
    0001.parquet
    0002.parquet
```

Important: `scripts/download_unlabeled.py` defaults to a split layout:

```text
data/
  unlabeled/
    train/
      0000.parquet
    val/
      0000.parquet
    test/
      0000.parquet
```

If you use that helper unchanged, pass a shard directory such as `data/unlabeled/train` to `--unlabeled-dir`, not the `data/unlabeled` parent.

#### Labeled datasets used by `finetune` / `evaluate`

These commands operate on a **single labeled dataset source** and create train/val/test splits in memory.

Common layouts:

```text
data/
  tox21/
    data.csv
```

or

```text
data/
  esol/
    0000.parquet
```

`finetune` accepts:

- `--labeled-dir <directory>` for a directory of shards, and optionally
- `--labeled-csv <file>` when a single CSV file should be used from that directory.

#### Explicit split directories used by `benchmark` or archival split generation

The repository also supports explicit split folders such as:

```text
data/
  tox21_scaffold/
    train/
      0000.parquet
    val/
      0000.parquet
    test/
      0000.parquet
```

`benchmark` detects this layout automatically. `finetune` does **not** require pre-generated split directories because it computes splits itself.

### Cache / experiment / report layouts

Typical generated directories are:

```text
cache/
  graphs_10m/
    <file-specific graph caches>
    prebuilt_datasets/
      <hashed dataset pickles>

ckpts/
  pretrain/
  finetune/

reports/

/data/mjepa/experiments/<EXP_ID>/        # CI / remote convention
  artifacts/
  pretrain/
  finetune/
  tox21/
  bench/
  report/
```

## Raw data vs generated artifacts

Keep the following categories distinct.

### Raw or manually supplied data

- Unlabeled corpus shards (`.parquet` / `.csv`) supplied to `--unlabeled-dir`.
- Labeled CSV/Parquet files such as `data/tox21/data.csv`.
- Optional manually created split directories under `train/`, `val/`, and `test/`.

### Generated scaffold splits

- `scripts/make_scaffold_splits.py` writes explicit `train/`, `val/`, and `test/` directories from a single CSV/Parquet file.
- `finetune` and the Tox21 case study can also generate scaffold splits internally at runtime from SMILES strings.
- These split definitions are logically separate from graph-feature caches.

### Cached graph features / cached artifacts

There are two main cache types.

1. **Per-file graph caches** written by `data.mdataset.GraphDataset.from_csv` / `from_parquet` / `from_directory` when `--cache-dir` is set. These are pickled graph objects plus schema metadata.
2. **`prebuilt_datasets` caches** under `cache/.../prebuilt_datasets`, used mainly by sweep/cache-warming helpers to avoid rebuilding full datasets repeatedly.

These caches are for speed only. They are **not** a substitute for documenting:

- where the raw data came from,
- how scaffold splits were generated, or
- which seed and split policy were used.

### Model checkpoints / logs / evaluation outputs

- Pretrain checkpoints in `ckpts/pretrain/` or CI stage directories.
- Fine-tune checkpoints in `ckpts/finetune/seed_<seed>/`.
- Benchmark JSON/CSV reports in `reports/`.
- Tox21 summaries, per-task JSON/CSV outputs, prediction files, calibration files, and run manifests in the chosen report directory.
- CI lineage artifacts under `/data/mjepa/experiments/<EXP_ID>/...`.

## How to reproduce the main results

This section uses the repository's real entry points. For remote automation, the corresponding CI wrappers call the same Python commands through `scripts/ci/run-pretrain.sh`, `scripts/ci/run-finetune.sh`, and `scripts/ci/run-tox21.sh`.

### 1. Prepare or place data

#### Unlabeled corpus for pretraining

Provide a flat shard directory, for example:

```text
data/zinc_pretrain/0000.parquet
```

If using the downloader, a minimal example is:

```bash
python scripts/download_unlabeled.py --out-root data/unlabeled --total 1000 --resume
```

Then use `data/unlabeled/train` as `--unlabeled-dir` for pretraining.

#### Labeled dataset for downstream evaluation

The repository already includes `data/tox21/data.csv`. If you use that checked-in file, no extra download step is required for the Tox21 smoke paths below.

### 2. Optional: generate explicit scaffold split files

If you want split files on disk instead of relying on in-memory split generation:

```bash
python scripts/make_scaffold_splits.py \
  --input data/tox21/data.csv \
  --smiles_col smiles \
  --out_dir data/tox21_scaffold \
  --format parquet \
  --train 0.8 \
  --val 0.1 \
  --seed 42
```

This is most directly useful for `benchmark` or for archiving explicit split files. `finetune` and `tox21` can generate splits internally. Unless you separately document their provenance, such generated split files should not be described as the manuscript splits by default.

### 3. What a reviewer can run immediately from this repo

Without downloading additional benchmark assets, a reviewer can immediately use the checked-in `data/` contents for:

- CPU smoke-test pretraining on `data/ZINC-canonicalized/`.
- CPU fine-tuning / evaluation / Tox21 smoke paths on `data/tox21/data.csv`.
- Benchmark-path inspection or CI-style benchmark execution against the checked-in `data/katielinkmoleculenet_benchmark/` split directories.

### 4. Smoke-test path

A minimal reviewer-friendly smoke test is:

```bash
python scripts/train_jepa.py pretrain \
  --unlabeled-dir data/ZINC-canonicalized \
  --output encoder.pt \
  --ckpt-dir ckpts/pretrain_smoke \
  --epochs 1 \
  --sample-unlabeled 128 \
  --batch-size 32 \
  --device cpu
```

Then run a small downstream evaluation:

```bash
python scripts/train_jepa.py evaluate \
  --labeled-dir data/tox21 \
  --encoder ckpts/pretrain_smoke/encoder.pt \
  --label-col NR-AR \
  --task-type classification \
  --epochs 1 \
  --batch-size 32 \
  --device cpu
```

### 5. Full pretraining

```bash
python scripts/train_jepa.py pretrain \
  --unlabeled-dir data/ZINC-canonicalized \
  --output encoder.pt \
  --ckpt-dir ckpts/pretrain \
  --cache-dir cache/graphs_10m \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-4 \
  --mask-ratio 0.15 \
  --device cuda
```

Notes:

- `--output` defines the final checkpoint path; the command also maintains `ckpts/pretrain/encoder.pt` as a stable link for downstream use.
- If `--cache-dir` is set, graph featurizations are reused across runs.
- CI/remote runs typically set `ARTIFACTS_DIR` so a manifest is also written to `<experiment>/artifacts/encoder_manifest.json`.

### 6. Fine-tuning / evaluation

Single-task example using a labeled CSV directory:

```bash
python scripts/train_jepa.py finetune \
  --labeled-dir data/tox21 \
  --labeled-csv data/tox21/data.csv \
  --label-col NR-AR \
  --encoder ckpts/pretrain/encoder.pt \
  --ckpt-dir ckpts/finetune_tox21_nr_ar \
  --task-type classification \
  --use-scaffold \
  --epochs 50 \
  --batch-size 128 \
  --device cuda
```

Evaluation-only alias:

```bash
python scripts/train_jepa.py evaluate \
  --labeled-dir data/tox21/data.csv \
  --label-col NR-AR \
  --encoder ckpts/pretrain/encoder.pt \
  --task-type classification \
  --epochs 50 \
  --batch-size 256 \
  --device cuda
```

### 7. Benchmarking

The repository already includes one benchmark-ready split fixture under `data/katielinkmoleculenet_benchmark/`. That bundled fixture is configured throughout CI/sweeps as the ESOL regression dataset, so use its actual label column and task type when reviewing the benchmark path:

```bash
python scripts/train_jepa.py benchmark \
  --labeled-dir data/katielinkmoleculenet_benchmark/train \
  --jepa-encoder ckpts/pretrain/encoder.pt \
  --task-type regression \
  --label-col "measured log solubility in mols per litre" \
  --report-dir reports \
  --device cuda
```

If sibling `val/` and `test/` directories exist next to the supplied `train/` directory, `benchmark` discovers them automatically. The checked-in `data/katielinkmoleculenet_benchmark/` folder is arranged exactly in that fixture style, so the bundled `train/` path resolves the matching ESOL `val/` and `test/` shards automatically.

### 8. Tox21 case study

```bash
python scripts/train_jepa.py tox21 \
  --csv data/tox21/data.csv \
  --tasks NR-AR NR-AhR SR-p53 \
  --encoder-checkpoint ckpts/pretrain/encoder.pt \
  --evaluation-mode hybrid \
  --report-dir reports/tox21 \
  --device cuda
```

For full automation, the equivalent remote path is the `ci-vast` workflow, which prepares the environment and runs the stage wrappers (`run-pretrain.sh`, `run-finetune.sh`, `run-tox21.sh`).

## CPU vs GPU notes

- Device selection is controlled with `--device` and, for distributed runs, `--devices`.
- On **CPU**, use smaller `--batch-size` values and expect smoke-test scale only for pretraining.
- On **single GPU**, the normal CLI is sufficient.
- On **multi-GPU**, benchmark and Tox21 only use multi-GPU execution when launched under DDP (`torchrun` or the CI stage wrappers). Passing `--devices > 1` without a distributed launcher falls back to a single-device run for benchmark, and Tox21 may relaunch itself with `torchrun` when appropriate.
- If GPU memory is limited, reduce `--batch-size` first; this is the least invasive change.

## Caching behavior

### What is cached

- `--cache-dir` enables on-disk graph caches derived from raw SMILES rows.
- Cache warming helpers also build hashed dataset-level caches under `prebuilt_datasets/`.

### Cache reuse

- These caches are intended to be reused across repeated runs on the same underlying data.
- Cache identity includes the absolute source file path and the label column for file-based caches.

### Cache invalidation

- The cache filename schema explicitly depends on whether `--add-3d` is enabled.
- The dataset loader also validates cached schema metadata, including node and edge dimensions and cache/schema versions.
- In practice, if you change featurization-related settings such as `--add-3d`, or if a schema/version mismatch is reported, clear or rebuild the corresponding cache directory.

### Do caches include dataset splits?

- **No** for the standard graph caches used by dataset loading: they cache featurized graphs, labels, and schema metadata, not train/val/test split indices.
- The split policy is determined separately by explicit split files, `finetune` runtime split generation, or the Tox21 case-study logic.

## Reproducing dataset splits

### Explicit split generation

Use:

```bash
python scripts/make_scaffold_splits.py \
  --input data/tox21/data.csv \
  --out_dir data/tox21_scaffold \
  --format parquet \
  --seed 42
```

Defaults are `train=0.8`, `val=0.1`, and therefore `test=0.1`.

This command writes a **new split artifact** to disk. It does not modify the repository-bundled benchmark fixture directories, and it is separate from the runtime split logic used by `finetune` / `tox21`.

### Implicit split generation during `finetune`

- `finetune` computes splits separately for each seed.
- When `--use-scaffold` is active and SMILES are available, it calls scaffold splitting with that seed.
- If scaffold splitting is unavailable, classification tasks fall back to stratified random splitting.
- Tox21 fine-tuning auto-enables scaffold splitting unless the user explicitly disables or overrides that behavior.

### Implicit split generation during `tox21`

- The Tox21 case study prefers scaffold splitting when RDKit is available.
- It retries up to five seeds (`seed`, `seed+1`, ... ) to satisfy validation/test diversity and positive-count checks.
- If scaffold splitting cannot satisfy those constraints, it falls back to stratified splitting, again with retries.
- The chosen split strategy and split counts are written into the Tox21 diagnostics and manifests in the report directory.

## Split Definitions and Reproducibility

This section is intended to make the repository's split behavior explicit for reviewers.

- **Pre-existing split folders (`train/`, `val/`, `test/`)**
  - In this public repository, the clearest example is `data/katielinkmoleculenet_benchmark/train|val|test`.
  - These folders are used as benchmark-ready fixtures or archival split artifacts that `benchmark` can consume directly.
  - They should not automatically be interpreted as the exact manuscript splits unless a provenance record is added for that specific fixture.
- **Runtime scaffold splits**
  - `finetune` and `tox21` can generate splits at execution time from a single labeled dataset source such as `data/tox21/data.csv`.
  - When scaffold splitting is enabled and SMILES are available, the split is derived inside the command using the active seed; if scaffold splitting is unavailable, the code may fall back to stratified/random alternatives as documented above.
  - `scripts/make_scaffold_splits.py` can also materialize a new split artifact on disk for archival or benchmark use, but that output is distinct from the checked-in benchmark folders.
- **Seeds**
  - The manuscript's Phase-1 seed set is `{1,2,3,4,5}`.
  - The repository's automated workflow also exposes `PHASE1_SEEDS: 1,2,3,4,5` in `.github/workflows/ci-vast.yml`, which is the strongest in-repo evidence for that Phase-1 seed policy.
  - Unless otherwise stated, experiments follow the seed configurations described in Table 1 (seeds `{1,2,3,4,5}` for Phase-1). Downstream runs use multiple seeds for averaging where applicable.
- **What is not claimed here**
  - This repository does **not** currently prove that the exact manuscript split identities are stored in checked-in files.
  - The checked-in `train|val|test` directories are therefore documented conservatively as reusable benchmark fixtures, while runtime split generation is documented as execution-time behavior.

## Outputs and verification

### Expected outputs

#### Pretraining

Look for:

- `ckpts/pretrain/pt_epoch_<N>.pt`
- `ckpts/pretrain/encoder.pt`
- `ckpts/pretrain/pretrain_losses.csv`
- `ckpts/pretrain/plots/pretrain_loss.png`
- `<artifacts_dir>/encoder_manifest.json` when `ARTIFACTS_DIR` or CI experiment roots are used

#### Fine-tuning

Look for:

- `ckpts/finetune/.../seed_<seed>/ft_best.pt`
- `ckpts/finetune/.../head.pt`
- optionally `encoder_ft.pt` / `encoder_ft_<alias>.pt` when encoder weights are exported after fine-tuning

#### Benchmark

Look for:

- `reports/benchmark_<timestamp>.json`
- `reports/benchmark_<timestamp>.csv`

#### Tox21

Look for:

- `tox21_summary.json`
- `tox21_<mode>_metrics.csv`
- per-task JSON/CSV summaries
- `*_scores.csv` and, when multiple seeds are used, `*_scores_by_seed.csv`
- `run_manifest.json` plus per-task `run_manifest_<task>.json`

### Quick verification checklist

- Pretrain completed without the "No unlabeled graphs found" error.
- An encoder checkpoint exists and is readable.
- If running under CI conventions, `encoder_manifest.json` exists under the experiment `artifacts/` directory.
- Downstream runs wrote stage-specific checkpoints or reports in the expected directory.
- Tox21 outputs include both aggregate summary files and per-task files.

## Troubleshooting

### `No unlabeled graphs found ... Check that --unlabeled-dir points to the ZINC dataset, not the cache`

Cause: `--unlabeled-dir` points to a cache root or a directory without flat shard files.

Fix:

- point `--unlabeled-dir` at the actual shard directory, and
- do not point it at `cache/graphs_*`.

### Stale or incompatible cache after changing featurization

Cause: cache built with a different schema, especially after switching `--add-3d` or other feature assumptions.

Fix:

- remove the relevant cache subtree under `--cache-dir`, then rerun.

### CPU-only execution is too slow or runs out of memory

Fix:

- use `--device cpu`,
- reduce `--batch-size`, and
- start with `--sample-unlabeled` for smoke tests instead of full pretraining.

### RDKit missing

Symptoms:

- scaffold splitting unavailable,
- featurization failures,
- fallback behavior in Tox21 split generation.

Fix:

- use Python 3.10 with `requirements.txt`, or
- install RDKit via micromamba/conda-forge as done in the CI environment.

### PyTorch Geometric / torch-scatter missing

Fix:

- install `torch`, `torch-scatter`, and `torch-geometric` with versions compatible with the PyTorch build, using the commands shown in **Minimal setup**.

### Parquet read/write support missing

Fix:

- ensure either `pyarrow` or `fastparquet` is installed; both are already listed in `requirements.txt`.

### Path mismatches between local runs and CI runs

Cause: CI writes to `/data/mjepa/experiments/<EXP_ID>/...`, while local runs often write to `ckpts/` and `reports/`.

Fix:

- compare artifacts by role, not absolute path.
- For remote reruns that reuse a pretrained lineage, set `PRETRAIN_EXP_ID` through the CI wrapper/workflow rather than editing paths by hand.

## Reproducibility checklist

- [ ] Use Python 3.10 or a compatible RDKit-supported environment.
- [ ] Install `requirements.txt` plus the matching PyTorch / PyG / torch-scatter packages.
- [ ] Keep raw datasets, generated split files, and caches in separate directories.
- [ ] Point `--unlabeled-dir` to a real shard directory, not a cache root.
- [ ] Record whether downstream splits were generated explicitly or implicitly.
- [ ] Record the seed(s) used for `finetune` / `evaluate` / `tox21`.
- [ ] Clear/rebuild caches when featurization settings change.
- [ ] Preserve the pretrained encoder checkpoint and, when available, `encoder_manifest.json`.
- [ ] Verify that downstream reports/checkpoints were written to the expected output directory.
