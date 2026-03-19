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
   - Unlabeled corpora for `scripts/train_jepa.py pretrain` must be provided as a
     **flat directory of `.parquet` or `.csv` shards**. Pass that shard directory
     to `--unlabeled-dir`.
   - `scripts/download_unlabeled.py` writes split subdirectories under
     `data/unlabeled/train`, `data/unlabeled/val`, and `data/unlabeled/test`.
     If you use that helper, point `--unlabeled-dir` at a shard directory such as
     `data/unlabeled/train`, **not** at the `data/unlabeled` parent.
   - Labeled datasets for `finetune` / `evaluate` are supplied separately, usually
     as a directory plus an optional CSV such as `data/tox21/data.csv`.
   - Generated scaffold split files are a separate artifact. Use
     `scripts/make_scaffold_splits.py` if you want explicit `train/`, `val/`, and
     `test/` directories on disk; `finetune` and `tox21` can also generate splits
     internally from SMILES.
   - `--cache-dir` stores **featurized graph caches** only. Cached graph artifacts
     speed up repeated runs, but they do **not** replace raw datasets or split
     generation metadata. Clear/rebuild caches when changing featurization such as
     `--add-3d`.
   - The test suite uses small synthetic or bundled samples and does **not**
     require the large training corpora.

   Minimal local examples:

   ```bash
   # Pretraining: note the flat shard directory
   python scripts/train_jepa.py pretrain --unlabeled-dir data/unlabeled/train

   # Fine-tuning/evaluation: note the labeled dataset is separate
   python scripts/train_jepa.py finetune --labeled-dir data/tox21 --labeled-csv data/tox21/data.csv --encoder encoder.pt --label-col NR-AR
   python scripts/train_jepa.py evaluate --labeled-dir data/tox21/data.csv --encoder encoder.pt --label-col NR-AR
   ```

   Keep the README brief and use [`REPRODUCE.md`](REPRODUCE.md) for full setup,
   cache, split, CPU/GPU, and output verification details.

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
