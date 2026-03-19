# M-JEPA

> **Summary**
> - Frozen encoder lineages are immutable.
> - Dependent runs must set `PRETRAIN_EXP_ID` to reuse them.
> - To rebuild, remove or override the freeze marker explicitly.

Joint Embedding Predictive Architectures (JEPA) for molecular graphs. The
project includes data download scripts, self-supervised pretraining, and
utilities for downstream evaluation on MoleculeNet benchmarks.

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
   pip install -r requirements.txt
   pip install torch
   pip install torch-geometric
   pip install deepchem
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
   - Large corpora such as **ZINC** and **PubChem** can be downloaded with
     `scripts/download_unlabeled.py`. The resulting Parquet shards are stored
     under `data/unlabeled/`.
   -  Labeled benchmarks from **MoleculeNet** (ESOL, FreeSolv, Lipophilicity,
      BACE, BBBP, Tox21, ClinTox, SIDER) should be placed under `data/` as
      scaffold‑split CSV/Parquet files. The repository previously downloaded
      copies of ZINC, PubChem, Tox21 and MoleculeNet; if these folders are
      absent, the code will attempt to fetch them on the fly.

      Links for parquet files downloaded that are manual and placed in data folder than running the scripts
    
      https://huggingface.co/datasets/BASF-AI/PubChem-Raw - 2.09M
      https://huggingface.co/datasets/sagawa/ZINC-canonicalized - 20.7M
      https://huggingface.co/datasets/HUBioDataLab/tox21/resolve/main/data.csv - 7.83K
      https://huggingface.co/datasets/katielink/moleculenet-benchmark/tree/af500889de49a7c64ede443c2928fd5e876dd677/esol - train 1.21K

   -  The test suite uses small synthetic or bundled samples and does **not**
      require any of the large datasets.
   -  Pass `--cache-dir` to `scripts/train_jepa.py` to store featurised graphs on disk.
      Grid search enables caching by default under `cache/graphs_10m` unless `--no-cache`
      is given. Clear the cache when switching featurisation options such as
     `--add-3d` to avoid stale representations. The enlarged 10 M graph dataset keeps
      its cached featurisations under `cache/graphs_10m` to avoid clashes with other
      datasets. Remove any stale cache directories when switching featurisation
      options to prevent accidental reuse of mismatched features.

   -  Pipeline usage
      Individual stages of the JEPA workflow can be invoked via subcommands in
      ``scripts/train_jepa.py``. This allows external deployment pipelines to run
      only the required phase:

      ```bash
      # Self-supervised pretraining
      python scripts/train_jepa.py pretrain --unlabeled-dir data/unlabeled
      
      # Fine‑tune a linear head on labelled data
      python scripts/train_jepa.py finetune --labeled-dir data/labeled --encoder encoder.pt
      
      # Evaluate a pretrained encoder with a fresh probe
      python scripts/train_jepa.py evaluate --labeled-dir data/labeled --encoder encoder.pt
      ```

  -  Notes

      - Example grid searches and plotting utilities are provided under `experiments/`
        and `analysis/`.
      - Baseline self‑supervised methods are included as git submodules inside
        `third_party/`; run `git submodule update --init --recursive` after cloning if
        you need them.
      - The repository includes sample CSVs (e.g. `samples/tox21_mini.csv`) for quick
        smoke tests.
        - Scripts repo does not have a _init_.py as so its not treated as a package or module. note while deploying
      - Grid search results are automatically resused for pretraining action, Once pretraining has finished, that encoder is saved to disk
        (e.g. outputs/encoder.pt) and is loaded directly in the finetune, benchmark and case‑study commands.
        Those later stages don’t rebuild the encoder; they just attach a linear head or evaluate the already‑trained model.
        Because of that, you don’t need to pass the same flags again to finetune, benchmark or tox21. The current workflow does exactly this:
      - The Tox21 case study separates the JEPA pretraining learning rate from the downstream probe. Use
        `--pretrain-lr` to tune pretraining without perturbing the linear head. When running with
        `--evaluation-mode fine_tuned` the command automatically enables full encoder fine-tuning when no
        checkpoint is supplied.
      - `--evaluation-mode hybrid` runs a three-phase schedule (freeze → partial unfreeze → full fine-tune)
        driven by `TOX21_EPOCHS` with phase split defaults (`TOX21_FREEZE_EPOCHS`, `TOX21_UNFREEZE_TOP_LAYERS`).
        Hybrid uses warmup+cosine scheduling via `TOX21_LR_SCHEDULER`/`TOX21_WARMUP_RATIO` and a minimum LR
        floor (`TOX21_MIN_LR`/`TOX21_MIN_LR_RATIO`).
      - Per-task hyperparameters in `scripts/ci/per_task_hparams/tox21_hparams.yaml` are the source of truth
        for encoder/head LRs and other overrides; best-config sweeps should not override those values.
      - `threshold_metric` only affects post-hoc threshold selection and does not change the training loss.
      - `TOX21_CHECKPOINT_METRIC` selects which validation metric drives best-checkpoint selection and early stopping
        inside the Tox21 case study (defaults to `pr_auc`; set `TOX21_CHECKPOINT_METRIC: roc_auc` in `ci-vast.yml`
        to revert to ROC-AUC).
      - CI keeps its larger pretraining sample sizes and streaming chunk knobs in `scripts/ci/train_jepa_ci.yml` so
        stage defaults in `scripts/default.yaml` remain lightweight for local runs; the best-config merger
        treats those CI-owned values as YAML-only so cached grids cannot overwrite them.
      - CI stages now perform defensive cleanup of MJepa `train_jepa.py` / `torchrun` processes at both stage
        start and stage exit (success or failure) to prevent orphaned DDP workers from hanging later stages.
        Torchrun launches are terminated by process group so all ranks exit together. Set
        `MJEPACI_DISABLE_CLEANUP=1` to opt out or `MJEPACI_CLEANUP_DRYRUN=1` to log matches without killing.
      - Fine-tuning defaults favour longer runs on smaller batches (50 epochs, batch size 128, patience 5) and expose
        separate `--encoder-lr` / `--head-lr` knobs so the backbone can adapt as quickly as the probe.
      - Benchmark runs that request multiple GPUs (`--devices > 1`) require a DDP launch via `torchrun`; otherwise the
        benchmark command will warn and fall back to a single-device run.
      - You can now dial in class imbalance and encoder updates explicitly:
        * `--pos-class-weight` accepts either a float or per-task `TASK=weight` override to up-weight rare positives.
        * `--use-focal-loss/--dynamic-pos-weight/--oversample-minority` mitigate skewed datasets; combine them with
          `--layerwise-decay` to slow learning in deeper encoder blocks.
        * Post-hoc calibration is supported via `--calibrate-probabilities` with temperature scaling or isotonic
          regression plus automatic validation-threshold tuning (`--threshold-metric`).
        * `--freeze-encoder/--no-freeze-encoder` toggles whether the backbone remains frozen even in fine-tuned mode.
        * `--head-ensemble-size` trains several lightweight heads and averages their predictions for a quick ensemble boost.
        * Pair these with `--no-calibrate` to benchmark calibrated vs. raw ROC-AUC without editing the code.
      - CI stages share a canonical pretrain experiment id recorded in ``/data/mjepa/experiments/pretrain_state.json``.
        Downstream phases (fine‑tune, tox21, reporting) read that file to discover the encoder checkpoint, manifest, and
        ``tox21_gate.env`` so artifact collection does not rely on timestamp guesses.
      - Optionally cache refresh for Grid search can be controlled through Git actions drop down for rerun flow


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
