# M-JEPA

Joint Embedding Predictive Architectures (JEPA) for molecular graphs. The
project includes data download scripts, self-supervised pretraining, and
utilities for downstream evaluation on MoleculeNet benchmarks.

## Local development

1. **Install dependencies**
   ```bash
   git clone https://github.com/.../M-JEPA.git
   cd M-JEPA
   pip install -r requirements.txt
   ```
   Optional: install RDKit via conda or `micromamba` for full chemistry
   features.

2. **Authenticate with Weights & Biases (optional)**
   ```bash
   wandb login
   ```
   or set `WANDB_API_KEY` in the environment. Logging is disabled by default in
   tests and examples.

3. **Datasets**
   - Large corpora such as **ZINC** and **PubChem** can be downloaded with
     `scripts/download_unlabeled.py`. The resulting Parquet shards are stored
     under `data/unlabeled/`.
   - Labeled benchmarks from **MoleculeNet** (ESOL, FreeSolv, Lipophilicity,
     BACE, BBBP, Tox21, ClinTox, SIDER) should be placed under `data/` as
     scaffold‑split CSV/Parquet files. The repository previously downloaded
     copies of ZINC, PubChem, Tox21 and MoleculeNet; if these folders are
     absent, the code will attempt to fetch them on the fly.
   - The test suite uses small synthetic or bundled samples and does **not**
     require any of the large datasets.

4. **Run tests**
   ```bash
   pytest --cache-clear tests -v -q -s -o log_cli=true
   ```

## Running on a server (GitHub Actions + Vast.ai)

1. **Provision a Vast.ai instance** with a GPU. Note the instance ID and the
   IP/SSH credentials.
2. **Register a self‑hosted GitHub runner** on that instance. Follow the
   instructions under the repository’s *Settings → Actions → Runners*.
3. **Repository secrets**
   - `WANDB_API_KEY`: API key for Weights & Biases logging.
   - `VAST_API_KEY`: key used by any automation scripts interacting with the
     Vast API.
4. The workflow `.github/workflows/train.yml` will install dependencies on the
   runner and launch `scripts/train_jepa.py`. Logs and metrics are sent to
   W&B automatically when the `WANDB_API_KEY` secret is present.

## Notes

- Example grid searches and plotting utilities are provided under `experiments/`
  and `analysis/`.
- Baseline self‑supervised methods are included as git submodules inside
  `third_party/`; run `git submodule update --init --recursive` after cloning if
  you need them.
- The repository includes sample CSVs (e.g. `samples/tox21_mini.csv`) for quick
  smoke tests.