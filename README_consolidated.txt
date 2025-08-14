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
   or set `WANDB_API_KEY` in the environment (e.g., `export WANDB_API_KEY=...` $env:WANDB_API_KEY = "your_api_key_here").
   This variable is used by `main.py`, plotting helpers, and the tests to
   establish a connection to W&B. Logging is disabled by default when the
   variable is absent.

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

  - Pipeline usage
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

4. **Run tests**
   ```bash
   pytest --cache-clear tests -v -q -s -o log_cli=true -W ignore
   # or single one
   pytest --cache-clear tests/test_plot_small.py -q -s -o log_cli=true -W ignore
   ```


## Running on a server (GitHub Actions ➜ Vast.ai via SSH)


This setup uses a GitHub-hosted runner that SSHes into your Vast instance, uploads the repo, and runs commands there. No self-hosted runner needed.
**Provision a Vast.ai instance** with a GPU. Note the instance ID and the IP/SSH credentials.

### 1) Create an SSH key for CI (on your laptop)
```bash
ssh-keygen -t ed25519 -C "vast-deploy" -f $env:USERPROFILE\.ssh\vast_deploy
# creates: id_vast_ci (private) and id_vast_ci.pub (public)
#Private key: ~/.ssh/vast_deploy (keep secret)
#Public key: ~/.ssh/vast_deploy.pub (safe to share)

### 2) SSH to your Vast box (use the IP/port Vast shows you), then add the public key:

ssh -p <VAST_SSH_PORT> root@<VAST_IP> (from from vast UI)
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

#on computer
Get-Content $env:USERPROFILE\.ssh\vast_deploy.pub

#back to vast to use the public key
cat > ~/vast_deploy.pub
#Now paste that single-line public key into the terminal.
# Press Enter once if needed so the cursor moves to a new line.
# Press Ctrl+D to finish the file.
cat ~/vast_deploy.pub >> ~/.ssh/authorized_keys
rm -f ~/vast_deploy.pub
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chown -R "$(whoami)":"$(whoami)" ~/.ssh
tail -n1 ~/.ssh/authorized_keys
# You should see the key line you just added (ssh-ed25519 ... vast-deploy)
sudo grep -E '^(Port|PubkeyAuthentication|PasswordAuthentication)' /etc/ssh/sshd_config
# Expect at least: PubkeyAuthentication yes
# Note the Port value — this must match secrets.VAST_PORT (22 if default)

### 3)  Add GitHub secrets
In Repository → Settings → Secrets and variables → Actions → New repository secret add:

VAST_HOST = your Vast IP

VAST_PORT = your Vast SSH port (often a high, non-22 port)

VAST_USER = root (or the user Vast provided)

VAST_SSH_KEY = contents of id_vast_ci (the private key - use pw cmd Get-Content $env:USERPROFILE\.ssh\vast_deploy)

WANDB_API_KEY = your W&B key (optional, for logging)

GH_PAT_RO = 
PAT token (GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens - Gen new token & content/metata read for repo M-JEPA/90 day expiry -Nov11)
And then set in repo secrets

### 4) Add a deploy workflow

The workflow `.github/workflows/ci-vast.yml` will install dependencies on the
runner and launch `scripts/train_jepa.py`. Logs and metrics are sent to
W&B automatically when the `WANDB_API_KEY` secret is present.

### Start vast ai

1) Login into vas ai
2) Open terminal from instances
3) Launch with the provided SSH - e.g. ssh -p 40129 root@167.179.138.57 -L 8080:localhost:8080
ssh -o IdentitiesOnly=yes -o PasswordAuthentication=no `
    -i $env:USERPROFILE\.ssh\vast_deploy `
    -p 40129 `
    root@167.179.138.57 -vv

4) Use password when generating key - e.g., crypt
5) Start runner -
   5.1) su - runner
   5.2) cd ~/actions-runner
   5.3) pkill -f "Runner.Listener" || true   # clears any stuck runner process
   5.4) ./run.sh
   5.5) Verify on git hub actions - Repo → Settings → Actions → Runners → it should show Online.
   or
   5.6) tmux ls || true # check if its running already
   5.7) tmux attach || true

### Setting keys in vast ai

In powershell

1) New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
2) ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_vast_ci" -C "vast-ci"
set passphrase
3) Get-ChildItem "$env:USERPROFILE\.ssh\id_vast_ci*"
4) Get-Content "$env:USERPROFILE\.ssh\id_vast_ci.pub" -Raw |
ssh -p 40129 root@167.179.138.57 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
5) ssh -p 40129 root@167.179.138.57 'chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys'
6) in git secrets add

VAST_SSH_PASSPHRASE = your passphrase
VAST_HOST=167.179.138.57 (or the Vast hostname)
VAST_PORT=40129
VAST_USER=root
VAST_SSH_KEY = contents of C:\Users\karth\.ssh\id_vast_ci

steps 2 and 4 are required for every new computer we use to SSH to vast

## Notes

- Example grid searches and plotting utilities are provided under `experiments/`
  and `analysis/`.
- Baseline self‑supervised methods are included as git submodules inside
  `third_party/`; run `git submodule update --init --recursive` after cloning if
  you need them.
- The repository includes sample CSVs (e.g. `samples/tox21_mini.csv`) for quick
  smoke tests.
  - Scripts repo does not have a _init_.py as so its not treated as a package or module. note while deploying