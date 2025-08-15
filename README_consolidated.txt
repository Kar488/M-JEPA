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

2. **Authenticate with Weights & Biases **
   ```bash
   wandb login
   ```
   or set `WANDB_API_KEY` in the environment (e.g., `export WANDB_API_KEY=...` $env:WANDB_API_KEY = "your_api_key_here").
   This variable is used by `main.py`, plotting helpers, and the tests to
   establish a connection to W&B. Logging is disabled by default when the
   variable is absent.

   In Repository → Settings → Secrets and variables → Actions → New repository secret add:

   WANDB_API_KEY = your W&B key

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

    ## Links for parquet files downloaded that are manual and placed in data folder than running the scripts
    
    https://huggingface.co/datasets/BASF-AI/PubChem-Raw - 2.09M
    https://huggingface.co/datasets/sagawa/ZINC-canonicalized - 20.7M
    https://huggingface.co/datasets/HUBioDataLab/tox21/resolve/main/data.csv - 7.83K

    ## Notes

    - Example grid searches and plotting utilities are provided under `experiments/`
      and `analysis/`.
    - Baseline self‑supervised methods are included as git submodules inside
      `third_party/`; run `git submodule update --init --recursive` after cloning if
      you need them.
    - The repository includes sample CSVs (e.g. `samples/tox21_mini.csv`) for quick
      smoke tests.
      - Scripts repo does not have a _init_.py as so its not treated as a package or module. note while deploying


4. **Run tests**
   ```bash
   pytest --cache-clear tests -v -q -s -o log_cli=true -W ignore
   # or single one
   pytest --cache-clear tests/test_plot_small.py -q -s -o log_cli=true -W ignore
   ```


5. ** Running on a server (GitHub Actions ➜ Vast.ai via SSH) **

  1) Provision a vast instance (pick right container and volume sizes)

  2) Set these up in GIT Hub rep - secrets, copy the values available from Vast UI - In Repository → Settings → Secrets and variables → Actions → New repository secret add:

  VAST_HOST = your Vast IP 

  VAST_PORT = your Vast SSH port

  VAST_USER = root (or the user Vast provided)

  3) Create Keys to connect to Vash on SSH

      a) On local windows powershell do this - 
      
      ssh-keygen -t ed25519 -C "vast-deploy" -f $env:USERPROFILE\.ssh\vast_deploy

      # creates: id_vast_ci (private) and id_vast_ci.pub (public)
      #Private key: ~/.ssh/vast_deploy (keep secret)
      #Public key: ~/.ssh/vast_deploy.pub (safe to share)

      b) on computer in powershell type this to ket the key
      Get-Content $env:USERPROFILE\.ssh\vast_deploy.pub

      c) Copy this public key to Vast browser UI - Keys tab

      d) Load the private key into your SSH agent In an elevated PowerShell session

        i. Set-Service -Name ssh-agent -StartupType Automatic
        ii. Start-Service ssh-agent
        iii. ssh-add $env:USERPROFILE\.ssh\vast_deploy
        iv. Get-Content $env:USERPROFILE\.ssh\vast_deploy.pub

      e) Launch Jupytr VM from Vast UI and Authorise the key on the running VM

        i. sudo mkdir -p /root/.ssh
        ii. sudo bash -c 'echo "ssh-ed25519 AAAA... your_comment" >> /root/.ssh/authorized_keys'
        iii. sudo chmod 600 /root/.ssh/authorized_keys 
        iv - sudo cat /root/.ssh/authorized_keys #verify the key added is there

      f) once done you SSH from local machine to login to Vast instance

      ssh -o IdentitiesOnly=yes `
        -i $env:USERPROFILE\.ssh\vast_deploy `
        -p 17259 `
        root@144.6.107.170 -vv

      g) set up private key on GIT server repository secrets so it can connect to Vast

      VAST_SSH_KEY = contents of id_vast_ci (the private key - use pw cmd Get-Content $env:USERPROFILE\.ssh\vast_deploy)

      h) Setup git to be able to deploy code - GH_PAT_RO

      GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens - Gen new token & content/metata read for repo M-JEPA/90 day expiry -Nov11)

      Copy that GH_PAT_RO key value into Git GIT server repository secrets

  4) Configure git hub runner on vast so it can see if Vast instance is running

    a) Open your repository on GitHub (for example: https://github.com/Kar488/M‑JEPA).
    b) Click Settings at the top of the repo page.
    c) In the left sidebar, select Actions → Runners.
    d) Click New self‑hosted runner (or Add runner).
    e) Choose Linux for the operating system and x64 for the architecture.
    f) GitHub will display a set of commands under download (e.g., below)

        # Make a directory and download the runner package
        i. adduser --disabled-password --gecos "" github # overcome MUST not run on sudo for step XX
        ii. su - github # Switch to that user
        iii. mkdir ~/actions-runner && cd ~/actions-runner # # Create a folder
        iv. curl -o actions-runner-linux-x64-2.327.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.327.1/actions-runner-linux-x64-2.327.1.tar.gz # Download the latest runner package
        v. echo "d68ac1f500b747d1271d9e52661c408d56cffd226974f68b7dc813e30b9e0575  actions-runner-linux-x64-2.327.1.tar.gz" | shasum -a 256 -c # Optional: Validate the hash
        vi. tar xzf ./actions-runner-linux-x64-2.327.1.tar.gz # Extract the installer
        vii. ./config.sh --url https://github.com/Kar488/M-JEPA --token BJEZIAQHZ4ZBGFLV6RE5TG3IT4H3W --name vast-runner --labels vast
        viii. ./run.sh # Start the runner

        (Optional) pkill -f "Runner.Listener" || true   # clears any stuck runner process
        (Optional) tmux ls || true # check if its running already
        (Optional) tmux attach || true # attach to one running already
        
        Now Git hub action runner will show this as running and idle
        viiii. mkdir ~/actions-runner && cd ~/actions-runner # Recreate the actions‑runner directory in this user’s home and rerun the steps







This setup uses a GitHub-hosted runner that SSHes into your Vast instance, uploads the repo, and runs commands there. No self-hosted runner needed.
**Provision a Vast.ai instance** with a GPU. Note the instance ID and the IP/SSH credentials.


### 2) SSH to your Vast box (use the IP/port Vast shows you), then add the public key:

You can do it via Vast Browser UI in keys tab and copy the value from ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIwg66siIs/pTWsS29Gu/R5bf7sSzEgSX5pmRnXF4hhH vast-deploy

OR

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



### Setting keys in vast ai

In powershell

1) New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
2) ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_vast_ci" -C "vast-ci"
set passphrase
3) Get-ChildItem "$env:USERPROFILE\.ssh\id_vast_ci*"
4) Get-Content "$env:USERPROFILE\.ssh\id_vast_ci.pub" -Raw |
ssh -p 17259 root@144.6.107.170 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
5) ssh -p 17259 root@144.6.107.170 'chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys'
6) in git secrets add

VAST_SSH_PASSPHRASE = your passphrase
VAST_HOST=144.6.107.170 (or the Vast hostname)
VAST_PORT=17259
VAST_USER=root
VAST_SSH_KEY = contents of C:\Users\karth\.ssh\id_vast_ci

steps 2 and 4 are required for every new computer we use to SSH to vast


