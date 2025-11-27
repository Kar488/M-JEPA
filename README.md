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
      Grid search enables caching by default under `cache/graphs_250k` unless `--no-cache`
      is given. Clear the cache when switching featurisation options such as
      `--add-3d` to avoid stale representations. The enlarged 250 K graph dataset keeps
      its cached featurisations under `cache/graphs_250k` to avoid clashes with other
      datasets. Remove any lingering `cache/graphs_50k` directories so runs don't
      accidentally mix the old 50 K features with the new cache.

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
      - You can now dial in class imbalance and encoder updates explicitly:
        * `--pos-class-weight` accepts either a float or per-task `TASK=weight` override to up-weight rare positives.
        * `--freeze-encoder/--no-freeze-encoder` toggles whether the backbone remains frozen even in fine-tuned mode.
        * `--head-ensemble-size` trains several lightweight heads and averages their predictions for a quick ensemble boost.
        * Pair these with `--no-calibrate` to benchmark calibrated vs. raw ROC-AUC without editing the code.
      - CI stages share a canonical pretrain experiment id recorded in ``/data/mjepa/experiments/pretrain_state.json``.
        Downstream phases (fine‑tune, tox21, reporting) read that file to discover the encoder checkpoint, manifest, and
        ``tox21_gate.env`` so artifact collection does not rely on timestamp guesses.
      - Optionally cache refresh for Grid search can be controlled through Git actions drop down for rerun flow


4. **Run tests**

   ```bash
   cd 'C:\Users\karth\Dropbox\Documents\synched folder\my.certifications\La trobe\research\coding\M-JEPA>'
   pytest --cache-clear tests -v -q -s -o log_cli=true -W ignore
   # or single one
   pytest --cache-clear tests/test_plot_small.py -q -s -o log_cli=true -W ignore
   ```


5. **Running on a server (GitHub Actions ➜ Vast.ai via SSH)**

  1) Provision a vast instance (pick right container and volume sizes)

  2) Set these up in GIT Hub rep - secrets, copy the values available from Vast UI - In Repository → Settings → Secrets and variables → Actions → New repository secret add:

      VAST_HOST = your Vast IP 

      VAST_PORT = your Vast SSH port

      VAST_USER = root (or the user Vast provided)

  3) Create Keys to connect to Vash on SSH

      a) On local windows powershell do this - 
      
      use -N for no pass phrase
      Create the .ssh directory if it doesn't already exist
      
      ```bash
      New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh"
      ```
      Now generate the keypair in that folder - change "vast-deploy" to location specific as needed (e.g., office / home)

      ```bash
      ssh-keygen -t ed25519 -C "vast-deploy" -f $env:USERPROFILE\.ssh\vast_deploy
      ```
      creates: a public and private key
      Private key: ~/.ssh/vast_deploy (keep secret)
      Public key: ~/.ssh/vast_deploy.pub (safe to share)

      b) on computer in powershell type this to ket the key

      ```bash
      Get-Content $env:USERPROFILE\.ssh\vast_deploy.pub
      ```

      c) Copy this public key to Vast browser UI - Keys tab

      d) Load the private key into your SSH agent In an elevated PowerShell session

        ```bash
        Set-Service -Name ssh-agent -StartupType Automatic
        Start-Service ssh-agent
        ssh-add $env:USERPROFILE\.ssh\vast_deploy
        Get-Content $env:USERPROFILE\.ssh\vast_deploy.pub
        ```

      e) Launch Jupytr VM from Vast UI and Authorise the key on the running VM

        ```bash
        sudo mkdir -p /root/.ssh
        sudo bash -c 'echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC5ITBFL64GL5641TNGToOus7HaX5p04tRDZNoQQOLVi vast-deploy_office" >> /root/.ssh/authorized_keys'
        sudo chmod 600 /root/.ssh/authorized_keys 
        sudo cat /root/.ssh/authorized_keys #verify the key added is there
        ```

        OR overwrite keys in vast console - do this 
        ```bash
        nano /root/.ssh/authorized_keys # edit as needed / CTROL+O to save (enter) / CTRL+Q (quit)
        sudo cat /root/.ssh/authorized_keys # see the list of keys
        ```

      f) once done you SSH from local machine to login to Vast instance - change vast_deploy with location (office/home) as needed

      ```bash
      ssh -o IdentitiesOnly=yes -i $env:USERPROFILE\.ssh\vast_deploy_office -p 56982 root@38.58.100.42 -vv
      ```

      g) set up private key on GIT server repository secrets so it can connect to Vast - change vast_deploy with location (office/home) as needed

      VAST_SSH_KEY = contents of id_vast_ci (the private key - use pw cmd Get-Content $env:USERPROFILE\.ssh\vast_deploy)

      h) Setup git to be able to deploy code - GH_PAT_RO

      GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens - Gen new token & content/metata read for repo M-JEPA/90 day expiry)

      Copy that GH_PAT_RO key value into Git GIT server repository secrets

      i) add deploy key in GitHub - Repo - Deploy keys - VAST_DEPLOY

      Copy the output from Vast server - ssh-keygen -y -f ~/.ssh/mjepa_deploy_key | tee /root/.ssh/mjepa_deploy_key.pub
      and then add it to the new deploy key 

  4) Configure git hub runner on vast so it can see if Vast instance is running

    a) Open your repository on GitHub (for example: https://github.com/Kar488/M‑JEPA).
    b) Click Settings at the top of the repo page.
    c) In the left sidebar, select Actions → Runners.
    d) Click New self‑hosted runner (or Add runner).
    e) Choose Linux for the operating system and x64 for the architecture.
    f) GitHub will display a set of commands under download (e.g., below)

        # Make a directory and download the runner package

        ```bash
        adduser --disabled-password --gecos "" github # overcome MUST not run on sudo for step XX
        su - github # Switch to that user
        mkdir ~/actions-runner && cd ~/actions-runner # # Create a folder
        curl -o actions-runner-linux-x64-2.328.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.328.0/actions-runner-linux-x64-2.328.0.tar.gz # Download the latest runner package
        echo "01066fad3a2893e63e6ca880ae3a1fad5bf9329d60e77ee15f2b97c148c3cd4e  actions-runner-linux-x64-2.328.0.tar.gz" | shasum -a 256 -c # Optional: Validate the hash
        tar xzf ./actions-runner-linux-x64-2.328.0.tar.gz # Extract the installer
        ./config.sh --url https://github.com/Kar488/M-JEPA --token BJEZIARE7I3CQRCLKRWMIETI2CEDQ
        ./run.sh # Start the runner
        ```

        # Use this YAML in your workflow file for each job
        runs-on: self-hosted

        # Optional

        ```bash
        pkill -f "Runner.Listener" || true   # clears any stuck runner process
        tmux ls || true # check if its running already
        tmux attach || true # attach to one running already
        #Now Git hub action runner will show this as running and idle
        mkdir ~/actions-runner && cd ~/actions-runner # Recreate the actions‑runner directory in this user’s home and rerun the steps

        git config --global --add safe.directory /srv/mjepa # listed reposistory as allowed in vast
        ```
    g) setup git pro account for increased action minutes and set budget limit to some $ and disallow block on expiry
    h) give the public key of vast to git hub

        # Create ~/.ssh if it doesn’t exist
        ```bash
        mkdir -p ~/.ssh && chmod 700 ~/.ssh
        ```
        # Generate a key pair without a passphrase
        ```bash
        ssh-keygen -t ed25519 -f ~/.ssh/mjepa_deploy_key -N "" -C "deploy key for M-JEPA"
        ```
        #diplay its content
        ```bash
        cat ~/.ssh/mjepa_deploy_key.pub
        ```
        
        # Go to your repository on GitHub and click Settings.

        # In the sidebar, click Deploy keys 
        # Click Add deploy key
        # Provide a descriptive Title (e.g., “Vast deployment server”).
        # Paste the contents of ~/.ssh/mjepa_deploy_key.pub into the Key field
        # If the server needs to push changes (not just clone/pull), check Allow write access
        # Click Add key - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMSsRfYnsN29PnL3zXF/MM1StM407i1dTT8X1ETWQh2o deploy key for M-JEPA

        # To avoid interactive prompts when connecting via SSH, add GitHub’s host key to known_hosts:
        ```bash
        ssh-keyscan github.com >> ~/.ssh/known_hosts
        ```
  4.1) if deploy fails due to lack of ownership on experiments folder, do this

        # sanity check current ownership
        ls -ld /data /data/mjepa /data/mjepa/experiments

        # now fix it
        chown -R github:github /data/mjepa
        chmod -R 0775 /data/mjepa

        # verify
        ls -ld /data/mjepa /data/mjepa/experiments
        # expect something like:
        # drwxrwxr-x 7 github github ...
        # drwxrwxr-x 5 github github .

        # Give ownership to the GitHub Actions runner user
        chown -R github:github /srv/mjepa

        # Give group/owner write permission
        chmod -R 0775 /srv/mjepa

        ls -ld /srv/mjepa /srv/mjepa/logs
        # Should show something like:
        # drwxrwxr-x github github ...
        # drwxrwxr-x github github ...

        #full details
        stat -c '%U %G %a %n' /srv/mjepa /srv/mjepa/logs

        #instal CUDA key ring
        sudo -i   # become root

        # Download the keyring deb directly to /tmp
        cd /tmp
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

        # Install it once, system-wide
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update

  5) After 1st deployment need to ensure large parquest files are pulled down properly to avoid - Parquet magic bytes not found in footer. Either the file is corrupted or this is not a parquet file.

    From Vast Jupytr notebook terminal 

        ```bash
        run cd ~/srv
        git lfs install # sets up Git LFS hooks if needed 
        git lfs pull # downloads all large files tracked by LFS
        top # to see what proces is running
        kill 3358              # sends SIGTERM — lets the program clean up
        # if it doesn’t stop within a few seconds, force it:
        kill -9 3358           # sends SIGKILL — immediate terminatio
        ```
  6) Ensure we are using the larger disk space in vast instance

      Code enables to right directory already but just in case

      ```bash
      # make sure project is at /srv/mjepa (or your repo root)
      cd /srv/mjepa

      # create parent dirs first
      # make sure targets exist and are writable
      sudo mkdir -p /data/mjepa/{cache/graphs_250k,outputs,logs,wandb}
      sudo chown -R "$USER":"$USER" /data/mjepa

      # now re-point them into /data
      rm -rf cache/graphs_250k  # only if it exists already
      ln -s /data/mjepa/cache/graphs_250k cache/graphs_250k

      rm -rf outputs
      ln -s /data/mjepa/outputs outputs

      rm -rf logs
      ln -s /data/mjepa/logs logs

      rm -rf wandb
      ln -s /data/mjepa/wandb wandb

      echo 'export WANDB_DIR=/data/mjepa/wandb' >> ~/.bashrc
      export WANDB_DIR=/data/mjepa/wandb

      # we should see the logs proxy to the large disk folder
      ls -ld cache outputs logs wandb

      # if it fails we run this due to memoty disk full failures because we wrote to root, then do this after pre train hands
      # kill any stuck python/mamba
      pkill -f "[mM]amba|python.*train_jepa.py" || true

      # remove stale lock
      rm -f ~/.cache/mamba/proc/proc.lock 2>/dev/null || true

      # purge any leftover local artifacts
      rm -rf cache 2>/dev/null || true         # we just recreated it as a symlink above
      rm -rf wandb/* outputs/* 2>/dev/null || true
      pip cache purge || true

      # check space
      df -h / /data
      ```

  7) Other tasks - such as monitoring and cache enabling of grid search, pre train and fine tune

      cache grid etc
      set force_refresh to false in ci-vast

      ```bash

      #GPU consumption
      watch -n 1 nvidia-smi
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
           --format=csv,noheader,nounits --loop=1 \
           --filename /data/mjepa/experiments/$RUN_ID/logs/nv_smi.csv

      #thread dump on train jepa
      kill -USR1 <python-pid> 

      # Kill pre train hangs
      pkill -f "scripts/train_jepa.py pretrain" || true
      ```
  
  8) Other todos

  a. Break up train jepa and grid search to be more modular, they suffer from bloat because of procedural writes (evolving needs)
  b. Train subsequent steps with not just ESOL but larger corpus
  c. explore random and Bayesian search over grid search to optimise for timing of runs. Wire into pipeline of custom grid and then handover to bayesian

  ```bash
  export APP_DIR=/srv/mjepa
  wandb sweep --project mjepa grid_sweeps.yaml #or whatever name
  #measure your swwp at https://wandb.ai/karthik-iyer-la-trobe-university/mjepa/sweeps/jnx2hxkj # last bit is whatever the swwp URL is
  # paste wandb key whn asked and then copy the generated url to project code

  d. fix hyper‑parameters for grid we know do not strongly affect performance (e.g., maybe fix hidden_dim=256 and num_layers=3 if we know deeper networks give no benefit) or run cap (300-500)
  e. reporting threshold-free metrics and your finetune already selects best by VAL through a seperate eval step