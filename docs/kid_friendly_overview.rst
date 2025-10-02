M-JEPA: A Kid-Friendly Guide
============================

Welcome to the M-JEPA project! Think of this repository as a giant science
lab where computers learn to understand molecules.  Each folder is like a
different lab room with its own tools. Here's a tour that explains
everything in simple, kid-friendly language.

Main Entrance: ``main.py``
----------------------------------------
This is the control center of the whole lab. It lets you:

- Run tiny experiments just to see if things work (``demo`` mode).
- Train a full model with real data (``full`` mode).
- Try lots of combinations of settings to find the best one (``grid`` mode).

You choose what to run using command-line switches like ``--mode``,
``--device``, or ``--method``.

Data Lab: ``data/``
----------------------------------------
This room creates and handles the molecules.

``dataset.py``
  Defines a ``GraphData`` class (a molecule as a graph) and a
  ``GraphDataset`` class (a collection of them).  Can turn SMILES strings
  (a text way to describe molecules) into usable graph data for the models.
  Offers mini-batch helpers so multiple molecules can be processed together.

``augment.py``
  Adds a bit of playful randomness to molecules:

  - Spin them (``random_rotation``).
  - Freeze one of their angles (``mask_random_angle``).
  - Twist them slightly (``perturb_dihedral``).

Other helpers
  Files like ``parquet_loader.py`` and ``scaffold_split.py`` help load
  real datasets.  When 3D coordinates are needed (for example, with
  SchNet3D models), ``rdkit_geometry.py`` provides utilities to embed
  molecules in 3D space so that node positions (``g.pos``) are available.

Model Factory: ``models/``
----------------------------------------
These are different "brains" that understand molecular graphs.

``base.py``
  Defines abstract classes so all models share the same structure.

``encoder.py``
  A simple Graph Neural Network (GNN) that reads a molecule and makes an
  embedding (a fancy word for a vector of numbers that describes it).

``edge_encoder.py``
  An upgraded GNN that also looks at information stored on the bonds (the
  edges).

``gnn_variants.py``
  Several popular GNN flavors like GraphSAGE, GIN, a multi-head GAT,
  the Directed Message Passing Neural Network (DMPNN), and SchNet3D.
  Some variants (such as SchNet3D) require additional information like
  3D coordinates; see the data helpers for details.

``predictor.py``
  A small neural network that tries to guess a target embedding from the
  context embedding.

``ema.py``
  Implements Exponential Moving Average (EMA) to keep a smooth version
  of the model's weights, which helps with stability.

``factory.py``
  A factory that builds models based on the configuration settings. It
  can create different types of encoders and predictors.

Training Room: ``training/``
----------------------------------------
This is where models learn.

``unsupervised.py``
  Trains either the JEPA model or a contrastive baseline without
  labels.  It can run in distributed settings and supports mixed
  precision (bf16).  The module implements graph masking and
  augmentation to teach models to predict missing parts, and it
  supports multiple backbones (e.g. GINE, DMPNN, SchNet3D).

``supervised.py`` / ``supervised_with_val.py``
  Train simple "heads" on top of embeddings when labels are available.
  Optional validation for early stopping.

``train_on_embeddings.py``
  Uses pre-computed embeddings with simple models (like logistic
  regression) for quick evaluation.

``pretrain.py`` & ``baselines.py``
  Helpers to run baseline methods and pretraining flows.
  In particular, ``baselines.py`` wraps third‑party models so that
  contrastive and JEPA methods can be compared fairly.

Toolbox: ``utils/``
----------------------------------------
Handy helper functions used all over the project.

``seed.py``
  Makes experiments repeatable by setting random seeds.

``logging.py``
  Adds ``maybe_init_wandb``, which starts a real Weights & Biases run
  when available and otherwise falls back to a safe dummy logger.
  Tests use a ``wb`` pytest fixture based on this helper, and both tests
  and training scripts log metrics and artifacts to Weights & Biases.

``pooling.py``
  Squishes all node embeddings in a graph into one by averaging.

``checkpoint.py``
  Saves and loads model checkpoints.

``schedule.py``
  Helps control the learning rate with a warmup and a cosine curve.

``ddp.py``
  Small helpers for distributed data parallel training.

Experiment Lab: ``experiments/``
----------------------------------------
Extra scripts to run special setups or gather results.

``grid_search.py``
  Runs combinations of model settings (like different sizes or learning
  rates) and collects results.

``baseline_integration.py``
  Hooks up third-party baseline models so they can be compared fairly.

``case_study.py``, ``ablation.py``, ``probing.py``
  Additional experiments like removing features to see their impact or
  probing embeddings for meaning.

``paired_effect_from_wandb.py``
  After a phase‑1 sweep finishes, this script groups runs by shared
  hyper‑parameters and computes the difference between JEPA and
  contrastive results.  It reports which method wins and writes a
  summary JSON file.

``export_best_from_wandb.py``
  Exports the top‑K configurations from a sweep and writes a new
  phase‑2 YAML to continue the search for the winning method.

Handy Scripts: ``scripts/``
----------------------------------------
Small programs for everyday tasks.

``download_unlabeled.py``
  Grabs random molecules from the internet (ZINC and PubChem) and saves
  them for pretraining.

``eval_moleculenet.py``, ``make_scaffold_splits.py``, ``train_jepa.py``
  Utilities for evaluation, dataset prep, or running a specific training
  setup.  ``train_jepa.py`` also registers subcommands to run
  phase‑1 sweeps (``sweep-run``) and individual runs.

``run-grid-or-phase1.sh``
  A shell script used in continuous integration that creates phase‑1
  sweeps for JEPA and the contrastive baseline.  It splits the sweeps
  by backbone so that each GNN type is explored under the same
  hyper‑parameter budget and invokes the paired‑effect analysis.

``run-grid-or-phase2.sh``
  Launches the second phase of hyper‑parameter search using Bayesian
  optimisation on the winning method from phase 1.

Docs and Configs
----------------------------------------
``scripts/default.yaml``
  Example configuration settings.

``docs/``
  Sphinx documentation, ready to be expanded.

``requirements.txt`` & ``pyproject.toml``
  Lists of needed Python packages.

Sample Data: ``samples/``
----------------------------------------
Contains tiny datasets for quick demos and tests.

Hyperparameter Sweeps: ``sweeps/``
----------------------------------------
The YAML files here define Weights & Biases sweep setups to try many
parameter combinations automatically.

Quality Check: ``tests/``
----------------------------------------
Scripts in this room make sure each lab tool works correctly.
Running the tests keeps experiments reliable.

Third-Party Friends: ``third_party/``
----------------------------------------
Contains external repositories (like MolCLR, HiMol, GeomGCL) used as
baselines or references.

Baseline Connectors: ``adapters/``
----------------------------------------
This room plugs our lab into baseline models from other projects.

``cli_runner.py``
  Orchestrates baseline training and embedding through the command line.

``native_adapter.py``
  Loads baseline repositories directly as Python modules so they fit
  right in.

``config.yaml``
  Stores the paths and command templates that tell the adapters what
  to run.

Conclusion
----------
This repository is a playground for teaching computers how molecules
behave.  It has tools for data, model building, training, experimenting,
and evaluating.  Each module is built to be modular and reusable so you
can mix and match pieces to fit your research needs.

Have fun exploring!

What's New
----------

Since the initial version of this guide, the project has gained a more
structured experimentation pipeline.  The new ``run-grid-or-phase1.sh``
script splits the first hyper‑parameter sweep by GNN backbone to ensure
that JEPA and its contrastive baseline are compared fairly even under
tight training budgets.  A paired‑effect analysis then selects the
better of the two methods for each backbone.  A second phase of sweeps
explores a wider range of hyper‑parameters for the winning method.  See
``agents.md`` in the repository root for a detailed description of how
these agents and sweeps work.

FAQ
---

Why did a 30-run phase-1 sweep report only nine pairs?
  The phase-1 driver launches one sweep for JEPA and one for the
  contrastive baseline, and each sweep reuses the same grid of shared
  hyper-parameters: the backbone ``gnn_type`` values ``gine``, ``dmpnn``
  and ``schnet3d`` and the random seeds ``{1, 2, 3}``.  The paired-effect
  script only forms a pair when it finds matching runs from *both*
  methods that share the same backbone and seed, so there can be at most
  ``3 backbones × 3 seeds = 9`` such combinations.  Extra trials that a
  sweep might launch beyond that grid explore method-specific knobs
  (such as JEPA's ``mask_ratio`` or ``ema_decay``) and still feed phase-2
  selection, but they do not increase the matched-pair count because they
  lack a partner run from the other method with identical shared
  settings.  Seeing nine pairs therefore means *both* methods populated
  every backbone/seed slot the key expects; if JEPA were missing entirely
  the report would exit with "No runs found" (under ``--strict``) instead
  of printing a winner.

What happens when there is no clear winner on the primary metric?
  ``paired_effect_from_wandb.py`` compares JEPA and contrastive runs only
  on the requested metric (``val_rmse`` by default) and never consults a
  secondary score.  It first tries to compute per-seed deltas using the
  shared backbone/seed grid; if no overlapping seeds exist it falls back
  to mean-per-method deltas for that ``pair_id`` and, as a last resort,
  to a single global mean delta when *no* pairs survived the filters.
  After those reductions, the mean delta decides the winner—lower is
  better for metrics ending in ``rmse``/``loss`` and higher is better for
  metrics like ``auc``.  A tie (exactly zero mean delta) implicitly goes
  to JEPA because the code chooses contrastive only when the contrastive
  mean beats JEPA by the chosen direction.  There is no hidden
  tie-breaker or secondary metric to consult; instead, the script writes
  out ``pairs_used`` and ``aggregate`` fields in ``paired_effect.json`` so
  you can audit whether the decision came from seed-wise pairs, per-pair
  means, or the global fallback.

What configuration keys form a "pair" today?
  The ``pair_id`` that links JEPA and contrastive runs is the hash of four
  architecture knobs written by ``scripts/commands/sweep_run.py``:
  ``gnn_type`` (which backbone was chosen), ``hidden_dim`` (model width),
  ``num_layers`` (depth), and ``contiguity`` (whether the masking window
  must stay contiguous).  Because both sweeps share those knobs, they act
  as the common key when the paired-effect report groups runs.  There is
  no need to adjust the key as long as new sweeps stick to the same
  shared backbone grid—method-specific settings (like JEPA's ``mask_ratio``
  or contrastive temperature) live outside the key on purpose so each
  method can search them independently.  Only if we introduced another
  architecture hyper-parameter that *both* methods tuned (for example, a
  new notion of encoder width) would we extend the key to include it.

How do we increase the number of matched pairs beyond ``3 × 3 = 9``?
  If the variance of the paired comparison feels too high, you can grow the
  shared backbone grid instead of editing the pairing key.  The phase-1
  launcher ``scripts/ci/run-grid-or-phase1.sh`` now understands two
  environment variables:

  - ``PHASE1_BACKBONES`` – a comma-separated list of backbones to plug into
    both sweep specs (for example ``gine,dmpnn,schnet3d,gin``).
  - ``PHASE1_SEEDS`` – a comma-separated list of integer seeds to apply to
    both sweeps (for example ``1,2,3,4,5``).

  Setting either variable rewrites the temporary YAML files that the script
  hands to Weights & Biases, so you can run
  ``PHASE1_SEEDS=1,2,3,4,5 GRID_MODE=wandb bash scripts/ci/run-grid-or-phase1.sh``
  and obtain ``3 backbones × 5 seeds = 15`` potential pairs without touching
  the tracked sweep templates.  This keeps the shared pairing knobs explicit
  while making it easy to dial up the number of matched runs when additional
  budget is available.

How many phase-1 runs do we need before ``pair_id``
overlap becomes likely?
  Phase-1 sweeps sample one of three backbones (``gine``, ``dmpnn``,
  ``schnet3d``) at random for each run.  The ``pair_id`` only depends on the
  chosen backbone because the other pairing knobs are fixed during the sweep.
  With ``n`` runs per method, the probability that at least one backbone is
  shared between the JEPA and contrastive sweeps is shown below alongside the
  chance that **each** sweep covers all three backbones.

  .. list-table:: Pair-id coverage when backbones are sampled uniformly
     :header-rows: 1
     :widths: 15 25 30

     * - Runs per method (``n``)
       - Probability of a shared ``pair_id``
       - Probability both sweeps cover all three backbones
     * - 3
       - 94.2%
       - 4.9%
     * - 4
       - 98.6%
       - 19.8%
     * - 5
       - 99.7%
       - 38.1%
     * - 6
       - 99.9%
       - 54.9%

  Even with four runs per method there is already a 98.6% chance that the two
  sweeps share at least one ``pair_id``.  Increasing the run count mainly
  improves the odds that both sweeps exercise *every* backbone, which is useful
  when you want to make a decision per backbone rather than relying on the
  global fallback in the paired-effect report.
