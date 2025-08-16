M-JEPA: A Kid-Friendly Guide
============================

Welcome to the M-JEPA project! Think of this repository as a giant science lab where computers learn to understand molecules. 
Each folder is like a different lab room with its own tools. Here's a tour that explains everything in simple, kid-friendly language.

Main Entrance: ``main.py``
----------------------------------------
This is the control center of the whole lab. It lets you:

- Run tiny experiments just to see if things work (``demo`` mode).
- Train a full model with real data (``full`` mode).
- Try lots of combinations of settings to find the best one (``grid`` mode).

You choose what to run using command-line switches like ``--mode``, ``--device``, or ``--method``.

Data Lab: ``data/``
----------------------------------------
This room creates and handles the molecules.

``dataset.py``
  Defines a ``GraphData`` class (a molecule as a graph) and a ``GraphDataset`` class (a collection of them).
  Can turn SMILES strings (a text way to describe molecules) into usable graph data for the models.
  Offers mini-batch helpers so multiple molecules can be processed together.

``augment.py``
  Adds a bit of playful randomness to molecules:

  - Spin them (``random_rotation``).
  - Freeze one of their angles (``mask_random_angle``).
  - Twist them slightly (``perturb_dihedral``).

Other helpers
  Files like ``parquet_loader.py`` and ``scaffold_split.py`` help load real datasets.

Model Factory: ``models/``
----------------------------------------
These are different "brains" that understand molecular graphs.

``base.py``
  Defines abstract classes so all models share the same structure.

``encoder.py``
  A simple Graph Neural Network (GNN) that reads a molecule and makes an embedding (a fancy word for a vector of numbers that describes it).

``edge_encoder.py``
  An upgraded GNN that also looks at information stored on the bonds (the edges).

``gnn_variants.py``
  Several popular GNN flavors like GraphSAGE, GIN, and a multi-head GAT.

``predictor.py``
  A small neural network that tries to guess a target embedding from the context embedding.

``ema.py``
  Implements Exponential Moving Average (EMA) to keep a smooth version of the model's weights, which helps with stability.

``factory.py``
  A factory that builds models based on the configuration settings. It can create different types of encoders and predictors.  

Training Room: ``training/``
----------------------------------------
This is where models learn.

``unsupervised.py``
  Trains the JEPA model or a contrastive baseline without labels.
  Handles fancy features like distributed training and mixed precision.
  Uses graph masking and augmentation to teach models to predict missing parts.

``supervised.py`` / ``supervised_with_val.py``
  Train simple "heads" on top of embeddings when labels are available.
  Optional validation for early stopping.

``train_on_embeddings.py``
  Uses pre-computed embeddings with simple models (like logistic regression) for quick evaluation.

``pretrain.py`` & ``baselines.py``
  Helpers to run baseline methods and pretraining flows.

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
  Runs combinations of model settings (like different sizes or learning rates) and collects results.

``baseline_integration.py``
  Hooks up third-party baseline models so they can be compared fairly.

``case_study.py``, ``ablation.py``, ``probing.py``
  Additional experiments like removing features to see their impact or probing embeddings for meaning.

Handy Scripts: ``scripts/``
----------------------------------------
Small programs for everyday tasks.

``download_unlabeled.py``
  Grabs random molecules from the internet (ZINC and PubChem) and saves them for pretraining.

``eval_moleculenet.py``, ``make_scaffold_splits.py``, ``train_jepa.py``
  Utilities for evaluation, dataset prep, or running a specific training setup.

Docs and Configs
----------------------------------------
``configs/default.yaml``
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
The YAML files here define Weights & Biases sweep setups to try many parameter combinations automatically.

Quality Check: ``tests/``
----------------------------------------
Scripts in this room make sure each lab tool works correctly.
Running the tests keeps experiments reliable.

Third-Party Friends: ``third_party/``
----------------------------------------
Contains external repositories (like MolCLR, HiMol, GeomGCL) used as baselines or references.

Baseline Connectors: ``adapters/``
----------------------------------------
This room plugs our lab into baseline models from other projects.

``cli_runner.py``
  Orchestrates baseline training and embedding through the command line.

``native_adapter.py``
  Loads baseline repositories directly as Python modules so they fit right in.

``config.yaml``
  Stores the paths and command templates that tell the adapters what to run.

Conclusion
----------
This repository is a playground for teaching computers how molecules behave. It has tools for data, model building, training, experimenting, and evaluating. 
Each module is built to be modular and reusable so you can mix and match pieces to fit your research needs.

Have fun exploring!
