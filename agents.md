M‑JEPA Agents Overview

This document provides an overview of the training and evaluation agents used in the
M‑JEPA repository. It complements the existing kid_friendly_overview.rst by
focusing on the scripts and workflows that orchestrate experiments rather than
just describing the code layout.

Overview

The M‑JEPA project combines two pretraining paradigms: the Joint Embedding
Predictive Architecture (JEPA) and a contrastive baseline. Each of these
methods can be launched via command‑line scripts or hyper‑parameter sweeps.
Sweeps are used to discover good combinations of hyper‑parameters and to
compare the two training paradigms under matched conditions.

Directory Structure

The key directories relevant to the training agents are:

scripts/ – command‑line entry points and orchestration scripts. For
example, train_jepa.py launches pretraining and fine‑tuning runs,
run-grid-or-phase1.sh orchestrates phase‑1 sweeps, and
common.sh contains shared helper functions for the CI pipeline.

sweeps/ – YAML specifications for Weights & Biases sweeps. Phase‑1
sweeps explore a limited set of hyper‑parameters to decide between JEPA
and contrastive training, while phase‑2 sweeps perform a more thorough
search for the winning method.

experiments/ – Python modules that implement grid search and other
experiment helpers. The grid_search.py module defines the logic
executed during a sweep run.

tests/ – unit tests that ensure the training and sweeping logic
operates correctly. Any modifications to the agents should be
accompanied by appropriate tests.

Training Pipeline

The training pipeline is divided into two phases to balance exploration and
computational budget:

Phase 1 – Method Selection

A small sweep is run to compare JEPA and the contrastive baseline on a
shared set of hyper‑parameters. The sweep definitions live in
sweeps/sweep_phase1_jepa*.yaml and sweeps/sweep_phase1_contrastive*.yaml.

run-grid-or-phase1.sh splits the phase‑1 sweep by GNN backbone (e.g.
gine, dmpnn, schnet3d) to ensure each backbone is evaluated
fairly under limited time budgets. For each backbone it runs the JEPA
and contrastive sweeps, then computes a paired‑effect report that
determines which training paradigm performs better on that backbone.

The paired‑effect analysis pairs runs with identical hyper‑parameters to
compute a difference in validation metric (e.g. RMSE) and reports a
winner. Only the winning method is carried forward to phase 2 for
that backbone.

Phase 2 – Hyper‑Parameter Exploration

Once a winner is selected, a second sweep explores a wider set of
hyper‑parameters using Bayesian optimisation. The top‑K configurations
from phase 1 provide seed points for the phase‑2 search. These sweeps
are defined in sweeps/grid_sweep_phase2*.yaml and launched by
run-grid-or-phase2.sh.

Phase 2 focuses on optimising model capacity (hidden dimension, number
of layers), masking strategy, learning rate schedules and other knobs
specific to the winning method.

Evaluation

After pretraining, the repository offers several ways to evaluate models:

The benchmark and tox21 stages of the CI pipeline fine‑tune
pre‑trained embeddings on downstream tasks such as MoleculeNet
benchmarks and the Tox21 toxicity prediction dataset.

train_on_embeddings.py trains lightweight classifiers on
pre‑computed embeddings for rapid prototyping.

Running Sweeps

To launch a phase‑1 sweep manually, set the GRID_MODE and call the
orchestration script:

.. code-block:: bash

GRID_MODE=wandb bash scripts/run-grid-or-phase1.sh

The script will create the appropriate sweeps on Weights & Biases and
dispatch agents according to the configured WANDB_COUNT. When the
sweeps finish, it runs the paired‑effect analysis and exports the best
configuration for phase 2.

Similarly, phase‑2 sweeps can be launched by reading the sweep ID saved
in grid/phase2_sweep_id.txt and running the W&B agent:

.. code-block:: bash

export SWEEP_ID=$(<grid/phase2_sweep_id.txt)
wandb agent "$SWEEP_ID"

For more details on the available flags and configurations, see
scripts/train_jepa.py and the YAML files in the sweeps/ folder.

Contributing

Contributions are welcome! Please follow these guidelines:

Keep agent logic modular and reusable. New training methods should be
implemented as separate modules and registered through the existing
factory functions.

Update or add unit tests under tests/ to cover new behaviour.

Document any new scripts or workflows in this file so users have a
clear overview of how to run them.