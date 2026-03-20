# M-JEPA

M-JEPA is a research repository for Joint Embedding Predictive Architectures on molecular graphs. The repository contains the software used to train and evaluate molecular graph encoders, together with pipeline wrappers and selected curated data artifacts that support repository inspection and reviewer access.

At a high level, this repository includes:

- model and training code for JEPA pretraining and downstream evaluation;
- pipeline entry points and CI/Vast wrappers used to orchestrate multi-stage runs;
- analysis and reporting utilities for benchmark and case-study outputs; and
- reviewer-usable data artifacts already present under `data/`, including curated/preprocessed local copies or shards used by the repository workflows.

## Repository scope

This repository is intended to support inspection of the implemented methods, the structure of the pipeline, and the provenance of bundled data artifacts. It also supports limited local inspection or smoke-style validation where the environment and dependencies permit.

It is **not** presented as a short, one-command route to recreate the full manuscript end-to-end. Manuscript-scale results depend on longer multi-stage execution, runtime orchestration, hardware, caches, checkpoints, and other environment-specific details.

## Data and Software Availability

The repository provides software plus curated or preprocessed data artifacts that are already present in `data/`. These local artifacts are derived from public upstream sources and are included to support inspection of data lineage and repository behavior. Additional outputs such as caches, checkpoints, reports, and diagnostics are generated at runtime.

A more detailed, reviewer-facing note is provided in [`REPRODUCE.md`](REPRODUCE.md). That document explains what is bundled in the repository, what is generated during execution, and what kinds of reproducibility checks are realistically supported without overstating end-to-end manuscript reruns.
