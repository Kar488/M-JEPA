# Repository Reproducibility Note

This note is a conservative, reviewer-facing summary of repository evidence. It describes what data and software are present, what their provenance is, what is bundled versus generated at runtime, and what kinds of checks this repository can realistically support.

The wording here is intentionally restrained. It reflects the repository as checked in and avoids overstating reproducibility claims.

## Purpose

This repository provides:

- software for JEPA pretraining, downstream evaluation, benchmarking, and the Tox21 case-study path;
- pipeline entry points and CI/Vast wrappers used in longer-running staged execution; and
- curated or preprocessed local data artifacts under `data/` that support inspection of inputs and data lineage.

The repository therefore supports review of methods, software structure, and data provenance. It should not be read as a claim that a short local command sequence will recreate the full manuscript outputs.

## Data and Software Availability

### Software included in the repository

The repository includes:

- training and evaluation code under the tracked source tree;
- command-line entry points such as `scripts/train_jepa.py`;
- CI and orchestration wrappers, including the automated pipeline path used in `.github/workflows/ci-vast.yml`; and
- analysis/reporting utilities used to inspect or summarize run outputs.

These components document how the repository is organized and how staged runs are launched, even when the full manuscript workflow was executed through broader automation rather than a single local command.

### Data artifacts bundled in the repository

The `data/` directory contains reviewer-usable local artifacts that are already present in the repository. These are not described here as newly downloaded at review time; they are bundled local copies or prepared forms used by the software.

| Path | Repository role | Bundled form | Public provenance |
| --- | --- | --- | --- |
| `data/tox21/data.csv` | Labeled input used by the Tox21 and related evaluation paths | Local curated CSV | Public Tox21 data distributed via Hugging Face (`HUBioDataLab/tox21`). |
| `data/ZINC-canonicalized/` | Unlabeled corpus used by JEPA pretraining paths | Local parquet shards | Public dataset distributed via Hugging Face (`sagawa/ZINC-canonicalized`). |
| `data/BASF_AIPubChem_v4/` | Alternate unlabeled corpus available to the repository | Local parquet shards | Public dataset distributed via Hugging Face (`BASF-AI/PubChem-Raw`). |
| `data/katielinkmoleculenet_benchmark/train`, `val`, `test` | Checked-in benchmark fixture for benchmark-style evaluation paths | Local prepared split directories | Public benchmark material distributed via Hugging Face (`katielink/moleculenet-benchmark`). |

In this repository, these bundled artifacts are important because they make at least part of the data lineage public and machine-readable inside the submission itself. They also distinguish repository-inspectable inputs from outputs that are only created during execution.

## Bundled vs external vs runtime-generated

### Bundled in the repository

Bundled materials include the tracked software and the local data artifacts under `data/` listed above.

These materials support:

- inspection of the repository's methods and execution structure;
- inspection of data lineage and prepared inputs; and
- limited smoke-style validation on a local machine, where dependencies and resources permit.

### External or upstream public sources

The upstream origin of the bundled data artifacts is public. The repository includes prepared local forms because the software operates on those local CSV/parquet/pre-split inputs, not because the submission claims exclusive ownership of the underlying raw public datasets.

### Runtime-generated materials

The repository also works with outputs that are generated during execution rather than bundled in Git. These can include:

- graph or dataset caches;
- model checkpoints;
- run manifests and diagnostics;
- benchmark summaries or report files; and
- runtime-generated splits used by some evaluation paths.

Those generated materials depend on the execution context and should be treated separately from the bundled inputs already present in `data/`.

## Important distinctions for reviewers

### Public upstream data sources

The underlying datasets referenced above originate from public sources.

### Preprocessed or curated local artifacts in this repository

The repository contains local prepared forms that are directly usable by the implemented software, including CSV tables, parquet shards, and benchmark split directories.

### Caches

Caches are generated to accelerate featurization or repeated runs. They are convenience artifacts, not the primary record of public data provenance.

### Checkpoints

Checkpoints are runtime outputs from training or evaluation stages. They are not the same thing as the bundled data artifacts.

### Reports and diagnostics

Reports, summaries, manifests, and similar outputs are produced during execution and depend on the stage that was run.

### Runtime-generated splits

Some repository paths can derive splits during execution rather than reading only checked-in split directories. Where that occurs, those split assignments are execution products and should not be conflated with the bundled benchmark fixture already present in `data/`.

## What a reviewer can realistically do

This repository can reasonably support the following reviewer activities:

- inspect the software that implements pretraining, fine-tuning/evaluation, benchmarking, and the Tox21 case-study path;
- inspect the structure of the automated pipeline and wrapper scripts;
- inspect bundled local data artifacts and their public provenance; and
- perform limited smoke-style checks or partial runs if the local environment is configured with the required dependencies and sufficient compute.

For example, a reviewer may inspect the main CLI entry points and the checked-in `data/` contents, or run a narrowly scoped local command for software validation. Such checks should be understood as repository inspection or smoke validation, not as a claim of full manuscript rerun.

## Optional smoke/inspection paths only

Where a reviewer wants a minimal command example, the most appropriate repository-level examples are inspection-oriented entry points rather than claims of complete reproduction. For example:

```bash
python scripts/train_jepa.py --help
```

or, for wrapper inspection,

```bash
bash scripts/ci/run-tox21.sh --help
```

These examples are included only to show the available entry points. They are not presented as guarantees that a short local invocation will regenerate manuscript figures, tables, or final numerical results.

## Relationship to manuscript-scale outputs

The manuscript-scale outputs associated with this project arise from a broader multi-stage execution context. In practice, that context may involve staged automation, environment preparation, hardware-specific execution, caches, checkpoints, and pipeline orchestration.

Accordingly, repository outputs should be described conservatively:

- repository artifacts and generated outputs can support or contribute to manuscript analyses; but
- the repository documentation does not claim that manuscript figures or tables are recreated by a trivial local command sequence.

## Not claimed

This repository is **not** presented as a one-command, end-to-end manuscript rerun.

Simple local commands, where used at all, are for inspection or smoke validation only.

Exact numeric parity may depend on runtime environment, hardware, seeds, cached state, and the orchestration path used for the staged execution.

## Documentation boundary

This note reflects repository evidence only. It is intended to clarify data/software availability and practical reproducibility boundaries without overstating what the repository alone guarantees.
