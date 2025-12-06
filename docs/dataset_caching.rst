Dataset caching behaviour
=========================

 This note summarises how cached molecular graphs are separated when the
 ``add_3d`` flag is toggled and why SchNet3D always receives 3-D coordinates.

GraphDataset caches
-------------------

``GraphDataset.from_parquet`` and ``GraphDataset.from_csv`` add the suffix
``_3d0`` or ``_3d1`` to their cache filenames before writing ``.pkl`` files.
This ensures that a cache created without coordinates is never reused when
``add_3d=True`` and vice versa.【F:data/mdataset.py†L390-L483】

When loading an entire directory, ``GraphDataset.from_directory`` stores each
file in a dedicated subdirectory whose name already encodes the ``add_3d`` flag,
so 2-D and 3-D corpora are separated even before the filename suffix is
applied.【F:data/mdataset.py†L560-L629】

Pre-built dataset caches
------------------------

The sweep runner can persist pre-built datasets to speed up repeated
experiments.  The cache key hashed in ``scripts/commands/sweep_run.py`` includes
``add_3d`` so a 2-D cache cannot satisfy a 3-D request.  When a miss occurs it
rebuilds the dataset with the requested flag and stores the new
result.【F:scripts/commands/sweep_run.py†L260-L399】  The default ZINC corpus now
draws 10 M molecules, so caches materialised by automation live under
``cache/graphs_10m`` to make the size explicit and avoid mixing with older
250 K or 50 K runs.

Cache warmers
-------------

``scripts/ci/cache_warm_prebuilt_datasets.py`` layers two limits: a global
``--sample-{kind}`` cap (defaults pulled from the sweep YAMLs) and an optional
per-run ceiling (``--max-graphs-per-run``, default 250 K). The warmer now loops
internally when both limits are set: it removes only the manifest between
iterations so existing shards are reused, keeps writing manifests after every
chunk, and resumes until the sample cap is reached or the corpus is exhausted.
You no longer need to wrap the script in an external loop; expect ~40 internal
 iterations when ``--sample-unlabeled`` is 10 M and ``--max-graphs-per-run`` is
 250 K.【F:scripts/ci/cache_warm_prebuilt_datasets.py†L334-L447】 Use ``--force``
 only when discarding an existing manifest and shards to restart from scratch and
 set ``--sample-unlabeled`` (and ``--sample-labeled``) high enough to allow the
 loop to reach your intended total.【F:scripts/commands/dataset_cache.py†L55-L115】

Interpreting slow first epochs
------------------------------

When the unlabeled budget is set to 10 M, the first epoch can appear stalled for
an hour or more if the cache is missing or incomplete. This is usually the
worker building shards on the fly rather than a training deadlock. Confirm the
cache exists under ``cache/graphs_10m`` (or your ``SWEEP_CACHE_DIR`` override),
warm it via ``scripts/ci/cache_warm_prebuilt_datasets.py`` before launching
sweeps, and expect materially faster epochs once the pre-built tensors are in
place.【F:data/mdataset.py†L390-L483】【F:scripts/ci/cache_warm_prebuilt_datasets.py†L334-L447】

3-D coordinates for SchNet3D
----------------------------

``GraphDataset.smiles_to_graph`` generates RDKit conformers when ``add_3d`` is
true.  Successful embeddings append the ``(x, y, z)`` coordinates to the node
features and store them in the ``pos`` field, so any consumer such as SchNet3D
receives the expected geometry from both freshly built and cached
datasets.【F:data/mdataset.py†L260-L355】
