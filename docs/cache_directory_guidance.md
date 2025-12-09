# Cache Directory Guidance for Pretraining

JEPA pretraining stages should point to the cache-warmed unlabeled datasets produced by the CI cache warmers (e.g., `/data/mjepa/cache/graphs_10m` or `graphs_10m_3d0_hd256` variants) rather than the raw ZINC source trees. The cache warmers canonicalize and shard the dataset ahead of time, so using these directories ensures:

- **Full corpus coverage.** The cache contains millions of featurized graphs instead of the ~4k samples in the small ZINC canonicalized directory.
- **Consistent featurization.** Warmers apply the same RDKit and hashing settings CI expects, avoiding drift between environments.
- **Faster startup.** Jobs can load prebuilt pickles directly without re-featurizing molecules, reducing CI wall time and variability.

If the cache directory is missing, CI will fail fast when `MJEPA_ALLOW_DATA_FALLBACKS=0` is set; otherwise it may fall back to the ZINC source directory, which is only suitable for smoke tests. Ensure the cache warmers run before pretraining stages to benefit from the larger, consistent dataset.
