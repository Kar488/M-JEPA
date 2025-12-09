# Cache Directory Guidance for Pretraining

JEPA jobs must keep the unlabeled **dataset** and the on-disk **graph cache** separate:

- `--unlabeled-dir` should always point to the ZINC dataset root (for CI this is typically `/srv/mjepa/data/ZINC-canonicalized`).
- `--cache-dir` is only the destination for prebuilt graph pickles (for example `/data/mjepa/cache/graphs_10m` or `/data/mjepa/cache/graphs_250k`).
- The cache warmer is the only component that writes into the cache roots; training and sweep stages only read from them to avoid repeated featurisation.

Using a cache directory as `--unlabeled-dir` can produce empty datasets or tiny samples (~4k graphs), trigger pretrain crashes, and generate synthetic fallbacks. Always pass the ZINC dataset path instead; the loader will stream all available shards (or whatever you explicitly subsample with `--sample-unlabeled`) and reuse cached tensors when they exist under `--cache-dir`.

Typical pretrain invocation:

```bash
python -u scripts/train_jepa.py pretrain \
  --unlabeled-dir /srv/mjepa/data/ZINC-canonicalized \
  --cache-dir /data/mjepa/cache/graphs_10m \
  ...
```

If the dataset path is wrong or empty, `scripts/commands/pretrain.py` now logs a clear error and exits before training so that runs do not crash with missing graphs.

Outcomes of the fix
-------------------

- JEPA pretraining stages always read unlabeled graphs from the ZINC dataset directory (for example `${APP_DIR}/data/ZINC-canonicalized`), so the loader sees the full corpus instead of the ~4k fallback sample.
- Cache roots such as `/data/mjepa/cache/graphs_10m` remain purely for reuse; pointing `--unlabeled-dir` at them triggers the empty-dataset guard and exits early instead of running on zero graphs.
