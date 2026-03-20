# M-JEPA

> **Summary**
> - Frozen encoder lineages are immutable.
> - Dependent runs must set `PRETRAIN_EXP_ID` to reuse them.
> - To rebuild, remove or override the freeze marker explicitly.

Joint Embedding Predictive Architectures (JEPA) for molecular graphs. This
repository contains the training entry points, CI wrappers, and bundled example
benchmark data used for self-supervised pretraining, downstream evaluation, and
Tox21 analysis.

Detailed reviewer-facing reproducibility guidance lives in
[`REPRODUCE.md`](REPRODUCE.md). It is the canonical source for data layout,
runtime split behavior, expected outputs, troubleshooting, and manuscript
mapping notes.

## Reviewer path

- Start with [`REPRODUCE.md`](REPRODUCE.md) for the end-to-end reproduction
  guide.
- Use the documented Phase-3 `python scripts/train_jepa.py tox21 ...` path in
  `REPRODUCE.md` when reviewing Tox21 evaluation outputs.
- Treat `REPRODUCE.md` as the source of truth for what is bundled in `data/`,
  what is generated at runtime, and which outputs are actually emitted.

## 🔒 Frozen Encoder Lineages

CI discovers encoders that carry the `bench/encoder_frozen.ok` marker and
reuses them for downstream stages. Phase-1 sweeps now allocate their own
`EXP_ID` and set `GRID_EXP_ID` to that value so fresh grid logs land under a
dedicated directory. When a freeze marker is present, automation launches new
runs under a fresh `EXP_ID` while binding `PRETRAIN_EXP_ID` (and, when reading
existing sweeps, `GRID_EXP_ID`) to the frozen lineage.

```bash
# Run tox21 grading on a frozen encoder
export PRETRAIN_EXP_ID=1759825317
bash scripts/ci/run-tox21.sh
# Output -> ${TOX21_DIR:-/data/mjepa/experiments/$EXP_ID/tox21}/
```

See `docs/frozen_lineage_policy.rst` for override flags and lineage semantics,
including `FORCE_UNFREEZE_GRID=1` (rebuild a frozen lineage) and
`FORCE_RERUN=stage1,stage2` to selectively invalidate caches.

## Local development

1. **Install dependencies**

   ```bash
   git clone https://github.com/.../M-JEPA.git
   cd M-JEPA
   python -m pip install --upgrade pip setuptools wheel "numpy<2"
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1
   pip install --no-cache-dir -f https://data.pyg.org/whl/torch-2.2.1+cpu.html torch-scatter==2.1.2
   pip install torch-geometric==2.5.3
   pip install -r requirements.txt
   pre-commit install
   ```

   Turn on developer mode in developer settings for symlinks to work when
   running tests.
