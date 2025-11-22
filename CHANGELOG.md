# Changelog

## [vNext] — Frozen Encoder Lineage Policy
- Added frozen lineage marker (``encoder_frozen.ok``) after successful Tox21 grading.
- All downstream runs now write under new ``EXP_ID``s.
- Introduced explicit flags: ``FORCE_UNFREEZE_GRID``, ``STRICT_FROZEN``, ``ALLOW_CODE_DRIFT_WHEN_FROZEN``.
- Phase‑1 sweeps allocate new ``EXP_ID``/``GRID_EXP_ID`` pairs and record results in dedicated directories.
- Updated documentation for agents, pipeline overview, and CI behavior.
- Housekeeping update to retrigger Vast smoke builds.
