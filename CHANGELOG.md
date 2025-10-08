# Changelog

## [vNext] — Frozen Encoder Lineage Policy
- Added frozen lineage marker (``encoder_frozen.ok``) after successful Tox21 grading.
- All downstream runs now write under new ``EXP_ID``s.
- Introduced explicit flags: ``FORCE_UNFREEZE_GRID``, ``STRICT_FROZEN``, ``ALLOW_CODE_DRIFT_WHEN_FROZEN``.
- Updated documentation for agents, pipeline overview, and CI behavior.
