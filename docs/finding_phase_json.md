# FINDINGS: Grid Best Config Exports

Phase-2 runs were staging an empty `best_grid_config.json` because the exporter
was writing the raw `wandb.Config` object, whose serialisation collapses to `{}`
when dumped directly; Phase-1 occasionally produced malformed output for the
same reason and because nested `value` wrappers and `_wandb` metadata leaked
into the file.  The exporter now normalises the winning run’s config into plain
Python primitives, enforces the presence of the knobs consumed downstream
(`training_method`, `gnn_type`, `hidden_dim`, `num_layers`), and refuses to
write when they are missing, guaranteeing valid JSON for both phases.
