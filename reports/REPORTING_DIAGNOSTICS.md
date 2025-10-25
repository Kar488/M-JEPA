# Reporting pipeline capabilities diagnostic

This note summarises how `reports/build_wandb_report.py` decides between the
three publishing paths and why the current static artifact fallback fails when
running against W&B 0.22.2.

## Capability detection

`detect_wandb_capabilities()` probes the runtime for the
`wandb_workspaces.reports.v2` module and records whether the environment can
instantiate panels or Markdown blocks. The result drives the publishing
strategy: full reports (panels + blocks), Markdown-only reports (blocks only),
or a static artifact upload when neither capability is available. 【F:reports/wandb_utils.py†L50-L177】【F:reports/build_wandb_report.py†L660-L713】

## Static artifact fallback

When only the static path is available, `_upload_static_report_artifact()` builds
a Markdown file locally and attempts to create an artifact via either
`Api.artifact_type(...).create_artifact` or `Api.create_artifact`. If both
methods are missing, the helper logs `Unable to create W&B artifact for static
report fallback` and the script exits without a report URL. 【F:reports/build_wandb_report.py†L545-L607】

## Client support in W&B 0.22.2

The public `wandb.Api` and `wandb.apis.public.ArtifactType` classes in W&B
0.22.2 do not expose a `create_artifact` method, so the fallback cannot succeed
with that client. 【c33c72†L12-L14】【2e5a05†L12-L14】

## Recommended fix

Either make the Reports v2 API available (install `wandb-workspaces` in the
reporting environment) so the Markdown path succeeds, or update the static
fallback to create a run-scoped `wandb.Artifact` and log it via
`run.log_artifact`, which works with the current client.
