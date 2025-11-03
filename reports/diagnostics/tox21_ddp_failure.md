# Tox21 stage DDP failure modes

## Symptom recap

Recent Vast runs abort immediately after the stage wrapper prints the wall-clock
budget message:

```
[stage] wall budget=900m (54000s), grace=120s
```
followed by `train_jepa.py` returning exit code `1`. No training output is
emitted because the worker crashes before any of the case-study tasks begin.

## Why the crash happens right after the budget log

The tox21 stage now enables PyTorch distributed launchers whenever the resolved
`--devices` flag is greater than one. The wrapper clones the argument vector,
normalises the device count, and wraps the usual `train_jepa.py tox21 …`
invocation with `python -m torch.distributed.run` so that each GPU gets its own
worker process.【F:scripts/ci/stage.sh†L1580-L1774】 If at least two devices are
visible the wrapper exports `WORLD_SIZE`, chooses a free rendezvous port, and
prints the `[stage:tox21] enabling torch.distributed.run …` diagnostic before the
launch.【F:scripts/ci/stage.sh†L1726-L1742】 Any failure inside the launcher causes
`run_with_timeout` to surface the non-zero return code immediately after the wall
budget message.【F:scripts/ci/stage.sh†L1779-L1814】

Once the distributed runner starts, every tox21 task calls
`run_tox21_case_study(…, devices=N)`; the helper passes that device count into the
pretraining and linear-probe trainers.【F:scripts/commands/tox21.py†L576-L666】【F:experiments/case_study.py†L2072-L2092】【F:experiments/case_study.py†L2231-L2270】
The supervised trainer interprets any `devices > 1` request as a distributed job
and immediately executes `utils.ddp.init_distributed()` before touching the
dataset.【F:training/supervised.py†L772-L781】

`init_distributed` performs several hard checks before it ever reaches the
training loop. It tries to pin `CUDA_VISIBLE_DEVICES` to the local rank and
raises `RuntimeError` if fewer GPUs are visible than were requested, and it bails
out whenever NCCL cannot be initialised for the requested world size.【F:utils/ddp.py†L14-L63】【F:utils/ddp.py†L204-L239】 Those exceptions are not caught by the tox21
command path, so they bubble straight back to the launcher. Because the crash
happens during the distributed handshake, nothing else is logged between the
wall-budget line and the final `exit 1` banner.

## What to inspect on Vast

1. Open the stage log at `${LOG_DIR}/tox21.log`. When the GPU count is lower
   than `--devices`, the ddp shim prints a clamp warning or an “insufficient CUDA
   devices” error before exiting.【F:scripts/ci/stage.sh†L1714-L1756】【F:utils/ddp.py†L53-L234】
2. If the clamp never appears, grep the same log for NCCL errors – failed
   backends will also trigger the early exit from `init_distributed`.
3. Confirm the environment variables the wrapper exports (`WORLD_SIZE`,
   `MASTER_ADDR`, `MASTER_PORT`) are not being overridden by upstream job
   wrappers, as that can confuse the launcher before it spawns workers.

In short, the immediate exit code `1` occurs when the new distributed bootstrap
cannot satisfy the requested device topology; the failure happens before the
case-study loop, so the only visible clue in the high-level log is the wall
budget banner.
