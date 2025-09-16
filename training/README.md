# Training Notes

## Torch Compile Warmup

The JEPA training loop now wraps the encoder and predictor with
[`torch.compile`](https://pytorch.org/docs/stable/compile/torch.compile.html)
whenever the runtime provides it. The first forward/backward pass triggers the
compiler to trace and specialise the model graph, so expect an initial
warm‑up that is noticeably slower than steady state. On current PyTorch 2.x
builds this typically means the first one or two mini-batches take roughly
2-3× longer while the graph is captured and optimised.

Once the compilation step finishes the remaining iterations benefit from the
fused kernels produced by `torch.compile`. Depending on the encoder backbone,
masking strategy and hardware this has yielded 10–30% higher throughput in our
internal benchmarks.

### Disabling Compilation

Use the `--no-compile` flag on `scripts/train_jepa.py pretrain` to run the
trainer in eager mode. This is useful when profiling with debuggers, running on
platforms where `torch.compile` is unavailable, or comparing performance against
previous releases. The trainer also automatically falls back to eager execution
if compilation raises an error at runtime.
