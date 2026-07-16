# Performance Guide

This page collects practical knobs for getting the best throughput and memory
footprint out of this fork on CUDA 12.8 / Blackwell (and Ampere/Ada) GPUs.
All numbers below are illustrative measurements from a sparse U-Net training
run; your mileage will vary with model width, batch size, and point counts.


## TF32 on Ampere+

On Ampere (sm_80+) and newer architectures, enable TF32 so the fp32 GEMMs in
MinkowskiEngine engage the tensor cores:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True   # harmless to set alongside
```

TF32 trades a few mantissa bits (19-bit accumulation) for tensor-core
throughput. In our sparse U-Net runs this gave roughly a **15% step-time
reduction** with no measurable accuracy impact. It is on by default for some
PyTorch versions and off for others, so set it explicitly if you depend on it.


## Mixed precision (fp16 / bf16) — v0.5.6+

All MinkowskiEngine ops (convolution, pooling, broadcast, interpolation,
spmm) run in fp16 and bf16 as of v0.5.6, with GEMMs on tensor cores
accumulating in fp32. Use standard `torch.autocast`:

```python
import torch
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(sparse_input)   # ME layers cast + run in bf16 automatically
```

Typical effect: **~20% peak GPU memory reduction**; throughput gains grow
with channel width and batch size. bf16 needs no scaler. For fp16, wrap your
optimizer steps with `torch.cuda.amp.GradScaler`:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.autocast("cuda", dtype=torch.float16):
    out = model(sparse_input)
    loss = criterion(out.F, label)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

fp32/fp64 behavior is unchanged when autocast is not active.


## ME_LAZY_SYNC=1 — opt-in sync elimination (v0.5.6+)

MinkowskiEngine inserts a CUDA stream synchronization after every operation
by default. Convolution backward alone fires one sync *per kernel offset* —
27 times per layer for a 3³ kernel. Most of these are redundant under the
standard single-stream PyTorch training loop.

Set the environment variable to remove them:

```bash
ME_LAZY_SYNC=1 python train.py
```

Measured **~7–11% faster training steps** on a sparse U-Net at small batch
size. This is **off by default** (identical behavior to previous releases).
It assumes all ME work runs on a single CUDA stream (the PyTorch default) —
do not combine with custom multi-stream code or manual `torch.cuda.synchronize()`
expectations without auditing your code.


## Fused gather/scatter — v0.5.7+ (on by default)

The copy-GEMM convolution path stages rows for the GEMM with gather/scatter
kernels. Before v0.5.7 these launched once per kernel offset (27x per layer
for a 3³ kernel) as tiny scalar-copy kernels; profiling a sparse U-Net showed
them consuming **56% of total GPU time**. As of v0.5.7 they run as one
vectorized gather and one vectorized scatter-accumulate per direction —
measured: **−38% total GPU time per training step** on the same workload.

No action needed — it is on by default. Two knobs exist:

```bash
ME_FUSED_COPY=0            # restore the legacy per-offset path (A/B or debugging)
ME_FUSED_COPY_MAX_MB=2048  # cap on the fused staging buffer; larger convs fall
                           # back to chunked fusion above the cap (default 2048)
```


## OMP_NUM_THREADS on many-core machines

MinkowskiEngine uses OpenMP to parallelize kernel-map generation. On machines
with a very large number of cores (e.g. 64+), high thread counts spend most
of their time waiting on locks and throughput *drops*. Cap the thread count —
in our experience anything at or below ~24 is safe:

```bash
export OMP_NUM_THREADS=16
```

Sweep a few values to find the optimum for your CPU and workload.


## Clearing the CUDA cache for varying point counts

When the number of points (non-zero elements) varies across iterations — the
norm for point-cloud data — PyTorch's caching allocator fragments and can
hold far more memory than it needs. For long trainings, clear the cache at a
regular interval:

```python
if step % cache_clear_interval == 0:
    torch.cuda.empty_cache()
```

This is the same advice given in the upstream README; it is not a bug in
MinkowskiEngine but a consequence of fixed-size assumptions in the PyTorch
allocator.


## Build parallelism: MAX_COMPILATION_THREADS

When building from source (rather than installing a prebuilt wheel), the
number of parallel compile jobs can be capped to avoid running out of memory
or file descriptors on smaller machines:

```bash
export MAX_COMPILATION_THREADS=24   # default; lower for memory-constrained boxes
```

This only affects the `nvcc`/`c++` compile step, not runtime behavior.
