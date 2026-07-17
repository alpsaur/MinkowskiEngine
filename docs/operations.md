# Operations Cookbook

This page is a practical "running ME in production and long trainings"
cookbook. It covers the recurring pain points seen across upstream and
downstream issue trackers: memory fragmentation, coordinate-manager leaks,
multi-GPU setup, torch.compile incompatibility, determinism, and the
single-stream assumption behind `ME_LAZY_SYNC`.

Every section below is code-snippet-first. For throughput and memory knobs
(TF32, mixed precision, fused gather/scatter), see the
[Performance Guide](performance.md) instead.


## Memory management with varying point counts

When the number of active points (non-zero coordinates) changes every
iteration, which is the norm for point-cloud and segmentation workloads,
PyTorch's CUDA caching allocator fragments. It holds fixed-size blocks that
matched a previous iteration's allocation profile and cannot satisfy the new
one without over-allocating. Left unchecked, peak memory creeps upward until
the process OOMs. This is a consequence of the allocator's design, not a leak
in MinkowskiEngine, but it bites ME workloads harder than dense ones because
sparse tensor sizes swing widely. This is the pattern behind
[#150](https://github.com/NVIDIA/MinkowskiEngine/issues/150),
[#290](https://github.com/NVIDIA/MinkowskiEngine/issues/290), and
[#359](https://github.com/NVIDIA/MinkowskiEngine/issues/359).

Clear the caching allocator on a regular interval:

```python
cache_clear_interval = 500  # tune; every few hundred steps is typical

for step, batch in enumerate(loader):
    ...
    loss.backward()
    optimizer.step()

    if step % cache_clear_interval == 0:
        torch.cuda.empty_cache()
```

### The OOM-leak caveat (issue #359)

A C++-side allocation failure (a hard GPU OOM during a kernel) can leave GPU
memory in an intermediate state that the Python process never recovers. The
process keeps running but the GPU is stranded at near-full utilization, so the
next iteration OOMs again immediately. You cannot reliably catch and continue
from this with a `try/except RuntimeError` alone. For long unattended
trainings, structure training as a restartable loop so an OOM does not strand
the GPU until a manual restart:

```python
import subprocess, sys, os

def run_training_segment(resume_ckpt=None):
    """Launch one training subprocess; checkpoint every N steps inside."""
    cmd = [sys.executable, "train.py"]
    if resume_ckpt:
        cmd += ["--resume", resume_ckpt]
    env = dict(os.environ)
    completed = subprocess.run(cmd, env=env)
    return completed.returncode

ckpt = "ckpt/latest.pt"
while True:
    rc = run_training_segment(resume_ckpt=ckpt)
    if rc == 0:
        break                       # training finished cleanly
    # non-zero: the subprocess died (OOM, etc.). The GPU is freed when the
    # child process exits. Loop back and resume from the last checkpoint.
```

The key property: a hard OOM kills the child process, the OS reclaims the GPU,
and the parent relaunches from the last saved checkpoint. This turns a
stranded-GPU situation into a short delay instead of a silent stall.


## Coordinate-manager lifecycle

Every `SparseTensor` owns a coordinate manager that tracks the kernel maps
(hashed coordinate-to-coordinate mappings) for all operations applied to it.
There are two modes, set with:

```python
import MinkowskiEngine as ME

ME.set_sparse_tensor_operation_mode(
    ME.SparseTensorOperationMode.SEPARATE_COORDINATE_MANAGER  # default
)
```

Under the default `SEPARATE_COORDINATE_MANAGER` mode, each SparseTensor gets
its own coordinate manager that is garbage-collected when the tensor goes out
of scope. No manual cleanup is needed.

Under `SHARE_COORDINATE_MANAGER`, a single global coordinate manager is reused
across all SparseTensors. This is faster (kernel maps are cached and shared
across layers) but the cache grows for the lifetime of the process. If you use
this mode you **must** clear the global manager after every iteration, or the
kernel-map cache grows without bound and leaks memory:

```python
ME.set_sparse_tensor_operation_mode(
    ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER
)

for step, batch in enumerate(loader):
    out = model(sparse_input)
    loss = criterion(out.F, label)
    loss.backward()
    optimizer.step()

    ME.clear_global_coordinate_manager()  # required every iteration
```

If you see memory growing linearly with the number of steps and you use the
shared mode, a missing `clear_global_coordinate_manager()` call is the first
thing to check.


## Multi-GPU / DistributedDataParallel

Multi-GPU is the single most common source of confusion in downstream usage.
The rules:

**DataParallel is not supported.** PyTorch's `DataParallel` replicates module
attributes by shallow copy, which breaks the internal coordinate-manager
references inside ME modules. It silently produces wrong results or crashes.
This is documented in upstream
[#264](https://github.com/NVIDIA/MinkowskiEngine/issues/264). Use
`DistributedDataParallel` instead.

**Working DDP recipe:**

```python
import os
import torch
import torch.distributed as dist
import MinkowskiEngine as ME

def setup_ddp(local_rank):
    # 1. Set the device BEFORE init_process_group.
    torch.cuda.set_device(local_rank)

    # 2. NCCL is the only backend that works for ME on GPU.
    dist.init_process_group(backend="nccl")

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_ddp(local_rank)

    model = build_model().cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        # Required if your head has conditionally-unused parameters
        # (e.g. Mask3D-style heads with dynamic pruning).
        find_unused_parameters=True,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        drop_last=True,   # avoid ragged last batch that breaks collation
    )
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, ...)

    cache_clear_interval = 500
    for step, batch in enumerate(loader):
        coords, feats, labels = batch
        # Use ME.utils.sparse_collate / batched_coordinates to build
        # per-rank SparseTensor inputs.
        sin = ME.SparseTensor(feats, coords).cuda()

        out = model(sin)
        loss = criterion(out.F, labels)
        loss.backward()
        optimizer.step()

        if step % cache_clear_interval == 0:
            torch.cuda.empty_cache()   # per-rank; each rank clears its own

if __name__ == "__main__":
    main()
```

Launch with `torchrun` (one process per GPU):

```bash
torchrun --nproc_per_node=4 train.py
```

**Pickling note:** ME modules could not be pickled under `ddp_spawn` in early
upstream releases; this was fixed in upstream
[PR #139](https://github.com/NVIDIA/MinkowskiEngine/pull/139) and is inherited
by this fork. Prefer standard `torchrun` launch over `mp.spawn` regardless, as
`spawn` complicates debugging and checkpointing.

> **Honesty note:** multi-GPU is upstream functionality that this fork has
> **not** re-validated on torch 2.9. Single-GPU is the tested path in this
> fork's CI. The guidance above is community-sourced and matches what works in
> the major downstream repositories (MinkowskiEngine-based detection and
> segmentation pipelines). If you hit a DDP-specific issue, please report it.


## torch.compile / Dynamo

`torch.compile` does not work on models containing MinkowskiEngine layers. ME
operations are C++ autograd `Function` subclasses that hold opaque
coordinate-manager and kernel-map objects. Dynamo cannot trace through them,
so `torch.compile` on a model with ME layers either graph-breaks at every ME
op (losing most of the compile benefit) or errors outright.

**Workaround:** disable Dynamo on the ME-containing submodule and compile only
the dense head:

```python
import torch
import torch._dynamo
import MinkowskiEngine as ME

model = build_model()

# Option A: disable on the sparse backbone, compile the dense head.
torch._dynamo.disable(model.sparse_backbone)
model.dense_head = torch.compile(model.dense_head)

# Option B: compile nothing, avoid the overhead entirely.
# torch.compile(model)  # <- will graph-break; do not do this
```

Full `torch.compile` support would require redesigning the ME op API so that
Dynamo can see tensor-in/tensor-out boundaries with no opaque handles. This is
not planned for this fork.


## Determinism

There are two independent sources of non-determinism to be aware of.

### (a) Coordinate quantization differs between CPU and GPU

MinkowskiEngine quantizes continuous float coordinates to an integer grid as
the first step of building a SparseTensor. The GPU quantization kernel can
produce a different set of active coordinates (or a different ordering) than
the CPU path for identical float inputs, due to rounding and parallel
insertion order. This is upstream
[#441](https://github.com/NVIDIA/MinkowskiEngine/issues/441).

If you quantize on CPU during preprocessing but on GPU during training (or
vice versa), your active coordinate set can shift between data preparation and
training, or between train and inference. **Quantize on one device
consistently** across your entire pipeline:

```python
# Pick one: quantize on GPU for the whole pipeline.
coords = coords.cuda()
feats = feats.cuda()
sin = ME.SparseTensor(feats, coords, device="cuda")
```

### (b) Convolution outputs vary in low-order bits run-to-run

Sparse convolution uses atomic adds and a parallel dispatch order that is not
fixed across runs. Two forward passes on the same input produce identical
results up to floating-point reduction-order differences, which appear as
low-order-bit noise in the output. This is upstream
[#554](https://github.com/NVIDIA/MinkowskiEngine/issues/554).

A built-in `ME.set_deterministic()` option is planned (coordinate sorting
before each kernel plus deterministic atomics) but is **not yet merged** into
this fork. Until it lands, the manual workaround is to sort the coordinate
key before constructing the SparseTensor so that the insertion order is fixed
regardless of the upstream data order:

```python
# Manual deterministic ordering: sort rows by (batch, x, y, z, ...).
order = torch.argsort(coords[:, 0])           # stable sort on full key
coords = coords[order]
feats = feats[order]
sin = ME.SparseTensor(feats, coords)
```

This fixes the coordinate ordering but does not eliminate the atomic-add
noise inside the GEMM. For bitwise reproducibility you also need
`torch.use_deterministic_algorithms(True)`, which will error on any ME op
that lacks a deterministic kernel (currently all sparse convs), so full
bitwise determinism is not achievable today without the planned built-in.


## ME_LAZY_SYNC: single-stream assumption

The `ME_LAZY_SYNC=1` environment variable (see the
[Performance Guide](performance.md)) removes the end-of-operation CUDA stream
synchronization that ME inserts by default. It gives a measurable speedup but
relies on an assumption: **all ME work executes on a single CUDA stream**,
which is the PyTorch default.

If your code uses custom side streams, manual `torch.cuda.Stream()`
contexts around ME ops, or CUDA-graph-style capture (`torch.cuda.graph`),
`ME_LAZY_SYNC=1` is unsafe. The removed syncs were guarding against
cross-stream read-after-write hazards; without them, a consumer on stream B
can read output that a producer on stream A has not finished writing.
Symptoms are intermittent, non-reproducible NaNs or silently wrong outputs.

`ME_LAZY_SYNC` is **off by default**. Leave it off if you do any of the
following:

- use `torch.cuda.Stream` to overlap ME ops with other work,
- capture ME ops inside `torch.cuda.graph` / `make_graphed_callables`,
- call `torch.cuda.synchronize()` manually between custom streams.

If you only use the default stream (no explicit stream management), turning it
on is safe and recommended. See the
[Performance Guide](performance.md) for measured numbers.
