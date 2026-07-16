# Profiling MinkowskiEngine

MinkowskiEngine wraps its performance-critical backend calls in named
[`torch.profiler.record_function`](https://pytorch.org/docs/stable/profiler.html)
ranges so a profiled model shows *where* time is spent instead of one opaque
block. All ranges are prefixed `ME::`.

## Usage

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = model.cuda()
inp = inp  # a MinkowskiEngine.SparseTensor on CUDA

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    out = model(inp)
    out.F.sum().backward()

# Sort by CUDA time and filter to Minkowski ranges
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## What the ranges mean

- `ME::<Op>.forward` / `ME::<Op>.backward` — the backend compute call for an
  autograd op (`Convolution`, `ConvolutionTranspose`, `LocalPooling`,
  `LocalPoolingTranspose`, `GlobalPooling`, `DirectMaxPooling`, `Broadcast`,
  `Interpolation`, `SPMM`, `SPMMAverage`).
- `ME::CoordinateManager.<method>` — coordinate-map / kernel-map construction
  (`insert_and_map`, `insert_field`, `field_to_sparse_insert_and_map`,
  `stride`, `kernel_map`, `interpolation_map_weight`). Time here is the sparse
  bookkeeping that precedes the actual op compute; on the first iteration of a
  new input shape it can dominate, then largely disappears once maps are cached.

Comparing the two groups tells you whether a step is bound by op math or by
coordinate-map building. `record_function` is a no-op when no profiler is
active, so these ranges add no measurable overhead to normal training or
inference.
