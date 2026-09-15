#!/usr/bin/env python3
"""Reproducible sparse U-Net forward/backward latency and allocator benchmark.

No dataset, optimizer, or DataLoader is included. Run each configuration in a
fresh process: ME_LAZY_SYNC is latched by the native backend on first use.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import sys
import time


def build_model(ME, torch, channels):
    class SmallUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = ME.MinkowskiConvolution(channels, channels, 3, dimension=3)
            self.down = ME.MinkowskiConvolution(
                channels, 2 * channels, 2, stride=2, dimension=3
            )
            self.block = ME.MinkowskiConvolution(
                2 * channels, 2 * channels, 3, dimension=3
            )
            self.up = ME.MinkowskiConvolutionTranspose(
                2 * channels, channels, 2, stride=2, dimension=3
            )
            self.head = ME.MinkowskiConvolution(channels, channels, 3, dimension=3)
            self.relu = ME.MinkowskiReLU()

        def forward(self, x):
            skip = self.relu(self.stem(x))
            encoded = self.relu(self.down(skip))
            encoded = self.relu(self.block(encoded) + encoded)
            decoded = self.up(encoded, coordinates=skip.coordinate_map_key)
            return self.head(self.relu(decoded + skip))

    return SmallUNet()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--points",
        type=positive_int,
        default=10000,
        help="requested total points before deduplication",
    )
    parser.add_argument("--channels", type=positive_int, default=32)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--iterations", type=positive_int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--map-mode",
        choices=("cold", "warm"),
        default="cold",
        help="cold: rebuild maps per step; warm: reuse the same coordinate manager",
    )
    parser.add_argument("--fused-copy", choices=("on", "off"), default="on")
    parser.add_argument("--lazy-sync", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, help="also write JSON to this file")
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("mixed-precision benchmarking requires --device cuda")
    if args.batch_size > args.points:
        parser.error("--batch-size cannot exceed --points")

    os.environ["ME_FUSED_COPY"] = "1" if args.fused_copy == "on" else "0"
    os.environ["ME_LAZY_SYNC"] = "1" if args.lazy_sync else "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import MinkowskiEngine as ME

    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    ME.set_deterministic(args.deterministic)
    model = build_model(ME, torch, args.channels).to(args.device)
    # Fixed unique coordinates and features; generation and H2D input transfer
    # are deliberately outside the measured region.
    coords = torch.cat(
        [
            (torch.arange(args.points) % args.batch_size).reshape(-1, 1),
            torch.randint(0, 64, (args.points, 3)),
        ],
        dim=1,
    ).int()
    coords = torch.unique(coords, dim=0).to(args.device)
    features = torch.randn(len(coords), args.channels, device=args.device)
    probe = ME.SparseTensor(features, coords) if args.map_mode == "warm" else None
    if probe is not None:
        features = probe.F.detach()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if cuda else None

    def step():
        model.zero_grad(set_to_none=True)
        f = features.detach().requires_grad_()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if cuda:
            events[0].record()
        x = (
            ME.SparseTensor(f, coords)
            if probe is None
            else ME.SparseTensor(
                f,
                coordinate_map_key=probe.coordinate_map_key,
                coordinate_manager=probe.coordinate_manager,
            )
        )
        with (
            torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
        ):
            out = model(x)
        forward_end = time.perf_counter()
        if cuda:
            events[1].record()
        loss = out.F.float().square().mean()
        loss.backward()
        if cuda:
            events[2].record()
            events[2].synchronize()
        end = time.perf_counter()
        timings = {
            "step_wall_ms": (end - start) * 1000,
            "forward_ms": events[0].elapsed_time(events[1])
            if cuda
            else (forward_end - start) * 1000,
            "backward_and_loss_ms": events[1].elapsed_time(events[2])
            if cuda
            else (end - forward_end) * 1000,
        }
        # Capture allocation peaks BEFORE validation allocates temporary tensors.
        peaks = (
            (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            if cuda
            else (None, None)
        )
        # Checks and scalar extraction are outside measured timing/memory.
        if f.grad is None or not bool(torch.isfinite(f.grad).all()):
            raise RuntimeError("missing or non-finite feature gradients")
        if not all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise RuntimeError("missing or non-finite parameter gradients")
        return timings, loss.detach().item(), peaks

    for _ in range(args.warmup):
        step()
    model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline = None
    samples, memory_samples = [], []
    for _ in range(args.iterations):
        sample, loss, peaks = step()
        samples.append(sample)
        memory_samples.append(peaks)
    result = {
        "schema_version": 1,
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "environment": {
            "torch": torch.__version__,
            "minkowski": ME.__version__,
            "build": ME.get_build_info(),
            "device": torch.cuda.get_device_name() if cuda else "CPU",
            "omp_threads": torch.get_num_threads(),
        },
        "active_points": len(coords),
        "timing_kind": "CUDA event elapsed time (includes launch gaps)"
        if cuda
        else "wall clock",
        "timings": {
            key: {
                "median": statistics.median(s[key] for s in samples),
                "mean": statistics.mean(s[key] for s in samples),
                "min": min(s[key] for s in samples),
                "max": max(s[key] for s in samples),
            }
            for key in samples[0]
        },
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": max(p[0] for p in memory_samples) if cuda else None,
        "peak_reserved_bytes": max(p[1] for p in memory_samples) if cuda else None,
        "last_loss": loss,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    return result


if __name__ == "__main__":
    main()
