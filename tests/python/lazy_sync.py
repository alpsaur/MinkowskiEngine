# Copyright (c) 2026 alpsaur/MinkowskiEngine contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""ME_LAZY_SYNC stress test.

ME_LAZY_SYNC is latched once per process, so cross-flag comparison runs the
same deterministic stress loop in a subprocess with the flag flipped and
compares per-iteration losses within fp32 tolerance (the conv/pool kernels
accumulate with atomicAdd, so results are reproducible only up to reduction
order — corruption from a wrongly removed synchronization shows up as NaNs or
wildly divergent losses, not ulp-level noise).
"""
import json
import os
import subprocess
import sys
import unittest

import torch

import MinkowskiEngine as ME
from MinkowskiEngine import (
    MinkowskiAvgPooling,
    MinkowskiBroadcastAddition,
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiGlobalPooling,
    MinkowskiMaxPooling,
    SparseTensor,
)

LOSS_RTOL = 1e-3
LOSS_ATOL = 1e-4


def _random_coordinates(g, nrows, batch=2, D=3, max_coord=40):
    per_batch = []
    for b in range(batch):
        c = torch.randint(0, max_coord, (nrows, D), generator=g, dtype=torch.int32)
        c = torch.unique(c, dim=0)
        per_batch.append(
            torch.cat([torch.full((c.size(0), 1), b, dtype=torch.int32), c], 1)
        )
    return torch.cat(per_batch, 0)


class _StressNet(torch.nn.Module):
    """Mixed conv / pool / broadcast stack covering every op whose end-of-op
    synchronization is skipped under ME_LAZY_SYNC=1."""

    def __init__(self, D=3):
        super().__init__()
        self.conv1 = MinkowskiConvolution(3, 32, kernel_size=3, stride=1, bias=True, dimension=D)
        self.conv2 = MinkowskiConvolution(32, 64, kernel_size=3, stride=2, bias=True, dimension=D)
        self.max_pool = MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)
        self.avg_pool = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D)
        self.glob_pool = MinkowskiGlobalPooling()
        self.broadcast_add = MinkowskiBroadcastAddition()
        self.conv_tr = MinkowskiConvolutionTranspose(64, 32, kernel_size=2, stride=2, dimension=D)
        self.conv_out = MinkowskiConvolution(32, 8, kernel_size=3, stride=1, bias=True, dimension=D)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        y = self.max_pool(y)
        y = self.avg_pool(y)
        y = self.broadcast_add(y, self.glob_pool(y))
        y = self.conv_tr(y)
        y = self.conv_out(y)
        return y


def run_stress_loop(iters, device="cuda", seed=1234):
    torch.manual_seed(seed)
    net = _StressNet().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)

    losses = []
    for i in range(iters):
        coords = _random_coordinates(g, nrows=2000 + 13 * (i % 7))
        feats = torch.randn(coords.size(0), 3, generator=g)
        x = SparseTensor(feats.to(device), coords.to(device))
        out = net(x)
        loss = out.F.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _subprocess_losses(iters, lazy_sync, seed):
    env = os.environ.copy()
    env["ME_LAZY_SYNC"] = "1" if lazy_sync else "0"
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--emit-losses", str(iters), str(seed)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"stress subprocess (ME_LAZY_SYNC={env['ME_LAZY_SYNC']}) failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout.splitlines()[-1])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestLazySync(unittest.TestCase):
    ITERS = 200
    SEED = 1234

    def test_stress_cross_flag(self):
        """200 iterations of mixed conv/pool/broadcast fwd+bwd: losses under
        ME_LAZY_SYNC=1 must match ME_LAZY_SYNC=0 within fp tolerance."""
        ref = _subprocess_losses(self.ITERS, lazy_sync=False, seed=self.SEED)
        lazy = _subprocess_losses(self.ITERS, lazy_sync=True, seed=self.SEED)
        ref_t, lazy_t = torch.tensor(ref), torch.tensor(lazy)
        self.assertTrue(torch.isfinite(lazy_t).all(), "NaN/Inf loss under ME_LAZY_SYNC=1")
        self.assertTrue(
            torch.allclose(ref_t, lazy_t, rtol=LOSS_RTOL, atol=LOSS_ATOL),
            f"loss divergence between sync modes; max abs diff "
            f"{(ref_t - lazy_t).abs().max().item():.3e}",
        )

    def test_stress_lazy_repeatable(self):
        """Two independent flag-on runs must agree within the same tolerance
        (catches nondeterministic races that happen to miss the reference)."""
        a = _subprocess_losses(self.ITERS, lazy_sync=True, seed=self.SEED)
        b = _subprocess_losses(self.ITERS, lazy_sync=True, seed=self.SEED)
        a_t, b_t = torch.tensor(a), torch.tensor(b)
        self.assertTrue(torch.isfinite(a_t).all() and torch.isfinite(b_t).all())
        self.assertTrue(
            torch.allclose(a_t, b_t, rtol=LOSS_RTOL, atol=LOSS_ATOL),
            f"flag-on runs disagree; max abs diff {(a_t - b_t).abs().max().item():.3e}",
        )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--emit-losses":
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
        print(json.dumps(run_stress_loop(iters, seed=seed)))
    else:
        unittest.main()
