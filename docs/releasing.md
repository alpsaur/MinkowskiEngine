# Validation and release checklist

This fork is maintained best-effort. A green CPU CI badge does not establish
GPU correctness, and a successful wheel build does not establish importability.

## Local/trusted GPU validation

Use a fresh environment with the intended torch/CUDA versions. CUDA 12.8 /
Blackwell releases currently target `torch~=2.9.0` from the cu128 index.

```bash
export OMP_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export MAX_COMPILATION_THREADS=6  # tune to available RAM
python setup.py build_ext --inplace
python -c "import torch, MinkowskiEngine as ME; assert torch.cuda.is_available() and ME.is_cuda_available(); ME.print_diagnostics()"
python -m pytest -q
bash scripts/build_wheels.sh
python scripts/check_wheel.py --cuda --tests wheelhouse/minkowskiengine-*.whl
```

`check_wheel.py` installs into a temporary target using the current interpreter's
torch/numpy dependencies. Its smoke process runs outside the checkout, with
`PYTHONPATH` pointing only to the installation target, and asserts that both
Python and native modules were loaded from that target. It does not modify the
current environment. `--tests` also copies the regression tests (not the
package) into that temporary directory and runs them against the installed
wheel; this needs pytest. Without `--cuda`, GPU visibility is disabled for the
check. Do not test a cp310 wheel with a cp312 interpreter.

Long legacy convolution leak probes are disabled during ordinary tests; opt in
with `ME_LEAK_TEST_ITER=100000` when specifically investigating lifetime issues.
Optional Open3D/data tests are separate from the synthetic regression suite:
opt in with `python -m pytest --run-data-tests` (or `ME_RUN_DATA_TESTS=1`).
These include very large channel/batch sweeps and 1000-iteration point-cloud
stress loops; they can take hours. Merely installing Open3D does not enable them.

## CI boundaries

- Pushes and PRs: CPU builds/tests for Python 3.10–3.13, torch 2.7/2.9, plus
  isolated installed-wheel smoke tests.
- GPU CI: **trusted manual dispatch only**, with `run_gpu=true`. A registered
  self-hosted `[linux, gpu]` runner is required; this repository does not
  provision one. Without a runner, perform and record the GPU checks locally.
  Never automatically run untrusted PR code on a persistent GPU host.
- Wheel workflow: CUDA-toolkit container builds for Python 3.10–3.13, pinned
  torch 2.9 cu128. Every wheel must pass install/import/CPU smoke tests before
  artifact upload. This step needs no visible GPU.

## Publishing

1. Update the version, changelog, README, citation, and docs fallback version.
2. Run local gates and independent review; merge only after required CI passes.
3. Push the version tag. The wheel workflow attaches artifacts to a **draft**
   release only after every matrix job passes. Manual wheel dispatches upload
   Actions artifacts without creating a release.
4. Download the actual release wheel(s) to the appropriate trusted GPU runtime;
   run `scripts/check_wheel.py --cuda --tests` on the artifact, not just a local build.
   Record which Python/GPU combinations were tested; do not imply that all
   wheels ran on hardware when only one did.
5. Inspect wheel assets, notes, and validation evidence, then publish the draft.
   Never publish merely because compilation succeeded.
