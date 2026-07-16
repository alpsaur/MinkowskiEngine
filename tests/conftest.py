"""Pytest conftest for the MinkowskiEngine test suite.

Puts the repository root first on sys.path so that `import MinkowskiEngine`
and `import MinkowskiEngineBackend._C` resolve to the in-repo source / freshly
built extension (e.g. from `python setup.py build_ext --inplace`), rather than
some other installed copy.
"""
import glob
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# common.py holds shared helpers/data, not test cases — skip collecting it.
collect_ignore_glob = ["python/common.py"]

# Several test modules (summary.py, global.py, network_speed.py,
# strided_conv.py) import open3d — a heavy optional dependency that is not in
# requirements.txt and not installed in CI — and re-raise on ImportError, so
# they break collection when open3d is absent. When open3d is not importable,
# skip collecting every tests/python module that references it; when open3d is
# available (e.g. local runs) they collect normally.
try:
    import open3d  # noqa: F401
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _open3d_re = re.compile(r"^\s*(?:import|from)\s+open3d\b", re.MULTILINE)
    collect_ignore = [
        os.path.relpath(p, _HERE)
        for p in glob.glob(os.path.join(_HERE, "python", "*.py"))
        if _open3d_re.search(open(p, encoding="utf-8").read())
    ]
