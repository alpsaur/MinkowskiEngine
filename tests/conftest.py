"""Pytest conftest for the MinkowskiEngine test suite.

Puts the repository root first on sys.path so that `import MinkowskiEngine`
and `import MinkowskiEngineBackend._C` resolve to the in-repo source / freshly
built extension (e.g. from `python setup.py build_ext --inplace`), rather than
some other installed copy.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# common.py holds shared helpers/data, not test cases — skip collecting it.
collect_ignore_glob = ["python/common.py"]
