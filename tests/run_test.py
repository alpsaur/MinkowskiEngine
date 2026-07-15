"""Entrypoint that runs the MinkowskiEngine test suite under pytest.

Equivalent to running `pytest` from the repository root (see pytest.ini).

Usage:
    python tests/run_test.py
"""
import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["tests/python", "-v"]))
