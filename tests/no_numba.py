"""Run a snippet in a fresh interpreter in which importing numba or llvmlite fails, even if they are installed."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_BLOCK = """
import importlib.abc, sys

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numba", "llvmlite"):
            raise ImportError(f"{name} is blocked: this code must not need it")

sys.meta_path.insert(0, _Block())
"""


def run_without_numba(code: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    script = _BLOCK + textwrap.dedent(code) + '\nassert "numba" not in sys.modules and "llvmlite" not in sys.modules\n'
    return subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
