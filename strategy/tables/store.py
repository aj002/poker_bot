"""Write and load generated tables with provenance.

A table file is gzip-compressed canonical JSON ``{"provenance": {...}, "data": {...}}``.
``provenance.data_sha256`` hashes the canonical ``data`` JSON, and
``provenance.inputs_sha256`` hashes the generator and key-definition sources the data
depends on. :func:`load` verifies the data hash; :func:`stale` reports tables whose
inputs changed since they were built.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = Path(__file__).resolve().parent / "data"
ENV_DIR = "STRATEGY_TABLES_DIR"


def table_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)


def table_path(name: str, directory: Path | None = None) -> Path:
    return (directory or table_dir()) / f"{name}.json.gz"


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sources_sha256(modules: list[str]) -> str:
    """Hash of the source files (package-relative paths) a table's content depends on."""
    h = hashlib.sha256()
    for rel in sorted(modules):
        h.update(rel.encode())
        h.update((PACKAGE_ROOT / rel).read_bytes())
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT.parent, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write(name: str, data: dict[str, Any], *, generator: str, inputs: list[str], params: dict[str, Any],
          seconds: float, directory: Path | None = None) -> Path:
    body = canonical(data)
    provenance = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "generator": generator,
        "params": params,
        "inputs": sorted(inputs),
        "inputs_sha256": sources_sha256(inputs),
        "data_sha256": hashlib.sha256(body).hexdigest(),
        "git_commit": _git_commit(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "wall_seconds": round(seconds, 1),
        "command": " ".join([os.path.basename(sys.executable), "-m", generator, *sys.argv[1:]]),
    }
    path = table_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical({"provenance": provenance, "data": data})
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(raw)
    load.cache_clear()
    return path


class TableError(ValueError):
    pass


@lru_cache(maxsize=None)
def load(name: str, directory: str | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(provenance, data)`` or None when the table has not been generated."""
    path = table_path(name, Path(directory) if directory else None)
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        obj = json.loads(f.read())
    prov, data = obj["provenance"], obj["data"]
    if prov.get("format_version") != FORMAT_VERSION:
        raise TableError(f"{path}: unsupported table format {prov.get('format_version')!r}")
    if hashlib.sha256(canonical(data)).hexdigest() != prov["data_sha256"]:
        raise TableError(f"{path}: data hash mismatch")
    return prov, data


def stale(name: str) -> bool | None:
    """True when the inputs changed since the table was built; None when it is missing."""
    if array_path(name).exists() and not table_path(name).exists():
        return stale_arrays(name)
    loaded = load(name)
    if loaded is None:
        return None
    prov, _ = loaded
    return sources_sha256(prov["inputs"]) != prov["inputs_sha256"]


# ------------------------------------------------------------------ numpy tables
#
# Larger numeric tables are ``<name>.npz``: the arrays plus a ``__provenance__`` entry
# holding the same provenance JSON as above. ``data_sha256`` hashes every array's name,
# dtype, shape and bytes in name order.


def array_path(name: str, directory: Path | None = None) -> Path:
    return (directory or table_dir()) / f"{name}.npz"


def _hash_array(h: Any, key: str, array: Any) -> None:
    import numpy as np

    a = np.ascontiguousarray(array)
    h.update(key.encode())
    h.update(str(a.dtype).encode())
    h.update(repr(a.shape).encode())
    h.update(a.tobytes())


def arrays_sha256(arrays: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        _hash_array(h, key, arrays[key])
    return h.hexdigest()


def write_arrays(name: str, arrays: dict[str, Any], *, generator: str, inputs: list[str], params: dict[str, Any],
                 seconds: float, extra: dict[str, Any] | None = None, directory: Path | None = None) -> Path:
    import numpy as np

    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    if any(k.startswith("__") for k in arrays):
        raise ValueError("array names starting with '__' are reserved")
    provenance = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "generator": generator,
        "params": params,
        "inputs": sorted(inputs),
        "inputs_sha256": sources_sha256(inputs),
        "data_sha256": arrays_sha256(arrays),
        "git_commit": _git_commit(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "wall_seconds": round(seconds, 1),
        "command": " ".join([os.path.basename(sys.executable), "-m", generator, *sys.argv[1:]]),
        **(extra or {}),
    }
    path = array_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    meta = np.frombuffer(canonical(provenance), dtype=np.uint8)
    np.savez_compressed(tmp, __provenance__=meta, **arrays)
    tmp.replace(path)
    load_arrays.cache_clear()
    return path


def _read_arrays(name: str, directory: str | None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    import numpy as np

    path = array_path(name, Path(directory) if directory else None)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        prov = json.loads(bytes(z["__provenance__"]).decode())
        arrays = {k: z[k] for k in z.files if k != "__provenance__"}
    if prov.get("format_version") != FORMAT_VERSION:
        raise TableError(f"{path}: unsupported table format {prov.get('format_version')!r}")
    if arrays_sha256(arrays) != prov["data_sha256"]:
        raise TableError(f"{path}: data hash mismatch")
    for a in arrays.values():
        a.setflags(write=False)
    return prov, arrays


@lru_cache(maxsize=None)
def _load_arrays_cached(name: str, directory: str | None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    return _read_arrays(name, directory)


def load_arrays(name: str, directory: str | None = None, *, cache: bool = True
                ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(provenance, arrays)`` or None when the table has not been generated. Arrays are read-only.

    ``cache=False`` reads the file afresh and keeps no reference, for a caller that holds its own
    (smaller) form of the arrays and should not pin the raw ones in memory as well."""
    return _load_arrays_cached(name, directory) if cache else _read_arrays(name, directory)


load_arrays.cache_clear = _load_arrays_cached.cache_clear  # type: ignore[attr-defined]


def stream_arrays(name: str, directory: str | None = None) -> tuple[dict[str, Any], Iterator[tuple[str, Any]]] | None:
    """``(provenance, iterator of (name, array))`` or None when the table has not been generated.

    The arrays come one at a time in name order, so a caller that keeps a smaller form of each
    never holds the whole table's raw arrays at once. The data hash is checked as the arrays go
    by: exhausting the iterator raises :class:`TableError` on a mismatch. Arrays are read-only."""
    import numpy as np

    path = array_path(name, Path(directory) if directory else None)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    try:
        prov = json.loads(bytes(z["__provenance__"]).decode())
        if prov.get("format_version") != FORMAT_VERSION:
            raise TableError(f"{path}: unsupported table format {prov.get('format_version')!r}")
    except BaseException:
        z.close()
        raise

    def items() -> Iterator[tuple[str, Any]]:
        h = hashlib.sha256()
        try:
            for key in sorted(k for k in z.files if k != "__provenance__"):
                a = z[key]
                _hash_array(h, key, a)
                a.setflags(write=False)
                yield key, a
                del a
        finally:
            z.close()
        if h.hexdigest() != prov["data_sha256"]:
            raise TableError(f"{path}: data hash mismatch")

    return prov, items()


def stale_arrays(name: str) -> bool | None:
    loaded = load_arrays(name)
    if loaded is None:
        return None
    prov, _ = loaded
    return sources_sha256(prov["inputs"]) != prov["inputs_sha256"]
