"""Bundle a parameter vector into a self-contained, sandbox-ready bot directory.

The sandboxed worker hides the repository before it imports bot code, so a bundle
carries its own copy of the decision-time closure of :mod:`strategy.policy.runtime`
(:data:`RUNTIME_FILES` plus the equity tables) next to ``bot.py`` and ``params.json``.
The worker puts the bundle directory first on ``sys.path``, so ``import strategy``
resolves to the staged copy; in-process the repository package (already imported)
is used, which is byte-identical. From the arena the closure needs only
``arena.agents.sdk``, which the worker loads before the sandbox applies.

``manifest.json`` lists every file's sha256, the parameter vector hash and the source
commit. Bundles are deterministic: the same vector at the same commit gives the same
bytes.

``kind="fallback"`` bundles :mod:`strategy.policy.generic` instead (:data:`FALLBACK_FILES`,
no parameters and no tables, not even the reveal ones). ``reveal=<dir>`` adds a build-day variant: its
``variant.json`` goes to ``strategy/reveal/variant.json`` and its ``tables/`` (the
``variant_hash`` marker plus regenerated equity tables) to ``strategy/reveal/tables/``,
because the sandbox gives a bot no environment and no files outside its bundle.

    python -m strategy.policy.package --params artifacts/search/nlhe-cem-v1/incumbent.json --out artifacts/bots/x
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from strategy.policy.baseline import SPACE
from strategy.policy.params import vector_hash

REPO = Path(__file__).resolve().parents[2]
RUNTIME_FILES = tuple(f"strategy/{rel}" for rel in (
    "__init__.py",
    "rules/__init__.py", "rules/adapter.py", "rules/canonical.py", "rules/reveal.py", "rules/variants.py",
    "features/__init__.py", "features/cards.py", "features/equity.py", "features/state.py",
    "tables/__init__.py", "tables/store.py", "tables/preflop.py", "tables/postflop.py",
    "policy/__init__.py", "policy/baseline.py", "policy/fallback.py", "policy/params.py", "policy/runtime.py",
    "policy/spot.py",
))
"""The decision-time import closure of :mod:`strategy.policy.runtime` (checked by the tests)."""
TABLE_PATTERNS = ("preflop-*.json.gz", "postflop-*.json.gz")
FALLBACK_FILES = tuple(f"strategy/{rel}" for rel in (
    "__init__.py",
    "rules/__init__.py", "rules/adapter.py", "rules/canonical.py", "rules/reveal.py", "rules/variants.py",
    "features/__init__.py", "features/cards.py",
    "policy/__init__.py", "policy/fallback.py", "policy/generic.py",
))
"""The decision-time import closure of :mod:`strategy.policy.generic` (checked by the tests)."""
REVEAL_DIR = "strategy/reveal"

FALLBACK_TEMPLATE = '''"""Packaged variant-agnostic fallback (strategy.policy.package). Do not edit."""

from strategy.policy.generic import GenericBot


def make_bot():
    return GenericBot(name={name!r})
'''

BOT_TEMPLATE = '''"""Packaged baseline policy {vector_hash_short} (strategy.policy.package). Do not edit."""

import json
from pathlib import Path

from strategy.policy.runtime import RuntimeBot

_P = json.loads((Path(__file__).resolve().parent / "params.json").read_text(encoding="ascii"))


def make_bot():
    return RuntimeBot(_P["params"], name={name!r}, gc_mode={gc_mode!r})
'''


def table_files() -> list[str]:
    data = REPO / "strategy" / "tables" / "data"
    return sorted(str(p.relative_to(REPO)) for pat in TABLE_PATTERNS for p in data.glob(pat))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reveal_files(reveal: Path, *, tables: bool = True) -> list[tuple[Path, str]]:
    """``(source, bundle path)`` for a reveal directory: the spec and (``tables``) every file under ``tables/``."""
    reveal = Path(reveal)
    out = [(reveal / "variant.json", f"{REVEAL_DIR}/variant.json")]
    tdir = reveal / "tables"
    if tables and tdir.is_dir():
        out += [(p, f"{REVEAL_DIR}/tables/{p.name}") for p in sorted(tdir.iterdir()) if p.is_file()]
    return out


def build_bundle(out: Path | str, params: Mapping[str, float] | None = None, *, name: str | None = None,
                 gc_mode: str = "freeze", commit: str | None = None, kind: str = "baseline",
                 reveal: Path | str | None = None) -> Path:
    """Write the bundle to ``out`` (replacing it) and return the path."""
    if kind not in ("baseline", "fallback"):
        raise ValueError(f"unknown bundle kind {kind!r}")
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    copies = [(REPO / rel, rel) for rel in (FALLBACK_FILES if kind == "fallback" else [*RUNTIME_FILES, *table_files()])]
    if reveal is not None:
        copies += reveal_files(Path(reveal), tables=kind != "fallback")
    for src, rel in copies:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    files = [rel for _, rel in copies]
    if kind == "fallback":
        name = name or "strategy_fallback"
        (tmp / "bot.py").write_text(FALLBACK_TEMPLATE.format(name=name), encoding="ascii")
        manifest: dict[str, Any] = {"kind": kind, "vector_hash": None, "space_hash": None, "name": name,
                                    "commit": commit,
                                    "files": {rel: _sha(tmp / rel) for rel in sorted([*files, "bot.py"])}}
    else:
        name = name or "strategy_baseline"
        values = SPACE.complete(params)
        vh = vector_hash(values)
        (tmp / "params.json").write_text(json.dumps({"vector_hash": vh, "space_hash": SPACE.space_hash(),
                                                     "params": values}, sort_keys=True, indent=1) + "\n",
                                         encoding="ascii")
        (tmp / "bot.py").write_text(BOT_TEMPLATE.format(vector_hash_short=vh[:12], name=name, gc_mode=gc_mode),
                                    encoding="ascii")
        manifest = {"vector_hash": vh, "space_hash": SPACE.space_hash(), "name": name,
                    "gc_mode": gc_mode, "commit": commit,
                    "files": {rel: _sha(tmp / rel) for rel in sorted([*files, "bot.py", "params.json"])}}
    if reveal is not None:
        from strategy.rules.variants import VariantSpec

        spec = VariantSpec.from_dict(json.loads((Path(reveal) / "variant.json").read_text(encoding="utf-8")))
        manifest["reveal_variant_hash"] = spec.variant_hash()
    manifest["bundle_hash"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    (tmp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="ascii")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def load_params(path: Path | str) -> dict[str, float]:
    """A vector from ``incumbent.json``/``params.json`` (``{"params": {...}}``) or a bare mapping."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(raw.get("params", raw))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.policy.package")
    p.add_argument("--params", help="JSON vector (default: the hand-written control)")
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    p.add_argument("--gc-mode", default="freeze", choices=("none", "freeze"))
    p.add_argument("--kind", default="baseline", choices=("baseline", "fallback"))
    p.add_argument("--reveal", help="reveal directory (variant.json and tables/) to bundle")
    args = p.parse_args(argv)
    params = load_params(args.params) if args.params else None
    from strategy.search.cem import _git_commit

    out = build_bundle(args.out, params, name=args.name, gc_mode=args.gc_mode, commit=_git_commit(), kind=args.kind,
                       reveal=args.reveal)
    print(json.dumps(json.loads((out / "manifest.json").read_text())["bundle_hash"]), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
