"""The build-day reveal slot on the strategy side: the variant behind ``reveal_variant``.

The arena plays a variant announced on build day under the ruleset id
:data:`REVEAL_RULESET_ID` (``arena.rules.reveal``). Its description is a
:class:`~strategy.rules.variants.VariantSpec` dictionary in ``variant.json``, found in

1. ``strategy/reveal/`` next to this package, which exists only inside a packaged bot
   (the sandbox gives a bot no environment and no files outside its bundle), else
2. ``$ARENA_REVEAL_VARIANT``, the directory the arena and the offline tools read.

:func:`get_variant` is :func:`strategy.rules.variants.get_variant` plus that slot. It is
kept out of ``variants`` on purpose: ``variants.py`` is a recorded input of the endgame
and showdown tables, so the reveal slot must not change its bytes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from strategy.rules.variants import UnsupportedVariant, VariantSpec
from strategy.rules.variants import get_variant as _practice_variant

REVEAL_RULESET_ID = "reveal_variant"
"""Same value as ``arena.domain.models.REVEAL_RULESET_ID``."""
REVEAL_ENV = "ARENA_REVEAL_VARIANT"
VARIANT_FILE = "variant.json"
TABLES_DIR = "tables"
HASH_MARKER = "variant_hash"
"""``tables/variant_hash`` names the variant the reveal tables were generated for."""
BUNDLED_REVEAL_DIR = Path(__file__).resolve().parents[1] / "reveal"
"""``strategy/reveal/`` inside a packaged bot. Never present in a repository checkout."""


def reveal_dir() -> Path | None:
    if (BUNDLED_REVEAL_DIR / VARIANT_FILE).exists():
        return BUNDLED_REVEAL_DIR
    raw = os.environ.get(REVEAL_ENV)
    return Path(raw) if raw else None


def reveal_active() -> bool:
    d = reveal_dir()
    return d is not None and (d / VARIANT_FILE).exists()


_CACHE: dict[tuple[str, float, int], VariantSpec] = {}


def reveal_variant() -> VariantSpec:
    d = reveal_dir()
    path = d / VARIANT_FILE if d is not None else None
    if path is None or not path.exists():
        raise UnsupportedVariant(f"{REVEAL_RULESET_ID} is not active: set ${REVEAL_ENV} to a directory with "
                                 f"{VARIANT_FILE} (or package the bot with its variant)")
    st = path.stat()
    key = (str(path), st.st_mtime, st.st_size)
    spec = _CACHE.get(key)
    if spec is None:
        spec = VariantSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if spec.ruleset_id != REVEAL_RULESET_ID:
            raise UnsupportedVariant(f"{path} declares ruleset_id {spec.ruleset_id!r}, not {REVEAL_RULESET_ID!r}")
        _CACHE[key] = spec
    return spec


def get_variant(ruleset_id: str) -> VariantSpec:
    if ruleset_id == REVEAL_RULESET_ID:
        return reveal_variant()
    return _practice_variant(ruleset_id)


def reveal_tables_dir() -> Path | None:
    """The reveal variant's generated tables, or None when there are none.

    A missing marker means no tables (callers use their labelled priors); a marker that
    names another variant raises, so tables from an earlier reveal never load.
    """
    d = reveal_dir()
    marker = d / TABLES_DIR / HASH_MARKER if d is not None else None
    if marker is None or not marker.exists():
        return None
    want = reveal_variant().variant_hash()
    got = marker.read_text(encoding="ascii").strip()
    if got != want:
        raise UnsupportedVariant(f"reveal tables in {marker.parent} were built for variant {got[:12]}, "
                                 f"not {want[:12]}")
    return marker.parent
