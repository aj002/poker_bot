"""Pinned download of the PokerStars six-max-capable HandHQ stakes into a cache outside git.

HandHQ is 16 GB in the phh-dataset repository; this fetches only the PokerStars folders
(the one site that records ``seat_count`` on every hand, so six-max tables can be told
apart from six players at a full-ring table) with a blobless sparse git clone at the
pinned commit. Each folder's git tree id is pinned, and git verifies every object
against it; :func:`verify` re-checks the tree ids and that the working copy is
unmodified. Nothing here is committed: the derived statistics are what the repository
keeps.

    python -m strategy.data.handhq_fetch            # fetch (or verify) the cache
    python -m strategy.data.handhq_fetch --verify   # verify only; exit 1 if missing or altered
    python -m strategy.data.handhq_fetch --where    # print the data directory

The cache lives under ``$ARENA_DATA_CACHE`` (default ``~/.cache/arena``), in
``phh-dataset/handhq-<commit>/`` (about 6.7 GB, a minute or two to download).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

from strategy.data.fetch import FetchError, _git, cache_root
from strategy.data.handhq import COMMIT

REPO = "https://github.com/uoftcprg/phh-dataset.git"
PREFIX = "data/handhq/PS-2009-07-01_2009-07-23_{stake}H_OBFU"
STAKES = MappingProxyType({
    # stake: (git tree id at COMMIT, .phhs files, bytes)
    "25NL": ("5acc955469733f616586e8a55055547fb4e38683", 4379, 3_789_244_560),
    "50NL": ("6de5810febd512e58a21d875acc79506678c98cd", 1231, 853_539_575),
    "200NL": ("9f6827043657140c2afc3cdd286b03792776ac19", 1198, 884_459_852),
    "400NL": ("8942da19cfbb93724b64d0615f2960b2f877431f", 852, 639_026_768),
    "600NL": ("fdf0f9d8000ce6ae376e3724ab594fc60d30dd84", 334, 256_435_199),
    "1000NL": ("2c8127371cddc80899fc9cba884916f2004bda8e", 322, 229_095_459),
})
"""Every PokerStars NLHE stake in HandHQ (there is no 100NL folder)."""


def checkout_dir() -> Path:
    return cache_root() / "phh-dataset" / f"handhq-{COMMIT[:8]}"


def data_dir() -> Path:
    return checkout_dir() / "data" / "handhq"


def verify() -> Path:
    """The verified ``data/handhq`` directory; raises :class:`FetchError` if missing or altered."""
    work = checkout_dir()
    if not (work / ".git").exists():
        raise FetchError(f"no HandHQ checkout at {work}; run python -m strategy.data.handhq_fetch")
    if _git("rev-parse", "HEAD", cwd=work) != COMMIT:
        raise FetchError(f"{work} is not at the pinned commit {COMMIT}")
    for stake, (tree, files, _) in STAKES.items():
        folder = PREFIX.format(stake=stake)
        got = _git("rev-parse", f"HEAD:{folder}", cwd=work)
        if got != tree:
            raise FetchError(f"{folder} is tree {got}, expected {tree}")
        n = sum(1 for _ in (work / folder).glob("*/*.phhs"))
        if n != files:
            raise FetchError(f"{folder} holds {n} files, expected {files}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=work)
    if dirty:
        raise FetchError(f"{work} has modified files: {dirty.splitlines()[:3]}")
    return data_dir()


def fetch() -> Path:
    try:
        return verify()
    except FetchError:
        pass
    work = checkout_dir()
    work.parent.mkdir(parents=True, exist_ok=True)
    if not (work / ".git").exists():
        _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", "--sparse", REPO, str(work))
    _git("sparse-checkout", "set", "--no-cone", "/LICENSE", "/README.md",
         *(f"/{PREFIX.format(stake=s)}/" for s in STAKES), cwd=work)
    try:
        subprocess.run(["git", "checkout", "--quiet", "--force", COMMIT], cwd=work, check=True, timeout=3600)
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"git checkout failed with {exc.returncode}") from None
    return verify()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--where", action="store_true")
    a = ap.parse_args(argv)
    if a.where:
        print(data_dir())
        return 0
    try:
        root = verify() if a.verify else fetch()
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"HandHQ PokerStars stakes {', '.join(STAKES)} at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
