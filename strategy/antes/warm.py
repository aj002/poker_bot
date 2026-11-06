"""Record the ante load-time warm-up observations (offline).

:class:`strategy.antes.runtime.Antes` replays them after the bot's own warm-up, so the
first timed decisions of a hand with an ante (stripped features, ante push/fold lookups,
overlay lookups) do not run cold. One seeded block of each posting mode's structure is
played with the ``tables`` candidate in every seat rotation; the hero's preflop
decisions in hands with an ante are kept, ``PER_POSTING`` of them evenly spaced.

    python -m strategy.antes.warm
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

PER_POSTING = 24
LINEUP = ("strategy/antes/bots/tables.py", "style:aggressive", "baseline:calling_station", "style:rock",
          "style:station", "style:switcher", "style:weak_tight", "baseline:nlhe_heuristic")


def write_warm(ruleset_id: str = "practice_nlhe") -> Path:
    from strategy.antes import runtime
    from strategy.antes.pushfold import POSTINGS
    from strategy.antes.structures import tournament_config
    from strategy.value.rollouts import run_rollouts

    picked: list[str] = []
    decide = runtime.AnteExploitBot.decide
    for posting in POSTINGS:
        seen: list[str] = []

        def record(self, obs, state, _seen=seen):
            if obs.street == "preflop" and obs.settled_pot > 0 and state.get("seed") != "warm":
                _seen.append(obs.model_dump_json())
            return decide(self, obs, state)

        runtime.AnteExploitBot.decide = record
        try:
            run_rollouts(tournament_config(posting), list(LINEUP), blocks=1, base_seed=f"antes-warm|{ruleset_id}",
                         jobs=1)
        finally:
            runtime.AnteExploitBot.decide = decide
        picked += [seen[i * len(seen) // PER_POSTING] for i in range(min(PER_POSTING, len(seen)))]
    out = runtime.WARM_DIR / f"warm-{ruleset_id}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(json.dumps(picked).encode(), mtime=0))
    return out


if __name__ == "__main__":
    path = write_warm()
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(json.loads(gzip.decompress(path.read_bytes())))} "
          f"observations)")
