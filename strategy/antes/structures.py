"""Named NLHE tournament structures with and without antes (decision-time safe: no imports).

Every structure shares the practice tournament (8 seats, 2,000 chips, 25 hands per level,
blinds 10/20 to 200/400, 200-hand cap, 50/30/20): only the antes differ. The schedules
are working assumptions until the official structure is published; tables and training
are parameterised by :func:`ante_ratio` and the posting mode, so a new schedule needs new
specs, not new code.

``classic``     every dealt seat antes about an eighth of the big blind from level 2;
``big_blind``   the big blind antes one big blind for the table from level 2 (dead money).
"""

from __future__ import annotations

BLIND_LEVELS = ((10, 20), (15, 30), (25, 50), (50, 100), (75, 150), (100, 200), (150, 300), (200, 400))

STRUCTURES: dict[str, dict] = {
    "none": {},
    "classic": {"antes": [0, 3, 5, 10, 15, 25, 40, 50], "ante_posting": "classic"},
    "big_blind": {"antes": [0, 30, 50, 100, 150, 200, 300, 400], "ante_posting": "big_blind"},
}
ANTE_STRUCTURES = ("classic", "big_blind")


def config_kwargs(name: str) -> dict:
    """``TournamentConfig`` keyword arguments of structure ``name`` (empty for ``none``)."""
    if name not in STRUCTURES:
        raise KeyError(f"unknown ante structure {name!r}; choose from {tuple(STRUCTURES)}")
    s = STRUCTURES[name]
    return {k: list(v) if isinstance(v, list) else v for k, v in s.items()}


def tournament_config(name: str, **overrides):
    from arena.domain.models import TournamentConfig

    return TournamentConfig(ruleset_id="practice_nlhe", **{**config_kwargs(name), **overrides})


def dead_money(ante: int, posting: str | None, dealt: int) -> int:
    """Chips the antes put in the pot before the blinds (nominal: nobody short)."""
    if not ante or posting is None:
        return 0
    return ante * dealt if posting == "classic" else ante


def ante_ratio(ante: int, posting: str | None, dealt: int, bb: int) -> float:
    """Total ante money per hand in big blinds: the parameter tables and training key on."""
    return dead_money(ante, posting, dealt) / bb if bb else 0.0
