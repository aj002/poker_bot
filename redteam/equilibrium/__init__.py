try:
    from redteam.equilibrium.bot import EquilibriumBot, make_bot
except ModuleNotFoundError:
    from bot import EquilibriumBot, make_bot  # type: ignore[import-not-found]

__all__ = ["EquilibriumBot", "make_bot"]
