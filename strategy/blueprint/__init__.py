"""Stage 4 abstracted MCCFR blueprint for eight-seat NLHE tournaments.

Offline (``--group train``): :mod:`.kernels` simulates abstract hands from sampled
between-hand states and runs external-sampling MCCFR on dense regret arrays;
:mod:`.train` drives it with checkpoints, :mod:`.exploit` measures exact heads-up
exploitability on a reduced game, :mod:`.response` trains approximate best responses.

Decision time (no numba): :mod:`.abstraction` (keys and bet sizes, shared with the
kernels), :mod:`.cards` (card buckets), :mod:`.policy` (the compact table) and
:mod:`.runtime` (the bot). See ``strategy/blueprint/README.md``.
"""
