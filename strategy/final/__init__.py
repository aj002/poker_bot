"""Stage 8 / M8: the competition bot, its distillation study and the one-shot holdout evaluation.

* :mod:`strategy.final.runtime` - :class:`FinalBot`, the single entry point: the exploiting
  blueprint (:class:`strategy.exploit.runtime.ExploitBot`) on NLHE, the variant-agnostic
  fallback (:class:`strategy.policy.generic.GenericBot`) on every other ruleset.
* :mod:`strategy.final.package` - the deterministic, hashed, self-contained bundle.
* :mod:`strategy.final.distill` - the table student of the teacher's combined policy:
  rollouts, pack, agreement and the deadline-hybrid counterfactual.
* :mod:`strategy.final.equivalence` - two bundles through the real sandbox worker, decision by decision.
* :mod:`strategy.final.holdout` - the pre-registered final holdout runs and their Holm analysis.
"""
