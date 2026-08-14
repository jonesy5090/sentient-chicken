"""Matched-seed comparison between conditions.

Every experiment in this project is a contrast between flocks that differ in exactly
one thing. This module is the harness for that, and `docs/backlog.md` specifies the
six-way condition ladder it will eventually need to support.

Two disciplines are enforced here rather than left to the caller:

**Seeds are matched across conditions.** Same world layout, same genome, same predator
arrival times. Without that, a difference between conditions is indistinguishable
from a difference between coops.

**The primary readout is within-run change, not absolute level.** A flock that starts
lucky looks good all run. What learning predicts is *improvement*, so we compare the
last third of a run against the first third, and then compare that change against the
control's change over the same period.

    usage:  python -m run.experiment --minutes 30 --seeds 4
"""

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate


class Condition(NamedTuple):
    name: str
    pc: PlasticConfig
    # Per-condition coop overrides, as (field, value) pairs applied via `_replace`.
    # A tuple rather than a dict so a Condition stays hashable and immutable, and so
    # any deviation from the default coop is written down in the condition itself
    # rather than set somewhere the reader will not look.
    cfg_patch: tuple = ()
    # Pallium size relative to default. The H4 ladder needs capacity to vary
    # independently of the channel; until E024 this was hardcoded, so the
    # capacity-matched control the whole design turns on could not be run.
    pallium_scale: float = 1.0
    # Innate comprehension. H4 asks whether an intact channel beats a shuffled one,
    # and a receiver who cannot respond to a call makes every condition identical.
    scaffold: bool = False

    def coop(self, cfg: CoopConfig) -> CoopConfig:
        return cfg._replace(**dict(self.cfg_patch)) if self.cfg_patch else cfg

    def regions(self):
        return (regions.DEFAULT_REGIONS if self.pallium_scale == 1.0
                else regions.DEFAULT_REGIONS.with_pallium(self.pallium_scale))


# The phase 1 contrast: does a hen that learns regulate herself better than one that
# cannot? Everything else -- genome, coop, predators -- is held identical.
# Exploration noise must be stated explicitly in every condition, never inherited.
# E010 compared a control that silently carried explore_sigma=0.6 -- the default,
# added in E007 *after* E004 ran -- against E004's noiseless one, and attributed the
# whole collapse to the gain change. Two things had changed, not one.
#
# The noise-only condition exists so that can never happen again: it separates what
# exploration *costs* from what learning *earns*, and those are different questions.
PHASE1 = (
    Condition("fixed (innate only)",
              PlasticConfig(enabled=False, explore_sigma=0.0)),
    Condition("noise only (no learning)",
              PlasticConfig(enabled=False, explore_sigma=0.6)),
    Condition("learning, no growth",
              PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)),
    Condition("learning + growth",
              PlasticConfig(enabled=True, explore_sigma=0.6)),
)


# E021. Two questions that share four of five conditions, so they share a run.
#
# **Does learning repay the cost of its own exploration?** E020 found learning at
# +0.001 and noise-only at +0.037 against the same control, with identical noise. That
# observation is post-hoc, so it is tested here on *fresh seeds* rather than re-read off
# the run that produced it, and as a first-class paired contrast rather than a
# difference of two differences.
#
# **Why did exploration become costly?** Noise-only went from t=0.32 (E013) to t=3.84
# (E020). For a condition with plasticity switched off, the audio path is the *only*
# thing that changed between those runs -- the strike-units fix, the reward composition
# and the readout rule all act through learning. So either the audio fix did it, or it
# was seed variation. The legacy pair below distinguishes them.
E021 = (
    Condition("fixed (innate only)",
              PlasticConfig(enabled=False, explore_sigma=0.0)),
    Condition("noise only (no learning)",
              PlasticConfig(enabled=False, explore_sigma=0.6)),
    Condition("learning, no growth",
              PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)),
    Condition("fixed, legacy audio",
              PlasticConfig(enabled=False, explore_sigma=0.0),
              cfg_patch=(("legacy_audio", True),)),
    Condition("noise only, legacy audio",
              PlasticConfig(enabled=False, explore_sigma=0.6),
              cfg_patch=(("legacy_audio", True),)),
)

# Contrasts to report beyond "everything against the first condition". Each is
# (treatment, control), and both must be named in the condition tuple.
E021_CONTRASTS = (
    ("learning, no growth", "noise only (no learning)"),
    ("noise only, legacy audio", "fixed, legacy audio"),
)


class Result(NamedTuple):
    hunger_early: float
    hunger_late: float
    fed_rate: float
    struck: float
    synapses: float


def run_condition(cond: Condition, seed: int, cfg: CoopConfig,
                  seconds: float, chunk_s: float) -> Result:
    cfg = cond.coop(cfg)
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), cond.regions(),
                         n_hens=cfg.n_hens, auditory_scaffold=cond.scaffold)
    x = brain.initial_state(p, cfg.n_hens)

    w_end, _x, p_end, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, chunk_s, cond.pc)

    n = len(s.hunger)
    third = max(1, n // 3)
    return Result(
        hunger_early=float(jnp.mean(s.hunger[:third])),
        hunger_late=float(jnp.mean(s.hunger[-third:])),
        fed_rate=float(jnp.sum(w_end.n_fed) / (cfg.n_hens * seconds / cfg.dt)),
        struck=float(jnp.sum(w_end.n_struck)),
        synapses=float(jnp.mean(jnp.sum(p_end.W != 0.0, axis=(1, 2)))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    ap.add_argument("--e021", action="store_true",
                    help="E021: exploration's cost, and where it came from")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="start seeds here; E021 uses 12 to test a post-hoc "
                         "observation on data that did not generate it")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    seconds = args.minutes * 60.0
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    conditions = E021 if args.e021 else PHASE1
    extra = E021_CONTRASTS if args.e021 else ()

    print(f"phase 1 contrast: {args.minutes:.0f} min of chicken time, "
          f"{len(seeds)} matched seeds ({seeds[0]}-{seeds[-1]}), "
          f"{cfg.n_hens} hens\n")
    hdr = (f"{'condition':<26} {'hunger early':>13} {'hunger late':>12} "
           f"{'change':>9} {'fed %':>7} {'struck':>7} {'synapses':>9}")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.perf_counter()
    table = {}
    for cond in conditions:
        rs = [run_condition(cond, s, cfg, seconds, args.chunk) for s in seeds]
        early = jnp.array([r.hunger_early for r in rs])
        late = jnp.array([r.hunger_late for r in rs])
        struck = jnp.array([r.struck for r in rs])
        table[cond.name] = {"hunger_change": late - early, "struck": struck}
        print(f"{cond.name:<26} {float(early.mean()):>13.3f} "
              f"{float(late.mean()):>12.3f} "
              f"{float(jnp.mean(late - early)):>+9.3f} "
              f"{float(jnp.mean(jnp.array([r.fed_rate for r in rs]))) * 100:>7.1f} "
              f"{float(struck.mean()):>7.0f} "
              f"{float(jnp.mean(jnp.array([r.synapses for r in rs]))):>9.0f}")

    print()
    _report(table, "hunger change", lambda v: v["hunger_change"], seeds,
            conditions, extra)
    _report(table, "predator exposure", lambda v: v["struck"], seeds,
            conditions, extra)

    print("\nnegative = better (drives regulated, or less time exposed to predators)")
    print(f"threshold is the two-tailed t at p=0.05 with {len(seeds) - 1} df "
          f"({_t_critical(len(seeds) - 1):.2f}), not 2 SE -- "
          f"read 'suggestive' as 'run more seeds', not as a result")
    print(f"wall clock: {time.perf_counter() - t0:.0f} s")


# Two-tailed t critical values at p=0.05, indexed by degrees of freedom.
# At the seed counts this project can afford, "exceeds 2 SE" is far too lenient --
# with 4 seeds (3 df) the threshold is 3.18, not 2. Using 2 SE at n=4 would call
# p=0.09 a result.
_T_CRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
           20: 2.086, 30: 2.042}


def _t_critical(df: int) -> float:
    if df in _T_CRIT:
        return _T_CRIT[df]
    return 1.96 if df > 30 else _T_CRIT[min(_T_CRIT, key=lambda k: abs(k - df))]


def _contrast(table: dict, label: str, pick, seeds, treat: str, ctrl: str,
              width: int = 42) -> None:
    """One paired contrast, `treat` minus `ctrl`.

    Paired because the seeds are matched: same coop, same genome, same predator
    arrivals, so the per-seed difference cancels most of the between-coop variance.
    Reported to three decimals with its standard error and a t verdict, never as a
    bare mean -- E003 was called a result off a 2-SE threshold that at n=4 admits
    p=0.09.
    """
    delta = pick(table[treat]) - pick(table[ctrl])
    mean = float(jnp.mean(delta))
    name = f"{treat} vs {ctrl}"
    n = len(seeds)
    if n < 2:
        print(f"{label:<18} {name:<{width}} {mean:+.3f} (single seed)")
        return
    se = float(jnp.std(delta, ddof=1)) / (n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    crit = _t_critical(n - 1)
    verdict = "better" if mean < 0 else "worse"
    if t > crit:
        flag = f"  SIGNIFICANT (t={t:.2f} > {crit:.2f})"
    elif t > 1.0:
        flag = f"  suggestive only (t={t:.2f}, need {crit:.2f})"
    else:
        flag = f"  noise (t={t:.2f})"
    print(f"{label:<18} {name:<{width}} {mean:+.3f} +/- {se:.3f} SE {verdict}{flag}")


def _report(table: dict, label: str, pick, seeds,
            conditions=PHASE1, extra=()) -> None:
    """Every condition against the first, then any explicitly requested pairs."""
    ctrl = conditions[0].name
    for cond in conditions[1:]:
        _contrast(table, label, pick, seeds, cond.name, ctrl)
    for treat, base in extra:
        _contrast(table, label, pick, seeds, treat, base)


if __name__ == "__main__":
    main()
