"""Can the intent-to-treat metric see an effect at all? (E029)

E028 rebuilt H4's metric so its denominator is every (hen, dive) pair, which no
behaviour can reach, and the registered contrast came back **-0.029 +/- 0.020, t=1.42**
-- not significant at 12 seeds. That number has two readings and they are opposite:

  1. there is no meaningful channel effect, or
  2. there is one, and 12 seeds on this metric cannot resolve it.

`CLAUDE.md` is explicit that a null is only informative if the instrument could have
shown a positive, and that a positive control is not optional before concluding a rule
or a measurement cannot detect something. **In twenty-eight experiments this project has
never run one.** Every null it has recorded was interpreted without knowing whether the
harness could have said otherwise.

So: plant effects of known, increasing size and see where the metric starts reporting
them. `scaffold_gain` scales the innate response to hearing an alarm -- gain 1.0 is the
hen E028 measured, and higher gains are a deliberately exaggerated bird that exists only
to test the harness. If the metric cannot detect gain 4, it could not have detected
anything, and E028's null says nothing about H4. If it detects gain 2 comfortably but not
gain 1, then E028 was underpowered and the fix is seeds, not a new hypothesis.

The contrast is the registered one throughout: **L vs C? yoked**, same as E028, so the
numbers are directly comparable to the -0.029 this is diagnosing.

    usage:  python -m run.poscontrol --minutes 10 --seeds 12
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp

from coop import spec
from run.experiment import Condition, _t_critical
from run.h4 import EXPANDED, INNATE, H4Result, _cache_load, _cache_save, run_condition

# The control is held at gain 1.0 throughout. Only the treatment is exaggerated -- the
# question is whether a *difference* of known size is visible, so scaling both would
# plant nothing.
CONTROL = Condition("C? yoked", INNATE,
                    cfg_patch=(("channel_mode", "yoked"),
                               ("call_log_steps", spec.YOKE_LOG_STEPS)),
                    pallium_scale=EXPANDED, scaffold=True)

GAINS = (1.0, 2.0, 4.0)


def treatment(gain: float) -> Condition:
    return Condition(f"L gain {gain:g}", INNATE,
                     cfg_patch=(("channel_mode", "intact"),),
                     pallium_scale=EXPANDED, scaffold=True, scaffold_gain=gain)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-offset", type=int, default=48,
                    help="fresh block; E028's ladder used 36-47")
    ap.add_argument("--hens", type=int, default=16)
    ap.add_argument("--hawk-period", type=float, default=20.0)
    ap.add_argument("--cache", default="scratchpad/e029_cache.json")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens,
                                     hawk_period_s=args.hawk_period)
    seconds = args.minutes * 60.0
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    cache = _cache_load(args.cache)
    t0 = time.perf_counter()

    print(f"E029 -- positive control on the intent-to-treat metric")
    print(f"{len(seeds)} seeds ({seeds[0]}-{seeds[-1]}), {args.minutes:.0f} min, "
          f"{args.hens} hens, hawk every {args.hawk_period:.0f} s, no plasticity")
    print("gain 1.0 is the real hen (E028 measured -0.029 +/- 0.020, t=1.42 here).")
    print("higher gains are an exaggerated bird, present only to test the metric.\n")

    def cells(cond):
        out = []
        for sd in seeds:
            k = f"{cond.name}|{sd}|{args.minutes}|{args.hens}|{args.hawk_period}"
            if k not in cache:
                cache[k] = list(run_condition(cond, sd, cfg, seconds))
                _cache_save(args.cache, cache)
            out.append(H4Result(*cache[k]))
        return out

    ctrl = cells(CONTROL)
    hdr = (f"{'planted effect':<18}{'caught/dive':>13}{'vs control':>12}"
           f"{'SE':>8}{'t':>7}{'detected?':>12}")
    print(hdr); print("-" * len(hdr))
    c_itt = jnp.array([r.caught_itt for r in ctrl])
    print(f"{'C? yoked (control)':<18}{float(jnp.mean(c_itt)):>13.3f}"
          f"{'--':>12}{'--':>8}{'--':>7}{'--':>12}")

    rows = []
    for g in GAINS:
        t_cells = cells(treatment(g))
        d = jnp.array([r.caught_itt for r in t_cells]) - c_itt
        n = len(seeds)
        mean = float(jnp.mean(d))
        se = float(jnp.std(d, ddof=1)) / (n ** 0.5)
        t = abs(mean) / (se + 1e-12)
        crit = _t_critical(n - 1)
        hit = "YES" if t > crit else "no"
        rows.append((g, mean, se, t, hit))
        print(f"{'L, gain ' + f'{g:g}':<18}"
              f"{float(jnp.mean(jnp.array([r.caught_itt for r in t_cells]))):>13.3f}"
              f"{mean:>12.3f}{se:>8.3f}{t:>7.2f}{hit:>12}")

    print(f"\n(significance threshold t={_t_critical(len(seeds) - 1):.3f} at "
          f"{len(seeds) - 1} df)\n")

    # The reading. Stated in the harness rather than left to the write-up, because the
    # whole point is that the interpretation was decided before the numbers arrived.
    detected = [g for g, _m, _se, _t, hit in rows if hit == "YES"]
    if not detected:
        print("NO planted effect was detected, including the largest.")
        print("=> the metric cannot see an effect of any size at this seed count, so")
        print("   E028's null is uninformative about H4. Fix the instrument or the")
        print("   sample before running the ladder again.")
    elif 1.0 in detected:
        print("The real hen's effect (gain 1.0) IS detected here.")
        print("=> disagrees with E028 on the same contrast; treat as a seed-block")
        print("   difference and reconcile before claiming anything.")
    else:
        smallest = min(detected)
        print(f"Detected from gain {smallest:g} upward, not at gain 1.0.")
        print("=> the metric works; E028's null is a POWER problem, not evidence of")
        print("   absence. The honest next step is more seeds, not a new hypothesis.")
    print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
