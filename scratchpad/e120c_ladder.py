"""E120 -- the H0 ladder. Evolve an intact channel against a yoked one.

The experiment `run/evolve.py` was built for, and the first route this project has to H5.
It runs at `hawk_period_s=20` because that is the only density of three tested where all
three pre-checks pass at once (E120 6a-6c):

  * predation is heritable enough for selection to see it   (r(caught) = +0.179)
  * the alarm channel carries news                          (r(channel) = +0.194)
  * a hand-planted comprehension actually saves hens        (-1.195 catches, t=-2.82)

The last is the mandatory positive control. At 10 s and 50 s it is null, and a ladder run
at either would have produced an uninterpretable null.

Founders carry NO auditory scaffold. The whole question is whether selection can discover
comprehension in `W`/`W_out`, which is the only substrate mutation touches -- the reflex
arc is never plastic and is never mutated.

Run:  PYTHONPATH=. python scratchpad/e120c_ladder.py [calibrate|run]
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, plasticity
from run import evolve, simulate
from scratchpad.e120_ladder_checks import founders

HAWK = 20.0
LIFETIME_S = 600.0
GENERATIONS = 16
N_LINEAGES = 4
BLOCK = int(os.environ.get("E120_BLOCK", "1"))
LIN_BASE = (BLOCK - 1) * N_LINEAGES
CACHE = f"scratchpad/e120c_cache{'' if BLOCK == 1 else f'_b{BLOCK}'}.json"

# E119's measured repeatabilities at this density.
R_CAUGHT, R_DRIVES = 0.179, 0.717


def cfg_for(mode):
    """Both arms carry the yoked control's call-history ring buffer.

    Only the yoked arm reads it, but allocating it in both keeps the two arms on exactly
    the same code path and the same memory traffic, so the contrast is the channel's
    *content* and nothing else. `spec.py` notes the buffer is 512 KB in the scan carry and
    cost a measured throughput regression -- which is precisely why leaving it out of one
    arm would be a difference between the arms rather than a saving.
    """
    return spec.DEFAULT_COOP._replace(hawk_period_s=HAWK, channel_mode=mode,
                                      call_log_steps=spec.YOKE_LOG_STEPS)


def calibrate(n_seeds=8):
    """Pick `caught_weight` by equalising the two terms' REPEATABLE variance.

    E116 chose 0.10 to equalise the terms' raw spreads. E117/E118 showed that is the wrong
    principle: a term's spread counts for nothing if none of it is repeatable, and
    weighting by spread is how three-quarters of the selection differential ended up on
    luck. Equalising `sd * sqrt(r)` instead weights each term by the part of it that
    carries information about the hen.
    """
    cfg = cfg_for("intact")
    pc = evolve.NO_LEARNING
    sd_c, sd_d = [], []
    for g in range(n_seeds):
        key = jax.random.key(4000 + g)
        p = founders(jax.random.fold_in(key, 0), cfg)
        w0 = world.reset(jax.random.fold_in(key, 1), cfg)
        x0 = brain.initial_state(p, cfg.n_hens)
        ps = plasticity.initial_state(p, cfg.n_hens, pc)
        w_end, *_ = simulate.rollout_quiet(
            w0, x0, p, jax.random.fold_in(key, 2), cfg,
            int(LIFETIME_S / cfg.dt), ps, pc)
        sd_c.append(float(jnp.std(w_end.n_caught_any)))
        sd_d.append(float(jnp.std(w_end.hunger + w_end.cold + w_end.thirst)))
    sd_c, sd_d = float(np.mean(sd_c)), float(np.mean(sd_d))
    w = (sd_d * np.sqrt(R_DRIVES)) / (sd_c * np.sqrt(R_CAUGHT) + 1e-12)
    print(f"  sd(caught)={sd_c:.4f}  sd(drives)={sd_d:.4f}")
    print(f"  repeatable sd: caught {sd_c*np.sqrt(R_CAUGHT):.4f}  "
          f"drives {sd_d*np.sqrt(R_DRIVES):.4f}")
    print(f"  -> caught_weight = {w:.4f}   (E116 used 0.10 by equalising raw spreads)")
    return w, sd_c, sd_d


def assay(p, cfg, key, lifetime_s=LIFETIME_S):
    """Score an evolved flock: catches, drives, and the motor profile."""
    pc = evolve.NO_LEARNING
    w0 = world.reset(key, cfg)
    x0 = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    w_end, *_ = simulate.rollout_quiet(
        w0, x0, p, jax.random.fold_in(key, 3), cfg, int(lifetime_s / cfg.dt), ps, pc)
    dives = float(jnp.sum(w_end.n_dives))
    return dict(caught=float(jnp.mean(w_end.n_caught_any)),
                per_dive=float(jnp.sum(w_end.n_caught_any)) / max(dives, 1.0),
                hunger=float(jnp.mean(w_end.hunger)))


def run(caught_weight):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    evo = evolve.EvoConfig(generations=GENERATIONS, lifetime_s=LIFETIME_S,
                           founder_sigma=2.0, caught_weight=caught_weight)
    for mode in ("intact", "yoked"):
        if mode in cache:
            print(f"[{mode}] cached")
            continue
        cfg = cfg_for(mode)
        rows = []
        print(f"[{mode}]", flush=True)
        for lin in range(N_LINEAGES):
            t0 = time.time()
            key = jax.random.key(9000 + LIN_BASE + lin)
            hist, p_final = evolve.run_lineage(key, cfg, evo)
            caught = [float(h["caught"]) for h in hist]
            hung = [float(h["hunger"]) for h in hist]
            # Score the evolved flock on the INTACT world in both arms, so the final
            # assay measures the brain rather than the channel it is being tested under.
            final_intact = assay(p_final, cfg_for("intact"),
                                 jax.random.fold_in(key, 555))
            rows.append(dict(lineage=LIN_BASE + lin, caught=caught, hunger=hung,
                             final_intact=final_intact,
                             improve_caught=caught[0] - caught[-1],
                             improve_hunger=hung[0] - hung[-1],
                             secs=time.time() - t0))
            print(f"    lin {LIN_BASE+lin}: caught/dive {caught[0]:.4f} -> "
                  f"{caught[-1]:.4f} ({caught[0]-caught[-1]:+.4f})   "
                  f"hunger {hung[0]:.4f} -> {hung[-1]:.4f}   "
                  f"({time.time()-t0:.0f}s)", flush=True)
        cache[mode] = rows
        cache["caught_weight"] = caught_weight
        json.dump(cache, open(CACHE, "w"), indent=1)
    summarise(cache)


def summarise(cache):
    print("\n" + "=" * 86)
    print("H0 ladder: does an intact channel beat a yoked one across generations?")
    print("=" * 86)
    a = np.array([r["improve_caught"] for r in cache["intact"]])
    b = np.array([r["improve_caught"] for r in cache["yoked"]])
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  reduction in catches/dive over {GENERATIONS} generations")
    print(f"    intact {a.mean():+.5f} +- {a.std(ddof=1)/np.sqrt(len(a)):.5f}")
    print(f"    yoked  {b.mean():+.5f} +- {b.std(ddof=1)/np.sqrt(len(b)):.5f}")
    print(f"    intact - yoked  {d.mean():+.5f} +- {se:.5f}  t={d.mean()/se:+.2f}  "
          f"(df={len(d)-1}, bar 3.182)")
    ha = np.array([r["improve_hunger"] for r in cache["intact"]])
    hb = np.array([r["improve_hunger"] for r in cache["yoked"]])
    hd = ha - hb
    hse = hd.std(ddof=1) / np.sqrt(len(hd))
    print(f"\n  hunger improvement (secondary; the channel should NOT help here)")
    print(f"    intact {ha.mean():+.5f}   yoked {hb.mean():+.5f}   "
          f"diff {hd.mean():+.5f} +- {hse:.5f} t={hd.mean()/hse:+.2f}")
    print("\n  final flocks re-assayed on an INTACT world (brain, not channel):")
    for mode in ("intact", "yoked"):
        f = [r["final_intact"] for r in cache[mode]]
        print(f"    reared {mode:>7}: caught/hen {np.mean([x['caught'] for x in f]):.3f}  "
              f"caught/dive {np.mean([x['per_dive'] for x in f]):.4f}  "
              f"hunger {np.mean([x['hunger'] for x in f]):.4f}")
    print("\n  reference: a hand-planted comprehension (auditory_scaffold) is worth")
    print("  -1.195 catches/hen at this density (E120 6c). That is the ceiling.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "calibrate":
        calibrate()
    else:
        wt = float(sys.argv[2]) if len(sys.argv) > 2 else None
        if wt is None:
            print("calibrating caught_weight first:")
            wt, _, _ = calibrate()
        run(wt)
