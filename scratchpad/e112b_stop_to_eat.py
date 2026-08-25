"""E112b (post-hoc diagnostic, NOT pre-registered): why the repair only got 27%.

E112's instrument falsifier fired: P(peck | on food) reached 75.3% against a bar of 80%.
The wiring works -- 41.8% -> 75.3%, and the sign reversed -- so the question is what caps
it. Measured: **median dwell at a patch is one step (0.01 s)**. Half of all on-food
episodes are a single step, and the arrival pulse is set by `world.step` *after* that
step's motor output was computed, so on a one-step visit she cannot peck at all.

She walks through the feeder. `peck_stops_walking` (E092) exists for exactly this and has
been off by default since it was built. E111's camped oracle *stayed* on its patch, which
is the other half of what it did.

So this measures whether the remaining gap is reachable by innate wiring at all. It is
labelled post-hoc because the arm was chosen after seeing E112's result, and it runs on a
disjoint seed block (8-15), which also replicates E112's borderline headline (t=-2.28).
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
STEPS, SEEDS = int(30 * 60 / BASE.dt), 8
OFFSET = int(os.environ.get("E112B_SEED_OFFSET", "8"))
reg = regions.DEFAULT_REGIONS
FROZEN = plasticity.PlasticConfig(enabled=True, eta_out=0.0)
LEARN = plasticity.PlasticConfig(enabled=True)
ORACLE = 0.4223

ARMS = (("baseline", 0.0, 0.0, FROZEN),
        ("repaired peck", 4.0, 0.0, FROZEN),
        ("repaired + STOPS", 4.0, 1.0, FROZEN),
        ("repaired + STOPS + learning", 4.0, 1.0, LEARN))


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def go(w, x, p, ps, key, cfg, pc, n):
    def step(c, _):
        w = c[0]
        d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
        at = jnp.any((d < cfg.peck_radius) & (w.food_amount[None, :] > 0.01), axis=-1)
        c, out = simulate._one_step(c, None, cfg, pc)
        peck = out[0][:, spec.M_PECK] > 0.5
        return c, (at, peck, at & peck)
    (w, x, p, ps, key), out = jax.lax.scan(step, (w, x, p, ps, key), None, length=n)
    return w, out


t0 = time.perf_counter()
print(f"E112b -- post-hoc diagnostic. seeds {OFFSET}-{OFFSET+SEEDS-1}\n")
print(f"E111/E112 reference: baseline 0.6332, camped oracle {ORACLE:.4f}, "
      f"E112's repair 0.5761 (t=-2.28, seeds 0-7)\n")

res = {}
for label, wgt, stops, pc in ARMS:
    cfg = BASE._replace(peck_stops_walking=stops)
    h, c, p_on, at_f, fed_f = [], [], [], [], []
    for s in range(OFFSET, OFFSET + SEEDS):
        k = jax.random.key(s)
        p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=cfg.n_hens,
                             arrival_peck_weight=wgt)
        w2, out = go(world.reset(k, cfg), brain.initial_state(p, cfg.n_hens), p,
                     plasticity.initial_state(p, cfg.n_hens, pc),
                     jax.random.fold_in(k, 2), cfg, pc, STEPS)
        at, peck, fed = (np.asarray(a).ravel() for a in out)
        h.append(float(np.mean(np.asarray(w2.hunger))))
        c.append(float(np.sum(np.asarray(w2.n_caught_any))
                       / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
        p_on.append(float(peck[at].mean()))
        at_f.append(float(at.mean()))
        fed_f.append(float(fed.mean()))
    res[label] = dict(h=np.array(h), c=np.array(c), on=np.mean(p_on),
                      at=np.mean(at_f), fed=np.mean(fed_f))
    print(f"{label:>28}  hunger {np.mean(h):.4f}   P(peck|on) {100*np.mean(p_on):>5.1f}%"
          f"   at patch {100*np.mean(at_f):>4.1f}%   feeding {100*np.mean(fed_f):>4.1f}%"
          f"   caught/dive {np.mean(c):.4f}")


def paired(a, b, name):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"    {name:<50}{d.mean():+.4f} +/- {se:.4f}  t={d.mean()/(se+1e-12):+.2f}")


print(f"\n  paired, df={SEEDS-1}, crit 2.365 (lower hunger is better):")
paired(res["repaired peck"]["h"], res["baseline"]["h"],
       "repair alone vs baseline  <-- replicates E112's -0.0571")
paired(res["repaired + STOPS"]["h"], res["baseline"]["h"],
       "repair + stopping vs baseline")
paired(res["repaired + STOPS"]["h"], res["repaired peck"]["h"],
       "what stopping adds on top of the repair")
paired(res["repaired + STOPS + learning"]["h"], res["repaired + STOPS"]["h"],
       "what LEARNING adds on top of both")

b = res["baseline"]["h"].mean()
for label in ("repaired peck", "repaired + STOPS"):
    g0, g1 = b - ORACLE, res[label]["h"].mean() - ORACLE
    print(f"\n  {label}: hunger {res[label]['h'].mean():.4f}, "
          f"closed {100*(1-g1/g0):.0f}% of the gap to the oracle")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
