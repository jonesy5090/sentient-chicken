"""E112: does repairing the peck reflex close E111's 0.21-hunger gap?

Reference ceiling, from E111 and not re-run here: camped oracle hunger 0.4223.
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
OFFSET = int(os.environ.get("E112_SEED_OFFSET", "0"))     # E033's lesson
reg = regions.DEFAULT_REGIONS
FROZEN = plasticity.PlasticConfig(enabled=True, eta_out=0.0)
LEARN = plasticity.PlasticConfig(enabled=True)
ORACLE = 0.4223          # E111, camped oracle, both seed blocks

ARMS = (("baseline / frozen", 0.0, FROZEN),
        ("REPAIRED / frozen", 4.0, FROZEN),
        ("baseline / learning", 0.0, LEARN),
        ("REPAIRED / learning", 4.0, LEARN))


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
print(f"E112 -- repair the peck reflex. seeds {OFFSET}-{OFFSET+SEEDS-1}, "
      f"{STEPS*BASE.dt/60:.0f} min\n")
print(f"E111 reference: reflex hen 0.6332, camped oracle {ORACLE:.4f}, "
      f"P(peck|on food) 39.65% vs P(peck|off) 59.59%\n")

res = {}
for label, wgt, pc in ARMS:
    h, c, p_on, p_off, at_f, fed_f = [], [], [], [], [], []
    for s in range(OFFSET, OFFSET + SEEDS):
        k = jax.random.key(s)
        p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=BASE.n_hens,
                             arrival_peck_weight=wgt)
        w2, out = go(world.reset(k, BASE), brain.initial_state(p, BASE.n_hens), p,
                     plasticity.initial_state(p, BASE.n_hens, pc),
                     jax.random.fold_in(k, 2), BASE, pc, STEPS)
        at, peck, fed = (np.asarray(a).ravel() for a in out)
        h.append(float(np.mean(np.asarray(w2.hunger))))
        c.append(float(np.sum(np.asarray(w2.n_caught_any))
                       / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
        p_on.append(float(peck[at].mean()))
        p_off.append(float(peck[~at].mean()))
        at_f.append(float(at.mean()))
        fed_f.append(float(fed.mean()))
    res[label] = dict(h=np.array(h), c=np.array(c), on=np.mean(p_on),
                      off=np.mean(p_off), at=np.mean(at_f), fed=np.mean(fed_f))
    print(f"{label:>21}  hunger {np.mean(h):.4f}   P(peck|on) {100*np.mean(p_on):>5.1f}%"
          f"   P(peck|off) {100*np.mean(p_off):>5.1f}%   at patch {100*np.mean(at_f):>4.1f}%"
          f"   feeding {100*np.mean(fed_f):>4.1f}%   caught/dive {np.mean(c):.4f}")


def paired(a, b, name):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / (se + 1e-12)
    print(f"    {name:<52}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}")
    return d.mean(), se


print(f"\n  paired, df={SEEDS-1}, crit 2.365 (lower hunger is better):")
rep, _ = paired(res["REPAIRED / frozen"]["h"], res["baseline / frozen"]["h"],
                "the repair, no learning  <-- the headline")
paired(res["baseline / learning"]["h"], res["baseline / frozen"]["h"],
       "learning on the OLD arc")
paired(res["REPAIRED / learning"]["h"], res["REPAIRED / frozen"]["h"],
       "learning on the REPAIRED arc")
paired(res["REPAIRED / frozen"]["c"], res["baseline / frozen"]["c"],
       "the repair, caught/dive")

base_h = res["baseline / frozen"]["h"].mean()
rep_h = res["REPAIRED / frozen"]["h"].mean()
gap0 = base_h - ORACLE
gap1 = rep_h - ORACLE
print(f"\n  headroom against the camped oracle ({ORACLE:.4f}):")
print(f"    before repair {gap0:+.4f}   after repair {gap1:+.4f}   "
      f"closed {100*(1 - gap1/gap0) if gap0 else float('nan'):.0f}% of the gap")

print("\n--- pre-registered falsifiers (E112 section 4) ---")
on = 100 * res["REPAIRED / frozen"]["on"]
print(f"instrument   P(peck|on food) repaired = {on:.1f}% (must exceed 80%)"
      f"{'  <-- FIRES, nothing here means anything' if on <= 80 else '  clear'}")
print(f"             and the sign must reverse: on {on:.1f}% vs off "
      f"{100*res['REPAIRED / frozen']['off']:.1f}%")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
