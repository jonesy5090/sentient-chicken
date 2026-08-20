"""Could E082/E083's occupancy metric resolve their own pre-registered 15% effect?

The coverage check found the flock aggregates and settles where it starts: occupancy of
a named feeder runs 0.000-0.481 across world keys with nothing else changed. E082 and
E083 are paired across gains on shared seeds, so between-seed variance partly cancels
and the raw spread is not the right number. The right number is the sd of the WITHIN-seed
difference between gain 0 and gain 2, which is what their paired comparison actually
rests on.

Runs E083's exact metric at the two ends of its ladder, 8 seeds -- double E083's block,
and a fresh-seed block on top of it, which the project requires before any status move.
"""
import sys, time
sys.path.insert(0, 'scratchpad')
import jax, jax.numpy as jnp, numpy as np
import e083_leaving_anchor as E
from coop import world
from hen import brain, connectome, plasticity, regions

SEEDS = 8
print(f"E083's metric at gain 0.0 vs 2.0, {SEEDS} seeds (E083 used 4), "
      f"{E.MINUTES:.0f} min\n")
t0 = time.perf_counter()
rows = []
for s in range(SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=E.HENS,
                          gakel_scaffold=True, shared_place_map=True)
    p, pf = E.plant(p0, E._PLANT_CFG)
    out = {}
    for gain in (0.0, 2.0):
        pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=gain)
        w = world.reset(k, E.CFG)
        w = w._replace(food_pos=jnp.asarray(np.stack([E.CEN[E.P], E.CEN[E.P2]]),
                                            dtype=jnp.float32))
        x = brain.initial_state(p, E.HENS); ps = plasticity.initial_state(p, E.HENS, pc)
        a, b, c, d, e, f = E.run(w, x, p, ps, jax.random.fold_in(k, 2), E.CFG, pc, E.STEPS)
        out[gain] = float(jnp.mean(a))
    rows.append((s, pf, out[0.0], out[2.0], out[2.0] - out[0.0]))
    print(f"  seed {s}: preflight {pf:.3f}  occ@gain0 {out[0.0]:.4f}  "
          f"occ@gain2 {out[2.0]:.4f}  diff {out[2.0]-out[0.0]:+.4f}")

A = np.array([r[2] for r in rows]); B = np.array([r[3] for r in rows])
D = B - A
print(f"\nbaseline occupancy: mean {A.mean():.4f}, sd {A.std(ddof=1):.4f} "
      f"(sd/mean {A.std(ddof=1)/A.mean():.2f})")
print(f"paired difference : mean {D.mean():+.4f}, sd {D.std(ddof=1):.4f}")

# minimum detectable effect, paired t, two-sided alpha=0.05
T = {4: 3.182, 8: 2.365}
print(f"\n{'n':>4}{'t crit':>9}{'min detectable diff':>22}{'as % of baseline':>19}")
for n in (4, 8):
    mde = T[n] * D.std(ddof=1) / np.sqrt(n)
    print(f"{n:>4}{T[n]:>9.3f}{mde:>22.4f}{100*mde/A.mean():>18.1f}%")
print(f"\nE082 and E083 both pre-registered a 15% effect at n=4.")
print(f"observed |diff| at n={SEEDS}: {abs(D.mean()):.4f} "
      f"= {100*abs(D.mean())/A.mean():.1f}% of baseline, "
      f"t={D.mean()/(D.std(ddof=1)/np.sqrt(SEEDS)):+.2f}")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
