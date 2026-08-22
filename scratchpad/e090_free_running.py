"""E090's starvation falsifier: does a hunger term on M_PECK make hens worse at feeding?

Pre-registered: mean hunger in a free-running flock must not rise more than 0.1 above the
control. Also reports the hunger distribution, because the whole design rests on where the
flock actually sits between "sated" and "starving" -- E089 measured mean 0.33, and a
conditional response is worthless if the hens never occupy the range where it varies.
"""
import sys, time
sys.path.insert(0, 'scratchpad')
from functools import partial
import jax, jax.numpy as jnp, numpy as np
import e083_leaving_anchor as E
import e085_repaired_instrument as M
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG, HENS, STEPS = M.CFG, M.HENS, M.STEPS
SEEDS = 8
ARMS = ((1.5, 0.0), (9.0, 4.0), (9.0, 8.0), (7.0, 8.0))


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        return carry, carry[0].hunger
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0, pred_bar_freeze_s=60.0)
t0 = time.perf_counter()
print(f"free-running flock, {SEEDS} seeds, {E.MINUTES:.0f} min, no plant\n")
print(f"{'W':>5}{'H':>5}{'mean hunger':>13}{'p10':>8}{'p50':>8}{'p90':>8}{'vs control':>12}")
base = None
for W, H in ARMS:
    hs = []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                             n_hens=HENS, gakel_scaffold=True, shared_place_map=True,
                             place_to_hippocampus=True,
                             gakel_peck_weight=W, hunger_peck_weight=H)
        w = world.reset(k, CFG)
        x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
        hs.append(np.asarray(run(w, x, p, ps, jax.random.fold_in(k, 9), CFG, pc, STEPS)))
    a = np.concatenate([h.ravel() for h in hs])
    m = a.mean()
    if base is None:
        base = m
    print(f"{W:>5.1f}{H:>5.1f}{m:>13.4f}{np.percentile(a,10):>8.3f}"
          f"{np.percentile(a,50):>8.3f}{np.percentile(a,90):>8.3f}{m-base:>+12.4f}")
print(f"\nstarvation falsifier: fires if any arm exceeds control by >0.10")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
