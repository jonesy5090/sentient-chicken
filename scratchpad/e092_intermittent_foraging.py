"""E092: do depletion and a locomotion gate produce intermittent foraging?

Sweeps `peck_stops_walking` with `food_deplete_rate` at its DEFAULT (the whole T2 arc had
it at 0.0). Measures the speed distribution, dwell per visit, flock spread and hunger,
against the figures E084/E085 recorded under continuous locomotion and infinite food.
"""
import sys, time
sys.path.insert(0, 'scratchpad')
from functools import partial
import jax, jax.numpy as jnp, numpy as np
import e083_leaving_anchor as E
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

HENS, SEEDS, MINUTES = 16, 8, 20.0
# depletion at its default -- the override the arc inherited from E082 is gone
BASE = spec.DEFAULT_COOP._replace(n_hens=HENS, place_cells_enabled=True,
                                  contamination_enabled=False, n_food=4)
STEPS = int(MINUTES * 60 / BASE.dt)
GRID = np.linspace(0.0, BASE.size, spec.PLACE_GRID + 2)[1:-1]
CEN = np.stack(np.meshgrid(GRID, GRID, indexing="ij"), -1).reshape(-1, 2)
CENJ = jnp.asarray(CEN, dtype=jnp.float32)
R = float(GRID[1] - GRID[0])


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        wl = carry[0]
        # Distance to the NEAREST FEEDER, not to the nearest grid cell. The cells tile
        # the arena, so "within R of the nearest cell" is trivially always true and the
        # first version of this measured dwell as the whole run at every setting.
        df = jnp.min(jnp.linalg.norm(wl.pos[:, None, :] - wl.food_pos[None, :, :],
                                     axis=-1), axis=-1)
        at_food = df < cfg.peck_radius * 3.0
        spread = jnp.mean(jnp.linalg.norm(wl.pos - jnp.mean(wl.pos, 0), axis=-1))
        return carry, (wl.speed, wl.hunger, at_food, spread, jnp.mean(wl.food_amount))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


def dwell(inside):
    runs = []
    a = np.asarray(inside)
    for h in range(a.shape[1]):
        d = np.diff(np.concatenate([[0], a[:, h].astype(np.int8), [0]]))
        runs.extend((np.where(d == -1)[0] - np.where(d == 1)[0]).tolist())
    return (np.mean(runs) * BASE.dt if runs else 0.0)


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0, pred_bar_freeze_s=60.0)
t0 = time.perf_counter()
print(f"E092 -- {SEEDS} seeds, {MINUTES:.0f} min, depletion ON "
      f"(rate {BASE.food_deplete_rate}), {BASE.n_food} feeders")
print("E084/E085 references (depletion OFF): dwell 17-75 s (279/717 on two seeds), "
      "spread 1.66-7.21 m, hunger 0.3997\n")
print(f"{'gate':>6}{'slow frac':>11}{'mean speed':>12}{'dwell (s)':>11}{'max dwell':>11}"
      f"{'spread (m)':>12}{'hunger':>8}{'food left':>11}"
      f"{'v@food':>10}{'v away':>10}")
for g in (0.0, 0.5, 0.8, 1.0):
    cfg = BASE._replace(peck_stops_walking=g)
    sp, hu, ins, spr, fa, dw, atf = [], [], [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                             n_hens=HENS, gakel_scaffold=True, shared_place_map=True,
                             place_to_hippocampus=True,
                             gakel_peck_weight=9.0, hunger_peck_weight=4.0)
        w = world.reset(k, cfg)
        x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
        speed, hunger, inside, spread, food = run(
            w, x, p, ps, jax.random.fold_in(k, 9), cfg, pc, STEPS)
        sp.append(np.asarray(speed)); hu.append(float(jnp.mean(hunger)))
        spr.append(float(jnp.mean(spread))); fa.append(float(jnp.mean(food)))
        dw.append(dwell(np.asarray(inside))); atf.append(np.asarray(inside))
    allsp = np.concatenate([a.ravel() for a in sp])
    slow = float((allsp < 0.2 * cfg.walk_speed).mean())
    # The discriminator: is she slow *at food* and normal elsewhere (intermittent), or
    # slow everywhere (just a slower hen)?
    at = np.concatenate([a.ravel() for a in atf])
    v_at = float(allsp[at].mean()) if at.any() else float("nan")
    v_off = float(allsp[~at].mean()) if (~at).any() else float("nan")
    print(f"{g:>6.1f}{slow:>11.3f}{allsp.mean():>12.4f}{np.mean(dw):>11.1f}"
          f"{max(dw):>11.1f}{np.mean(spr):>12.2f}{np.mean(hu):>8.4f}{np.mean(fa):>11.3f}"
          f"{v_at:>10.4f}{v_off:>10.4f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("bimodal foraging => slow frac > 0.3 (pre-registered); dwell capped; spread up")
