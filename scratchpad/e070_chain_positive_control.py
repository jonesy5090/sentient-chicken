"""E070: T2-revised whole-chain positive control.

Hand-plants a W_pred association (place P -> gakel channel), enables pred_gain, and
asks whether a hen avoids P -- with NO learning anywhere. If a planted success is
undetectable, nothing learned can be.

Three things measured, not one: avoidance at P (does it work), occupancy at a matched
control cell (is it referential, or does she just avoid everywhere), and food
intake/hunger (does over-prediction make her hallucinate danger and starve).
"""
import time
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

HENS, SEEDS = 16, 4
MINUTES = 20.0
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
STEPS = int(MINUTES * 60 / CFG.dt)
FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
GAKEL_CH = spec.AUDIO_LO + spec.GAKEL_CALL_IDX

# Grid geometry, mirroring sensing._place_cells exactly.
EDGES = np.linspace(0.0, CFG.size, spec.PLACE_GRID + 2)[1:-1]
CENTRES = np.stack(np.meshgrid(EDGES, EDGES, indexing="ij"), -1).reshape(-1, 2)
SPACING = float(EDGES[1] - EDGES[0])

# P and a matched control cell P': both off-centre by a similar amount, so occupancy
# at each is comparable a priori and any difference is attributable to the planting.
CENTRE = np.array([CFG.size / 2, CFG.size / 2])
d_from_centre = np.linalg.norm(CENTRES - CENTRE, axis=1)
order = np.argsort(np.abs(d_from_centre - d_from_centre.mean()))
P, P_CTRL = int(order[0]), int(order[1])


def plant(p, cfg):
    """Write place-P -> gakel into W_pred by hand. Nothing else is nonzero."""
    # Pallial state evoked by standing at P. Settle the network on that observation so
    # the planted weights are keyed to the representation the hen will actually have
    # there, not to a single-step transient.
    w = world.reset(jax.random.key(0), cfg)
    pos = jnp.broadcast_to(jnp.asarray(CENTRES[P], dtype=jnp.float32), (cfg.n_hens, 2))
    w = w._replace(pos=pos, heading=jnp.zeros((cfg.n_hens,)))
    x = brain.initial_state(p, cfg.n_hens)
    for _ in range(200):
        x, _m, _d = brain.step(x, sensing.observe(w, cfg), p, cfg.dt)
    r = neurons.rate(x)[0] * np.asarray(p.pred_src)       # pallium only sources
    r = r / (np.linalg.norm(r) + 1e-9)

    w_pred = np.zeros_like(np.asarray(p.W_pred))
    # Scaled so the prediction lands near 1.0 for this pattern: <r_hat, r> = |r|.
    scale = 1.0 / (float(np.linalg.norm(neurons.rate(x)[0] * np.asarray(p.pred_src))) + 1e-9)
    w_pred[:, GAKEL_CH, :] = r * scale
    return p._replace(W_pred=jnp.asarray(w_pred))


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w = carry[0]
        d_P = jnp.linalg.norm(w.pos - jnp.asarray(CENTRES[P], dtype=jnp.float32), axis=-1)
        d_C = jnp.linalg.norm(w.pos - jnp.asarray(CENTRES[P_CTRL], dtype=jnp.float32), axis=-1)
        return carry, (jnp.mean(d_P < SPACING), jnp.mean(d_C < SPACING),
                      jnp.mean(w.hunger), jnp.mean(motor[:, spec.M_FORWARD]),
                      jnp.mean(obs[:, GAKEL_CH]))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


print(f"E070 -- whole-chain positive control, {SEEDS} seeds, {MINUTES:.0f} min, "
      f"no learning")
print(f"planted place P = cell {P} at {CENTRES[P].round(2)}, "
      f"matched control P' = cell {P_CTRL} at {CENTRES[P_CTRL].round(2)}, "
      f"grid spacing {SPACING:.2f} m\n")
print(f"{'pred_gain':>10}{'occupancy P':>14}{'occupancy P(ctrl)':>19}"
      f"{'selectivity':>13}{'hunger':>9}{'fed':>10}{'fwd':>8}{'gakel heard':>13}")

t0 = time.perf_counter()
for gain in (0.0, 0.5, 1.0, 2.0):
    occ_p, occ_c, hun, fwd, heard, fed = [], [], [], [], [], []
    for seed in range(SEEDS):
        key = jax.random.key(seed)
        cfg = CFG
        w = world.reset(key, cfg)
        p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                             n_hens=HENS, gakel_scaffold=True, shared_place_map=True)
        p = plant(p, cfg)
        pc = FIXED._replace(pred_enabled=True, pred_gain=gain)
        x = brain.initial_state(p, HENS)
        ps = plasticity.initial_state(p, HENS, pc)
        oP, oC, h, f, g = run(w, x, p, ps, jax.random.fold_in(key, 2), cfg, pc, STEPS)
        occ_p.append(float(jnp.mean(oP))); occ_c.append(float(jnp.mean(oC)))
        hun.append(float(jnp.mean(h))); fwd.append(float(jnp.mean(f)))
        heard.append(float(jnp.mean(g)))
    sel = np.mean(occ_c) - np.mean(occ_p)
    print(f"{gain:>10.1f}{np.mean(occ_p):>14.4f}{np.mean(occ_c):>19.4f}"
          f"{sel:>+13.4f}{np.mean(hun):>9.3f}{'':>10}{np.mean(fwd):>8.3f}"
          f"{np.mean(heard):>13.4f}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
print("Reading it: avoidance = occupancy at P falling below control-gain baseline.")
print("Selectivity = P' occupancy holding up while P's falls (positive column).")
print("Hallucination = hunger climbing and forward drive collapsing at high gain.")
