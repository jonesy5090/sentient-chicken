"""E083: does the redesigned gakel anchor produce LEAVING rather than STOPPING?

Forked from `e082_chain_control_redone.py` and matched to it verbatim -- same seeds,
same discriminative plant, same gain ladder, same arena, same tour-settle. The only
difference upstream is `hen/innate.py`: `_add_gakel_scaffold` no longer touches
`M_FORWARD`, only `M_PECK`.

E082's chain conducted (fwd fell 17%) but occupancy at the bad feeder did not move,
because `actuation.py` derives speed from `M_FORWARD` -- a hen already at P who slows
down stays at P. Prediction here: she keeps walking, declines to eat, stays hungry,
and hunger drives forward harder, so she leaves.

One column added over E082: peck rate *among hens standing at P*, which is now the
direct read on the reflex firing where it matters.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

HENS, SEEDS, MINUTES = 16, 4, 20.0
CFG0 = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0,
                                  place_cells_enabled=True,
                                  contamination_enabled=False)
STEPS = int(MINUTES * 60 / CFG0.dt)
GAKEL_CH = spec.AUDIO_LO + spec.GAKEL_CALL_IDX
E = np.linspace(0.0, CFG0.size, spec.PLACE_GRID + 2)[1:-1]
CEN = np.stack(np.meshgrid(E, E, indexing="ij"), -1).reshape(-1, 2)
SPACING = float(E[1] - E[0])

CENTRE = np.array([CFG0.size / 2] * 2)
d = np.linalg.norm(CEN - CENTRE, axis=1)
order = np.argsort(np.abs(d - d.mean()))
P, P2 = int(order[0]), int(order[1])

CFG = CFG0._replace(n_food=2)

DWELL, LAPS, HOLD = 1000, 6, 1000        # 300 s tour = 15x baseline_tau_s, then parked
TOUR = [0, 6, 12, 18, 24]
OTHERS_FOR_DISC = [c for c in TOUR if c != P]
PREFLIGHT_MIN = 0.99


@partial(jax.jit, static_argnames=("cfg", "pc"))
def _settle(w, x, ps, p, positions, cfg, pc):
    def step(c, pos):
        w, x, ps = c
        w = w._replace(pos=jnp.broadcast_to(pos, (cfg.n_hens, 2)),
                       heading=jnp.zeros((cfg.n_hens,)))
        obs = sensing.observe(w, cfg)
        x, motor, _d = brain.step(x, obs, p, cfg.dt)
        ps = plasticity.update_traces(ps, neurons.rate(x), motor,
                                      jnp.zeros((cfg.n_hens,)), cfg, pc)
        return (w, x, ps), None
    (w, x, ps), _ = jax.lax.scan(step, (w, x, ps), positions)
    return ps


def _positions(cell):
    seq = []
    for _ in range(LAPS):
        for c in TOUR:
            seq += [c] * DWELL
    seq += [cell] * HOLD
    return jnp.asarray(CEN[np.asarray(seq)], dtype=jnp.float32)


def _centred(p, cell, pc):
    w = world.reset(jax.random.key(0), CFG)
    x = brain.initial_state(p, HENS)
    ps = plasticity.initial_state(p, HENS, pc)
    ps = _settle(w, x, ps, p, _positions(cell), CFG, pc)
    return np.asarray((ps.z_lag - ps.z_lag_bar)[0])


def plant(p, pc):
    """Discriminant: mean(P) - mean(elsewhere), on the centred signal W_pred reads."""
    sP = _centred(p, P, pc)
    sO = np.mean([_centred(p, c, pc) for c in OTHERS_FOR_DISC], axis=0)
    disc = (sP - sO) * np.asarray(p.pred_src)
    disc = disc / (float(disc @ sP) + 1e-9)          # prediction ~1.0 at P
    w_pred = np.zeros_like(np.asarray(p.W_pred))
    w_pred[:, GAKEL_CH, :] = disc
    return p._replace(W_pred=jnp.asarray(w_pred)), float(disc @ sP)


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w, x = carry[0], carry[1]
        dP = jnp.linalg.norm(w.pos - jnp.asarray(CEN[P], dtype=jnp.float32), axis=-1)
        d2 = jnp.linalg.norm(w.pos - jnp.asarray(CEN[P2], dtype=jnp.float32), axis=-1)
        pred = jnp.einsum("hon,hn->ho", p.W_pred,
                          (carry[3].z_lag - carry[3].z_lag_bar) * p.pred_src[None, :])
        at_P = dP < SPACING
        # peck among hens standing at P; sum/count so hens elsewhere do not dilute it
        peck_P = jnp.sum(motor[:, spec.M_PECK] * at_P) / (jnp.sum(at_P) + 1e-9)
        return carry, (jnp.mean(at_P), jnp.mean(d2 < SPACING),
                      jnp.mean(w.hunger), jnp.mean(motor[:, spec.M_FORWARD]),
                      peck_P, jnp.mean(jax.nn.relu(pred[:, GAKEL_CH])))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


FROZEN = dict(enabled=True, explore_sigma=0.0, eta=0.0, eta_out=0.0, eta_pred=0.0,
              scaling_strength=0.0, readout_scaling_strength=0.0,
              pred_enabled=True, pred_centred=True)

print(f"E083 -- M_PECK-only anchor, {SEEDS} seeds, {MINUTES:.0f} min, no learning")
print(f"planted P=cell {P} {CEN[P].round(1)}, control P'=cell {P2} {CEN[P2].round(1)}, "
      f"food at both, spacing {SPACING:.2f} m\n")
t0 = time.perf_counter()
_PLANT_CFG = PlasticConfig(**FROZEN, pred_gain=1.0)
PLANTED = {}
for s_ in range(SEEDS):
    k = jax.random.key(s_)
    _p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=HENS,
                          gakel_scaffold=True, shared_place_map=True)
    PLANTED[s_] = plant(_p, _PLANT_CFG)
print("pre-flight -- predicted gakel at P per seed: "
      + ", ".join(f"{v[1]:.3f}" for v in PLANTED.values()))
_worst = min(v[1] for v in PLANTED.values())
assert _worst >= PREFLIGHT_MIN, (
    f"PRE-FLIGHT FAILED: plant reads {_worst:.4f} at P, need >={PREFLIGHT_MIN}. "
    "The plant is not firing -- do not interpret anything downstream of this. "
    "Check that z_lag_bar has converged (E071) and that the plant is built against "
    "the same centred signal the runtime reads (E082).")
print("pre-flight OK\n")

print(f"{'pred_gain':>10}{'occupancy P':>13}{'occupancy P2':>14}{'hunger':>9}"
      f"{'fwd':>8}{'peck@P':>9}{'pred@gakel':>12}")
ROWS = {}
for gain in (0.0, 0.5, 1.0, 2.0):
    pc = PlasticConfig(**FROZEN, pred_gain=gain)
    oP, o2, hu, fw, pk, pr = [], [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        w = world.reset(k, CFG)
        w = w._replace(food_pos=jnp.asarray(np.stack([CEN[P], CEN[P2]]), dtype=jnp.float32))
        p = PLANTED[s][0]
        x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
        a, b, c, dd, e, f = run(w, x, p, ps, jax.random.fold_in(k, 2), CFG, pc, STEPS)
        oP.append(float(jnp.mean(a))); o2.append(float(jnp.mean(b)))
        hu.append(float(jnp.mean(c))); fw.append(float(jnp.mean(dd)))
        pk.append(float(jnp.mean(e))); pr.append(float(jnp.mean(f)))
    ROWS[gain] = (np.mean(oP), np.mean(o2), np.mean(hu), np.mean(fw), np.mean(pk), np.mean(pr))
    print(f"{gain:>10.1f}{np.mean(oP):>13.4f}{np.mean(o2):>14.4f}{np.mean(hu):>9.3f}"
          f"{np.mean(fw):>8.3f}{np.mean(pk):>9.3f}{np.mean(pr):>12.4f}")

b0, c0 = ROWS[0.0][0], ROWS[0.0][1]
b2, c2 = ROWS[2.0][0], ROWS[2.0][1]
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifiers (E083 section 4) ---")
mono = all(ROWS[g][0] >= ROWS[h][0] for g, h in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0)))
print(f"primary   occupancy P {b0:.4f} -> {b2:.4f} = {100*(b2-b0)/b0:+.1f}% "
      f"(need <=-15%, monotonic={mono}) -> "
      f"{'PASS' if (b2-b0)/b0 <= -0.15 and mono else 'FIRES'}")
print(f"agitation occupancy P' {c0:.4f} -> {c2:.4f} = {100*(c2-c0)/c0:+.1f}% "
      f"(fires if <=-10%) -> {'FIRES' if (c2-c0)/c0 <= -0.10 else 'clear'}")
print(f"starve    hunger at gain 2.0 = {ROWS[2.0][2]:.3f} "
      f"(fires if >0.60) -> {'FIRES' if ROWS[2.0][2] > 0.60 else 'clear'}")
print(f"reflex    live pred@gakel at gain 2.0 = {ROWS[2.0][5]:.3f} "
      f"(fires if <0.80) -> {'FIRES' if ROWS[2.0][5] < 0.80 else 'clear'}")
