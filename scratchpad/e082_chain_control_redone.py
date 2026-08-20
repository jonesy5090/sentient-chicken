"""E082: T2-revised's whole-chain positive control, with a DISCRIMINATIVE plant.

E070 planted a matched filter (18.8% at 'am I at P', below chance). E081 measured a
discriminant at 84.6% on identical states. This redoes the control with the latter,
plus E070's own two recorded design corrections: traces live / weights frozen, and food
at P and a matched control cell so occupancy has a baseline to fall from.
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
OTHERS = [c for c in (0, 6, 12, 18, 24) if c not in (P, P2)]

# food at P and at the matched control cell -- E070's occupancy metric had no baseline
CFG = CFG0._replace(n_food=2)


# --- planting -------------------------------------------------------------
# Settle long enough for z_lag_bar to converge (baseline_tau_s=20s) and plant against
# the SAME centred signal the runtime readout uses. A first version settled 300 steps
# and planted against raw z_lag; the bar was still ~0, so the plant was scaled against
# one signal and read back through another, and `pred@gakel` came out ~0.04 instead of
# ~1.0. Same timescale error E071 recorded and this repeated.
DWELL, LAPS, HOLD = 1000, 6, 1000        # 300 s tour = 15x baseline_tau_s, then parked
TOUR = [0, 6, 12, 18, 24]
OTHERS_FOR_DISC = [c for c in TOUR if c != P]


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
        return carry, (jnp.mean(dP < SPACING), jnp.mean(d2 < SPACING),
                      jnp.mean(w.hunger), jnp.mean(motor[:, spec.M_FORWARD]),
                      jnp.mean(jax.nn.relu(pred[:, GAKEL_CH])))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


FROZEN = dict(enabled=True, explore_sigma=0.0, eta=0.0, eta_out=0.0, eta_pred=0.0,
              scaling_strength=0.0, readout_scaling_strength=0.0,
              pred_enabled=True, pred_centred=True)

print(f"E082 -- discriminative plant, {SEEDS} seeds, {MINUTES:.0f} min, no learning")
print(f"planted P=cell {P} {CEN[P].round(1)}, control P'=cell {P2} {CEN[P2].round(1)}, "
      f"food at both, spacing {SPACING:.2f} m\n")
print(f"{'pred_gain':>10}{'occupancy P':>13}{'occupancy P2':>14}{'selectivity':>13}"
      f"{'hunger':>9}{'fwd':>8}{'pred@gakel':>12}")
t0 = time.perf_counter()
_PLANT_CFG = PlasticConfig(**FROZEN, pred_gain=1.0)
PLANTED = {}
for s_ in range(SEEDS):
    k = jax.random.key(s_)
    _p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=HENS,
                          gakel_scaffold=True, shared_place_map=True)
    PLANTED[s_] = plant(_p, _PLANT_CFG)
print("pre-flight -- predicted gakel at P per seed: "
      + ", ".join(f"{v[1]:.3f}" for v in PLANTED.values()) + "\n")

for gain in (0.0, 0.5, 1.0, 2.0):
    pc = PlasticConfig(**FROZEN, pred_gain=gain)
    oP, o2, hu, fw, pr = [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        w = world.reset(k, CFG)
        w = w._replace(food_pos=jnp.asarray(np.stack([CEN[P], CEN[P2]]), dtype=jnp.float32))
        p = PLANTED[s][0]
        x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
        a, b, c, dd, e = run(w, x, p, ps, jax.random.fold_in(k, 2), CFG, pc, STEPS)
        oP.append(float(jnp.mean(a))); o2.append(float(jnp.mean(b)))
        hu.append(float(jnp.mean(c))); fw.append(float(jnp.mean(dd))); pr.append(float(jnp.mean(e)))
    print(f"{gain:>10.1f}{np.mean(oP):>13.4f}{np.mean(o2):>14.4f}"
          f"{np.mean(o2)-np.mean(oP):>+13.4f}{np.mean(hu):>9.3f}{np.mean(fw):>8.3f}"
          f"{np.mean(pr):>12.4f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("avoidance = occupancy at P falling vs gain 0. selectivity = P2 holding up while P falls.")
print("hallucination = hunger climbing / fwd collapsing at high gain.")
