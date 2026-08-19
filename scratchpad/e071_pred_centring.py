"""E071: does centring the prediction source restore place selectivity?

Repeats E070's planted-association measurement with `pred_centred` off vs on.
Timescales matter and E070's own follow-up got them wrong first time: `z_lag` has
tau_lag=1.5s and `z_lag_bar` follows baseline_tau_s=20s, so a few seconds of settling
leaves the mean at ~0 and "centred" is indistinguishable from raw. The tour below runs
minutes, and convergence is reported rather than assumed.
"""
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from hen.plasticity import PlasticConfig

CFG = spec.DEFAULT_COOP._replace(n_hens=2, food_deplete_rate=0.0)
GAKEL_CH = spec.AUDIO_LO + spec.GAKEL_CALL_IDX
EDGES = np.linspace(0.0, CFG.size, spec.PLACE_GRID + 2)[1:-1]
CENTRES = np.stack(np.meshgrid(EDGES, EDGES, indexing="ij"), -1).reshape(-1, 2)
TOUR = [0, 6, 12, 18, 24]
DWELL = 1000          # 10 s per place -- well beyond tau_lag=1.5 s
LAPS = 6              # 5 places x 10 s x 6 = 300 s, i.e. 15x baseline_tau_s
HOLD = 1000           # 10 s parked at the test place before reading

p0 = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=2,
                      gakel_scaffold=True, shared_place_map=True)
PC = PlasticConfig(enabled=True, explore_sigma=0.0, eta=0.0, eta_out=0.0, eta_pred=0.0,
                   scaling_strength=0.0, readout_scaling_strength=0.0,
                   pred_enabled=True, pred_gain=1.0)


def _positions(cell):
    seq = []
    for _ in range(LAPS):
        for c in TOUR:
            seq += [c] * DWELL
    seq += [cell] * HOLD
    return jnp.asarray(CENTRES[np.asarray(seq)], dtype=jnp.float32)


@partial(jax.jit, static_argnames=("cfg", "pc"))
def _settle(w, x, ps, p, positions, cfg, pc):
    def step(carry, pos):
        w, x, ps = carry
        w = w._replace(pos=jnp.broadcast_to(pos, (cfg.n_hens, 2)),
                       heading=jnp.zeros((cfg.n_hens,)))
        obs = sensing.observe(w, cfg)
        x, motor, _d = brain.step(x, obs, p, cfg.dt)
        ps = plasticity.update_traces(ps, neurons.rate(x), motor,
                                      jnp.zeros((cfg.n_hens,)), cfg, pc)
        return (w, x, ps), None
    (w, x, ps), _ = jax.lax.scan(step, (w, x, ps), positions)
    return ps


def settle(cell):
    w = world.reset(jax.random.key(0), CFG)
    x = brain.initial_state(p0, 2)
    ps = plasticity.initial_state(p0, 2, PC)
    return _settle(w, x, ps, p0, _positions(cell), CFG, PC)


P, OTHERS = 2, [12, 22, 24]
print(f"tour {LAPS*len(TOUR)*DWELL*CFG.dt:.0f}s (baseline_tau_s={PC.baseline_tau_s:.0f}s), "
      f"then {HOLD*CFG.dt:.0f}s parked\n")

for centred in (False, True):
    ps_states = {c: settle(c) for c in [P] + OTHERS}
    psP = ps_states[P]
    a, b = float(jnp.linalg.norm(psP.z_lag)), float(jnp.linalg.norm(psP.z_lag_bar))
    src_P = (psP.z_lag - psP.z_lag_bar) if centred else psP.z_lag
    v = np.asarray(src_P[0]) * np.asarray(p0.pred_src)
    wp = np.zeros_like(np.asarray(p0.W_pred))
    wp[:, GAKEL_CH, :] = v / (float(v @ v) + 1e-12)
    p1 = p0._replace(W_pred=jnp.asarray(wp))

    def pred(c):
        ps = ps_states[c]
        s = (ps.z_lag - ps.z_lag_bar) if centred else ps.z_lag
        return float(jnp.einsum("hon,hn->ho", p1.W_pred,
                               s * p1.pred_src[None, :])[0, GAKEL_CH])

    at_P, others = pred(P), [pred(c) for c in OTHERS]
    print(f"pred_centred={str(centred):<5}  |z_lag|={a:.3f} |z_lag_bar|={b:.3f} "
          f"(ratio {b/max(a,1e-9):.3f})")
    print(f"{'':>21}at P {at_P:+.4f} | elsewhere "
          f"{', '.join(f'{o:+.4f}' for o in others)} | "
          f"mean elsewhere/P = {np.mean(others)/at_P:.3f}\n")
print("Selectivity = mean elsewhere/P well below 1.0. E070 measured 0.96 uncentred.")
