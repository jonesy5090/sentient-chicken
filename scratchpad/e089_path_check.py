"""Does the planted prediction actually reach M_PECK? CLAUDE.md check 4, done directly.

E089 measured peck@target falling only 2.9% between pred_gain 0 and 2, while the
arithmetic says a saturated gakel percept applies -1.5 to M_PECK and should move it far
more. One of those is wrong, and which one decides whether E089 is a stop condition or a
seventh instrument failure.
"""
import sys; sys.path.insert(0, 'scratchpad')
from functools import partial
import jax, jax.numpy as jnp, numpy as np
import e083_leaving_anchor as E
import e089_whole_chain_control as C
from coop import sensing, spec, world
from hen import brain, neurons, plasticity, regions

k = jax.random.key(3)
pc_fit = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0, pred_bar_freeze_s=C.FREEZE)
p, tgt, ctl, food, acc, ratio, dec = C.fit_and_gate(C.build(k), pc_fit, k)
print(f"seed 3: target cell {tgt}, gate acc {acc:.1%}, ratio {ratio:.2f}\n")

cfg = C.CFG
w = world.reset(k, cfg)._replace(
    pos=jnp.broadcast_to(jnp.asarray(C.CEN[tgt], dtype=jnp.float32), (C.HENS, 2)),
    heading=jnp.zeros((C.HENS,)),
    food_pos=jnp.asarray(np.stack([C.CEN[tgt], C.CEN[ctl]]), dtype=jnp.float32))
obs = sensing.observe(w, cfg)
print(f"gakel audio channel at rest : {float(obs[0, E.GAKEL_CH]):.4f}")
print(f"food in view (max over bins): "
      f"{max(float(obs[0, spec.vis_index(b, spec.CLS_FOOD)]) for b in range(spec.N_BINS)):.4f}\n")


@partial(jax.jit, static_argnames=("cfg", "pc", "n", "gain"))
def settle(x, ps, obs, cfg, pc, n, gain):
    def step(c, _):
        x, ps = c
        x, motor, d = brain.step(x, obs, p, cfg.dt, pred_gain=gain,
                                 pred_from=(ps.z_lag - ps.z_lag_bar))
        ps = plasticity.update_traces(ps, neurons.rate(x), motor,
                                      jnp.zeros((cfg.n_hens,)), cfg, pc)
        return (x, ps), (motor, d.predicted)
    (x, ps), (motor, pred) = jax.lax.scan(step, (x, ps), None, length=n)
    return motor[-1], pred[-1]


for gain in (0.0, 2.0):
    pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=gain, pred_bar_freeze_s=C.FREEZE)
    x = brain.initial_state(p, C.HENS)
    ps = plasticity.initial_state(p, C.HENS, pc)
    motor, pred = settle(x, ps, obs, cfg, pc, 9000, gain)     # 90 s, past the 60 s freeze
    g = np.asarray(pred)[:, E.GAKEL_CH]
    rin = np.clip(np.asarray(obs)[:, E.GAKEL_CH] + gain * np.maximum(g, 0), 0, 1)
    print(f"gain {gain}: predicted@gakel {g.mean():+.4f}   reflex_in[gakel] {rin.mean():.4f}"
          f"   M_PECK {float(jnp.mean(motor[:, spec.M_PECK])):.4f}"
          f"   M_FORWARD {float(jnp.mean(motor[:, spec.M_FORWARD])):.4f}")
