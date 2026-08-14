"""Where along sensory -> pallium -> motor does 'hawk' stop differing from 'alarm call'?

Presents each percept alone, holds it for 2 s of chicken time, and reads separability
at each stage as a fraction of the mean activity at that stage -- so the numbers are
comparable across regions of different size and different operating rate.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, regions

reg = regions.DEFAULT_REGIONS
DT, HOLD = 0.01, 200

o_hawk = np.zeros(spec.OBS_DIM, np.float32); o_hawk[spec.IDX_AERIAL] = 1.0
o_call = np.zeros(spec.OBS_DIM, np.float32); o_call[spec.AUDIO_LO + 2] = 1.0
o_rest = np.zeros(spec.OBS_DIM, np.float32)


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, motor, _ = brain.step(x, o, p, DT)
    return np.asarray(brain_rate(x))[0], np.asarray(motor)[0]


def brain_rate(x):
    from hen import neurons
    return neurons.rate(x)


def sep(a, b, ref):
    """RMS difference between two state vectors, as a fraction of mean activity."""
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.mean(np.abs(ref)) + 1e-9))


rows = {k: [] for k in ("sensory", "pallium", "arcopallium", "motor_stub", "motor_out")}
for s in range(6):
    p = connectome.build(jax.random.key(s), reg, n_hens=1)
    r_h, m_h = settle(p, o_hawk)
    r_c, m_c = settle(p, o_call)
    r_0, _ = settle(p, o_rest)
    for name, rid in (("sensory", regions.SENSORY), ("pallium", regions.PALLIUM),
                      ("arcopallium", regions.ARCOPALLIUM), ("motor_stub", regions.MOTOR)):
        lo, hi = reg.bounds(rid)
        rows[name].append(sep(r_h[lo:hi], r_c[lo:hi], r_0[lo:hi]))
    rows["motor_out"].append(sep(m_h, m_c, m_h))

print(f"{'stage':<13} separability (RMS diff / mean activity)")
for k, v in rows.items():
    print(f"{k:<13} {np.mean(v):.4f} +- {np.std(v):.4f}")

# What actually reaches the muscles: the reflex arc's own response to each percept.
p = connectome.build(jax.random.key(0), reg, n_hens=1)
R = np.asarray(p.reflex)
print("\nreflex arc drive to each motor channel:")
print(f"{'channel':<14}{'hawk seen':>11}{'alarm heard':>13}")
for i, nm in enumerate(("forward", "turn_L", "turn_R", "peck", "scratch", "crouch",
                        "flee", "call_contact", "call_food", "call_aerial", "call_ground")):
    print(f"{nm:<14}{R[i] @ o_hawk:>11.2f}{R[i] @ o_call:>13.2f}")
