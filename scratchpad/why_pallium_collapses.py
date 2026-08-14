"""Is the 17x collapse at the pallium recurrent mixing, or dilution by silent units?

Three variants of the same measurement:
  intact         as built
  no recurrence  pallium receives its afferents but does not talk to itself
  targeted       audio afferents given their own non-overlapping slice of pallium
                 (a crude stand-in for Field L being anatomically separate)
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, neurons, regions

reg = regions.DEFAULT_REGIONS
DT, HOLD = 0.01, 200
p_lo, p_hi = reg.bounds(regions.PALLIUM)

o_hawk = np.zeros(spec.OBS_DIM, np.float32); o_hawk[spec.IDX_AERIAL] = 1.0
o_call = np.zeros(spec.OBS_DIM, np.float32); o_call[spec.AUDIO_LO + 2] = 1.0
o_rest = np.zeros(spec.OBS_DIM, np.float32)


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, _, _ = brain.step(x, o, p, DT)
    return np.asarray(neurons.rate(x))[0]


def pallial_sep(p):
    h, c, z = (settle(p, o)[p_lo:p_hi] for o in (o_hawk, o_call, o_rest))
    return float(np.sqrt(np.mean((h - c) ** 2)) / (np.mean(np.abs(z)) + 1e-9))


def kill_pallial_recurrence(p):
    W = np.asarray(p.W).copy()
    W[:, p_lo:p_hi, p_lo:p_hi] = 0.0
    return p._replace(W=jnp.asarray(W))


def segregate_audio(p):
    """Route auditory afferents to their own sixth of the pallium, vision to the rest.

    Not anatomy -- a probe. If separability jumps, the collapse is modality mixing and
    structured wiring is the fix. If it does not, the problem is elsewhere.
    """
    W_in = np.asarray(p.W_in).copy()
    W = np.asarray(p.W).copy()
    s_lo, s_hi = reg.bounds(regions.SENSORY)
    n_stub = s_hi - s_lo
    split = s_lo + n_stub // 6                    # first sixth becomes "Field L"
    aud = slice(spec.AUDIO_LO, spec.AUDIO_HI)
    W_in[s_lo:split, :] = 0.0;  W_in[s_lo:split, aud] = np.asarray(p.W_in)[s_lo:split, aud]
    W_in[split:s_hi, aud] = 0.0
    # and give the two stub partitions disjoint pallial targets
    mid = p_lo + (p_hi - p_lo) // 6
    W[:, mid:p_hi, s_lo:split] = 0.0              # Field L -> only its own patch
    W[:, p_lo:mid, split:s_hi] = 0.0              # visual stub -> only the rest
    return p._replace(W_in=jnp.asarray(W_in), W=jnp.asarray(W))


res = {k: [] for k in ("intact", "no recurrence", "targeted (Field L)")}
for s in range(6):
    p = connectome.build(jax.random.key(s), reg, n_hens=1)
    res["intact"].append(pallial_sep(p))
    res["no recurrence"].append(pallial_sep(kill_pallial_recurrence(p)))
    res["targeted (Field L)"].append(pallial_sep(segregate_audio(p)))

print("pallial separability of 'saw hawk' vs 'heard alarm' (x mean rate)")
base = np.mean(res["intact"])
for k, v in res.items():
    print(f"  {k:<22} {np.mean(v):.4f} +- {np.std(v):.4f}   ({np.mean(v)/base:.2f}x intact)")
