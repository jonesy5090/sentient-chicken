"""E035: does connectivity-prior-driven modality segregation reproduce E017/E034's
ad hoc probe?

It does not (discovered during this run -- see the file's own docstring history in
docs/experiments/E035). The probe zeroed connections on an *already fan-in-normalised*
connectome without re-normalising afterward, leaving the segregated slice under-driven
relative to its true (reduced) fan-in. That is a confound, not a clean measurement of
anatomical segregation. This script measures three conditions to isolate it:

  intact          no segregation
  structural      connectome.build(modality_segregated=True) -- fan-in computed AFTER
                  the cross-modal cut, so surviving synapses are correctly re-normalised
                  (the codebase's standard convention, applied consistently)
  posthoc         E017/E034's method exactly: build unrestricted (fan-in normalised for
                  full connectivity), THEN zero the cross-modal entries with no
                  re-normalisation -- reproduced here for direct, same-key comparison
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, neurons, regions
from hen.connectome import _modality_bounds

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


def pallial_mean_rate(p, obs):
    return float(np.mean(np.abs(settle(p, obs)[p_lo:p_hi])))


def build_posthoc(seed, frac=1 / 6):
    """E017/E034's exact method: build unrestricted, zero afterward, no re-normalising."""
    p = connectome.build(jax.random.key(seed), reg, n_hens=1)
    s_lo, s_split, s_hi, p_lo2, p_split, p_hi2 = _modality_bounds(reg, frac)
    W = np.asarray(p.W).copy()
    W_in = np.asarray(p.W_in).copy()
    W[:, p_split:p_hi2, s_lo:s_split] = 0.0
    W[:, p_lo2:p_split, s_split:s_hi] = 0.0
    audio = slice(spec.AUDIO_LO, spec.AUDIO_HI)
    W_in[s_lo:s_split, :] = 0.0
    W_in[s_lo:s_split, audio] = np.asarray(p.W_in)[s_lo:s_split, audio]
    W_in[s_split:s_hi, audio] = 0.0
    return p._replace(W=jnp.asarray(W), W_in=jnp.asarray(W_in))


N = 12
print(f"Part A -- {N} genomes, same key across conditions (paired)")
print(f"{'seed':<6}{'intact':>10}{'structural':>12}{'posthoc':>10}")
rows = {"intact": [], "structural": [], "posthoc": []}
for s in range(N):
    p_intact = connectome.build(jax.random.key(s), reg, n_hens=1)
    p_struct = connectome.build(jax.random.key(s), reg, n_hens=1, modality_segregated=True)
    p_post = build_posthoc(s)
    a, b, c = pallial_sep(p_intact), pallial_sep(p_struct), pallial_sep(p_post)
    rows["intact"].append(a); rows["structural"].append(b); rows["posthoc"].append(c)
    print(f"{s:<6}{a:>10.4f}{b:>12.4f}{c:>10.4f}")

print()
base = np.mean(rows["intact"])
for k in ("intact", "structural", "posthoc"):
    v = np.array(rows[k])
    print(f"{k:<12} {v.mean():.4f} +- {v.std():.4f}   ({v.mean()/base:.2f}x intact)")

d = np.array(rows["structural"]) - np.array(rows["posthoc"])
print(f"\nstructural - posthoc, paired: {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(N):.4f}"
      f"  (mean rate check below)")

print("\nField-L slice mean rate under a call, intact vs structural (fan-in re-normalised):")
p_lo_f, p_split_f, p_hi_f = _modality_bounds(reg, 1 / 6)[3:6]
for s in range(4):
    p_intact = connectome.build(jax.random.key(s), reg, n_hens=1)
    p_struct = connectome.build(jax.random.key(s), reg, n_hens=1, modality_segregated=True)
    r_i = np.mean(np.abs(settle(p_intact, o_call)[p_lo_f:p_split_f]))
    r_s = np.mean(np.abs(settle(p_struct, o_call)[p_lo_f:p_split_f]))
    print(f"  seed {s}: intact {r_i:.4f}  structural {r_s:.4f}")

print("\nPart B -- fraction sweep, structural (properly normalised) only, 6 genomes")
print(f"{'fraction':<12}{'separability':>14}")
for frac, label in [(1/6, "1/6"), (1/3, "1/3"), (1/2, "1/2")]:
    vals = [pallial_sep(connectome.build(jax.random.key(s), reg, n_hens=1,
                                         modality_segregated=True, aud_fraction=frac))
            for s in range(6)]
    v = np.array(vals)
    print(f"{label:<12}{v.mean():>8.4f} +- {v.std():.4f}")
