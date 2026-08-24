"""E109: is the only direction the readout can be pushed the reflex arc's own?

Delta_cortical = dw_out @ stub = m * (dz_slow . stub) * dz_motor, so dz_motor is the
whole of the update's direction in motor space. This measures what it points at.

Traces are rebuilt offline from per-step drives, so the counterfactuals can be traced
identically. The reconstruction is checked against the live `ps.z_motor` first -- if it
does not match, nothing built on it means anything.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16)
PC = plasticity.PlasticConfig(enabled=True)
INTERVAL = PC.interval
WINDOWS, SEEDS = int(300 / BASE.dt) // INTERVAL, 4
# E033's lesson: a hardcoded `range(SEEDS)` silently re-runs block one when someone
# means to replicate. The offset is an argument.
OFFSET = int(os.environ.get("E109_SEED_OFFSET", "0"))
REAR = int(30 * 60 / BASE.dt)
reg = regions.DEFAULT_REGIONS
A_M = BASE.dt / PC.tau_motor
A_B = BASE.dt / PC.baseline_tau_s


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def trace_and_centre(x, z0=None, bar0=None):
    """Replicate `update_traces` for one signal: an EMA at tau_motor, and a slow mean
    of that EMA at baseline_tau_s. `x` is (T, H, D). Returns the centred trace.

    Two details that the first version of this script got wrong, and the reconstruction
    falsifier caught before any cosine was read.

    **Initial conditions.** A reared arm's live traces have been running for 30 minutes,
    so starting the offline reconstruction at zero compares two different signals. `z0`
    and `bar0` come from the live `PlasticState` at the moment the probe starts.

    **The slow mean lags by one step.** `update_traces` builds its whole return value
    from the OLD state, so `z_motor_bar` advances toward the *previous* `z_motor`, not
    the one computed in the same call. Reproduced exactly here rather than approximated,
    because the point of this script is to measure the rule that runs.
    """
    z = np.zeros_like(x[0]) if z0 is None else np.array(z0, dtype=x.dtype)
    bar = np.zeros_like(x[0]) if bar0 is None else np.array(bar0, dtype=x.dtype)
    out = np.empty_like(x)
    for t in range(len(x)):
        z_old = z
        z = z + A_M * (x[t] - z)
        bar = bar + A_B * (z_old - bar)
        out[t] = z - bar
    return out


def cosine(a, b):
    """Mean cosine between matched rows of two (N, D) arrays."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    keep = (na > 1e-9) & (nb > 1e-9)
    if keep.sum() == 0:
        return float("nan")
    return float((np.sum(a[keep] * b[keep], axis=-1)
                  / (na[keep] * nb[keep])).mean())


@partial(jax.jit, static_argnames=("cfg", "pc", "n_windows"))
def windows(carry, cfg, pc, n_windows):
    def inner(c, _):
        w0 = c[0]
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, None

    def emit(c, _):
        """One window: every step's drives, then the live trace at the boundary."""
        def step(cc, _):
            w = cc[0]
            fed0 = w.n_fed
            cc, _o = simulate._one_step(cc, None, cfg, pc)
            # Recompute the drives at the state the step just produced, noise-free, so
            # the reconstruction is deterministic and the noise share is the residual.
            _x, motor, d = brain.step(cc[1], _o[1], cc[2], cfg.dt)
            return cc, (_o[0], d.reflex, d.cortical, cc[2].b_motor,
                        cc[0].n_fed - fed0)
        c, out = jax.lax.scan(step, c, None, length=INTERVAL)
        w, x, p, ps, key = c
        return c, out + (ps.z_motor - ps.z_motor_bar, ps.z_motor, ps.z_motor_bar)
    return jax.lax.scan(emit, carry, None, length=n_windows)[1]


def run(seed, arm):
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    w = world.reset(k, BASE)
    x = brain.initial_state(p, 16)
    pc = {"untrained": PC, "instrumental": PC,
          "hebbian": PC._replace(hebbian_readout=True,
                                 readout_scaling_strength=0.3)}[arm]
    ps = plasticity.initial_state(p, 16, pc)
    if arm != "untrained":
        w, x, p, ps, _k = simulate.rollout_quiet(
            w, x, p, jax.random.fold_in(k, 2), BASE, REAR, ps, pc)
    carry = (w, x, p, ps, jax.random.fold_in(k, 5))
    return ([np.asarray(a) for a in windows(carry, BASE, pc, WINDOWS)], pc,
            np.asarray(ps.z_motor), np.asarray(ps.z_motor_bar))


t0 = time.perf_counter()
print(f"E109 -- what the rule writes. seeds {OFFSET}-{OFFSET+SEEDS-1}, {WINDOWS} windows\n")
for arm in ("untrained", "instrumental", "hebbian"):
    recon_err, cos_r, cos_c, sh_r, sh_c, sh_n, mags, cos_r_fed = ([] for _ in range(8))
    for s in range(OFFSET, OFFSET + SEEDS):
        ((motor, refl, cort, bm, dfed, dz_live, z_live, bar_live),
         pc, z0, bar0) = run(s, arm)
        W, T, H, D = motor.shape
        f = lambda a: a.reshape(W * T, H, D)
        refl_f, cort_f, motor_f = f(refl), f(cort), f(motor)
        # `b_motor` is (MOTOR_DIM,) -- shared across hens, not per hen -- so it is
        # emitted as (W, T, D) and broadcasts rather than reshaping like the rest.
        bm_f = bm.reshape(W * T, 1, D)

        det = sigmoid(refl_f + cort_f + bm_f)
        # Counterfactuals start from the same live state as the real trace, so the only
        # thing that differs between them is which drive term is present.
        dz_full = trace_and_centre(det, z0, bar0)
        dz_refl = trace_and_centre(sigmoid(refl_f + bm_f), z0, bar0)
        dz_cort = trace_and_centre(sigmoid(cort_f + bm_f), z0, bar0)
        # Sample at the boundaries, where consolidate reads.
        idx = np.arange(T - 1, W * T, T)
        # Reconstruction check: my offline trace of the ACTUAL motor output vs the live
        # one. Uses `motor`, noise included, which is what `update_traces` traces.
        dz_actual = trace_and_centre(motor_f, z0, bar0)
        recon_err.append(float(np.max(np.abs(dz_actual[idx] - dz_live))))

        a = dz_live.reshape(-1, D)
        cos_r.append(cosine(a, dz_refl[idx].reshape(-1, D)))
        cos_c.append(cosine(a, dz_cort[idx].reshape(-1, D)))
        # Variance decomposition of the DRIVE at the boundaries.
        r, c = refl_f[idx].reshape(-1, D), cort_f[idx].reshape(-1, D)
        vr, vc = r.var(0).sum(), c.var(0).sum()
        vt = (r + c).var(0).sum()
        sh_r.append(float(vr / vt))
        sh_c.append(float(vc / vt))
        # Noise share: the emitted motor minus the deterministic reconstruction.
        sh_n.append(float(np.var(motor_f - det) / max(np.var(motor_f), 1e-12)))
        mags.append((float(np.mean(np.abs(r))), float(np.mean(np.abs(c)))))
        fed = (dfed.sum(1) > 0).reshape(-1)
        if fed.sum() > 20:
            cos_r_fed.append(cosine(a[fed], dz_refl[idx].reshape(-1, D)[fed]))

    mr = np.mean([m[0] for m in mags]); mc = np.mean([m[1] for m in mags])
    print(f"[{arm}]")
    print(f"  reconstruction check   max |offline - live| = {np.max(recon_err):.2e} "
          f"(falsifier fires above 1e-4)")
    print(f"  |reflex| {mr:.4f}   |cortical| {mc:.4f}   ratio {mc/max(mr,1e-9):.4f} "
          f"(instrument void below 0.01)")
    print(f"  cosine(dz_motor, reflex-only deviation)    {np.nanmean(cos_r):>7.4f}")
    print(f"  cosine(dz_motor, cortical-only deviation)  {np.nanmean(cos_c):>7.4f}")
    if cos_r_fed:
        print(f"  same, restricted to feeding windows       {np.nanmean(cos_r_fed):>7.4f}")
    print(f"  drive variance: reflex {100*np.mean(sh_r):.1f}%  "
          f"cortical {100*np.mean(sh_c):.1f}%   noise share of motor "
          f"{100*np.mean(sh_n):.1f}%")
    print()

print("--- pre-registered falsifiers (E109 section 4) ---")
print("primary      cosine < 0.5 under the instrumental rule -> this explanation joins")
print("             the other three and H2's null has no remaining candidate in the rule")
print("triviality   noise share > 90% -> the alignment measures noise")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
