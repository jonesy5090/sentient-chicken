"""E110: does an update with its own direction produce learning that changes behaviour?

The frozen arm (eta_out=0) is what this turns on. It is what "learning changed nothing"
actually looks like, and no previous experiment in this arc has had one.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, SEEDS = int(30 * 60 / BASE.dt), 8
OFFSET = int(os.environ.get("E110_SEED_OFFSET", "0"))
PC = plasticity.PlasticConfig(enabled=True)
INTERVAL = PC.interval
reg = regions.DEFAULT_REGIONS
A_M = BASE.dt / PC.tau_motor
A_B = BASE.dt / PC.baseline_tau_s

ARMS = (("baseline (motor)", PC),
        ("noise", PC._replace(postsynaptic_factor="noise")),
        ("cortical", PC._replace(postsynaptic_factor="cortical")),
        ("FROZEN (eta_out=0)", PC._replace(eta_out=0.0)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def trace_and_centre(x, z0, bar0):
    """E109's offline reconstruction, verified there against the live trace to 1.8e-07."""
    z, bar = np.array(z0, dtype=x.dtype), np.array(bar0, dtype=x.dtype)
    out = np.empty_like(x)
    for t in range(len(x)):
        z_old = z
        z = z + A_M * (x[t] - z)
        bar = bar + A_B * (z_old - bar)
        out[t] = z - bar
    return out


def cosine(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    keep = (na > 1e-9) & (nb > 1e-9)
    if keep.sum() == 0:
        return float("nan")
    return float((np.sum(a[keep] * b[keep], -1) / (na[keep] * nb[keep])).mean())


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def probe(carry, cfg, pc, n):
    """Short instrumented run: the arm's postsynaptic factor against the arc's."""
    def step(c, _):
        w = c[0]
        c, out = simulate._one_step(c, None, cfg, pc)
        _x, _m, d = brain.step(c[1], out[1], c[2], cfg.dt)
        post = plasticity.postsynaptic_signal(out[0], d, c[2], pc)
        return c, (out[0], d.reflex, d.cortical, c[2].b_motor,
                   out[0] if post is None else post, d.cortical)
    return jax.lax.scan(step, carry, None, length=n)[1]


def rear(seed, pc):
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    w = world.reset(k, BASE)
    x = brain.initial_state(p, 16)
    ps = plasticity.initial_state(p, 16, pc)
    w2, x2, p2, ps2, _k = simulate.rollout_quiet(
        w, x, p, jax.random.fold_in(k, 2), BASE, REAR, ps, pc)
    dw = float(jnp.mean(jnp.abs(p2.W_out - p.W_out)))
    return k, w2, x2, p2, ps2, dw


t0 = time.perf_counter()
print(f"E110 -- a postsynaptic factor with its own direction. "
      f"seeds {OFFSET}-{OFFSET+SEEDS-1}\n")

hung, caught, res = {}, {}, {}
for label, pc in ARMS:
    h, c, cos_r, dws, mags = [], [], [], [], []
    for s in range(OFFSET, OFFSET + SEEDS):
        k, w2, x2, p2, ps2, dw = rear(s, pc)
        h.append(float(np.mean(np.asarray(w2.hunger))))
        c.append(float(np.sum(np.asarray(w2.n_caught_any))
                       / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
        dws.append(dw)
        # Instrument: is this arm's postsynaptic factor actually pointing somewhere
        # other than the reflex arc? Measured by E109's method on a short replay.
        carry = (world.reset(k, BASE), brain.initial_state(p2, 16), p2, ps2,
                 jax.random.fold_in(k, 7))
        motor, refl, cort, bm, post, cortd = [np.asarray(a) for a in
                                              probe(carry, BASE, pc, 2000)]
        T, H, D = motor.shape
        bm_f = bm.reshape(T, 1, D)
        z0 = np.asarray(ps2.z_post if pc.postsynaptic_factor != "motor"
                        else ps2.z_motor)
        b0 = np.asarray(ps2.z_post_bar if pc.postsynaptic_factor != "motor"
                        else ps2.z_motor_bar)
        dz_post = trace_and_centre(post, z0, b0)
        dz_refl = trace_and_centre(sigmoid(refl + bm_f),
                                   np.asarray(ps2.z_motor),
                                   np.asarray(ps2.z_motor_bar))
        idx = np.arange(INTERVAL - 1, T, INTERVAL)
        cos_r.append(cosine(dz_post[idx].reshape(-1, D),
                            dz_refl[idx].reshape(-1, D)))
        mags.append(float(np.mean(np.abs(cortd))))
    hung[label], caught[label] = np.array(h), np.array(c)
    res[label] = (np.mean(cos_r), np.mean(dws), np.mean(mags))
    print(f"{label:>20}  hunger {np.mean(h):.4f}   caught/dive {np.mean(c):.4f}   "
          f"cos(post, arc) {np.mean(cos_r):>7.4f}   |dW_out| {np.mean(dws):.2e}   "
          f"|cort| {np.mean(mags):.4f}")


def paired(a, b, name):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / (se + 1e-12)
    sig = "  SIGNIFICANT" if abs(t) > 2.365 else ""
    print(f"    {name:<46}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}{sig}")


print(f"\n  paired against the FROZEN readout, df={SEEDS-1}, crit 2.365 "
      f"(lower hunger is better):")
for label, _ in ARMS[:3]:
    paired(hung[label], hung["FROZEN (eta_out=0)"], f"{label} vs frozen (hunger)")
for label, _ in ARMS[:3]:
    paired(caught[label], caught["FROZEN (eta_out=0)"],
           f"{label} vs frozen (caught/dive)")

print("\n--- pre-registered falsifiers (E110 section 4) ---")
base_cos, base_dw = res["baseline (motor)"][0], res["baseline (motor)"][1]
print(f"instrument   cos(post, arc) must FALL from the baseline's {base_cos:.4f}: "
      f"noise {res['noise'][0]:.4f} (bar 0.30), cortical {res['cortical'][0]:.4f} "
      f"(bar 0.60)")
for label in ("noise", "cortical"):
    r = res[label][1] / max(base_dw, 1e-30)
    print(f"magnitude    |dW_out| {label}/baseline = {r:.2f}x "
      f"(confound falsifier fires outside 0.5-2.0)")
print("primary      if the cosine falls AND no arm beats frozen on hunger, removing")
print("             E109's obstacle does not produce learning and the line closes")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
