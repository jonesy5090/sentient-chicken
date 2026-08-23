"""E100: is the cortical drive a scaled copy of the reflex drive?

If W_out is trained with a postsynaptic factor that traces the final motor output -- which
at cortical/reflex 0.03-0.10 IS the reflex arc -- then the readout should converge on
reconstructing the arc. Measured as cosine similarity between the two drive vectors that
`brain.step` already returns separately.

The shuffled control is the load-bearing one: both vectors could score high simply because
both are dominated by the same always-on motor channels.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR = int(30 * 60 / CFG.dt)
PROBE = int(2 * 60 / CFG.dt)
SEEDS = 4
ARMS = {
    "fixed":        plasticity.PlasticConfig(enabled=False),
    "instrumental": plasticity.PlasticConfig(enabled=True),
    "hebbian":      plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                                             readout_scaling_strength=0.3),
}


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, ps, key, cfg, n):
    """Free-running, plasticity off. Returns per-step cortical and reflex drive."""
    def step(carry, _):
        w, x, p, ps, key = carry
        key, k = jax.random.split(key)
        obs = __import__("coop.sensing", fromlist=["observe"]).observe(w, cfg)
        x, motor, d = brain.step(x, obs, p, cfg.dt)
        w = world.step(w, motor, k, cfg)
        return (w, x, p, ps, key), (d.cortical, d.reflex)
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


def _cos(a, b):
    return ((a * b).sum(1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).mean()


def cosines(cort, refl, rng):
    """Cosine against two nulls.

    CHANNEL null (E100 as pre-registered): permute the reflex vector's channels. This
    turned out to be the wrong control -- it destroys the magnitude correspondence that
    makes ANY two motor vectors look alike, and it scored 0.53 against a real 0.59, so
    the pre-registered triviality falsifier fired and the measurement was uninformative.

    TIME null (E100b): pair cortical drive at step t with reflex drive at a different
    step. This keeps each vector's channel structure exactly as it is and destroys only
    the moment-to-moment correspondence -- which is the thing "the readout reconstructs
    the arc" actually asserts. Excess over this null is the quantity of interest.
    """
    c = np.asarray(cort).reshape(-1, cort.shape[-1])
    r = np.asarray(refl).reshape(-1, refl.shape[-1])
    keep = (np.linalg.norm(c, axis=1) > 1e-8) & (np.linalg.norm(r, axis=1) > 1e-8)
    c, r = c[keep], r[keep]
    cos = _cos(c, r)
    cos_chan = _cos(c, r[:, rng.permutation(r.shape[1])])
    cos_time = _cos(c, r[rng.permutation(r.shape[0])])
    ratio = np.linalg.norm(c, axis=1).mean() / max(np.linalg.norm(r, axis=1).mean(), 1e-9)
    return float(cos), float(cos_chan), float(cos_time), float(ratio)


print(f"E100 -- cosine(cortical drive, reflex drive). {SEEDS} seeds, 30 min rearing, "
      f"2 min probe\n")
t0 = time.perf_counter()
print(f"{'arm':>14}{'at hatch':>11}{'reared':>10}{'chan null':>11}{'TIME null':>11}"
      f"{'excess':>11}{'cort/refl':>12}")
rows = {}
for name, pc in ARMS.items():
    hatch, reared, shuf, tnull, ratios = [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        rng = np.random.default_rng(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=16)
        w = world.reset(k, CFG)
        x = brain.initial_state(p0, 16)
        ps = plasticity.initial_state(p0, 16, pc)
        # at hatch
        c, r = probe(w, x, p0, ps, jax.random.fold_in(k, 5), CFG, PROBE)
        h, _s, _t, _q = cosines(c, r, rng)
        hatch.append(h)
        # rear, then probe the same way
        w2, x2, p2, ps2, _k, _tr = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), CFG, REAR, pc=pc, ps=ps)
        w3 = world.reset(k, CFG)
        x3 = brain.initial_state(p2, 16)
        c, r = probe(w3, x3, p2, ps2, jax.random.fold_in(k, 5), CFG, PROBE)
        a, b, tn, q = cosines(c, r, rng)
        reared.append(a); shuf.append(b); tnull.append(tn); ratios.append(q)
    rows[name] = (np.mean(hatch), np.mean(reared), np.mean(shuf), np.mean(tnull),
                  np.mean(ratios))
    print(f"{name:>14}{np.mean(hatch):>11.4f}{np.mean(reared):>10.4f}"
          f"{np.mean(shuf):>11.4f}{np.mean(tnull):>11.4f}"
          f"{np.mean(reared)-np.mean(tnull):>+11.4f}{np.mean(ratios):>12.4f}")

heb = rows["hebbian"]; ins = rows["instrumental"]
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifiers (E100 section 4) ---")
print(f"primary     hebbian reared cosine {heb[1]:.4f} (fires if <0.40, predicted >=0.70) "
      f"-> {'FIRES -- claim rejected on measurement' if heb[1] < 0.40 else 'clear'}")
print(f"triviality  hebbian channel-null {heb[2]:.4f} (fires if >0.40) -> "
      f"{'FIRES -- channel null is uninformative, see TIME null' if heb[2] > 0.40 else 'clear'}")
print(f"E100b       excess over TIME null: hebbian {heb[1]-heb[3]:+.4f}, "
      f"instrumental {ins[1]-ins[3]:+.4f}, fixed {rows['fixed'][1]-rows['fixed'][3]:+.4f}")
print("            (this is the quantity 'the readout reconstructs the arc' asserts)")
print(f"direction   hatch {heb[0]:.4f} -> reared {heb[1]:.4f} "
      f"(fires if no increase) -> {'FIRES' if heb[1] <= heb[0] else 'clear'}")
print(f"prediction2 hebbian {heb[1]:.4f} vs instrumental {ins[1]:.4f} "
      f"-> {'held' if heb[1] > ins[1] else 'NOT held'}")
