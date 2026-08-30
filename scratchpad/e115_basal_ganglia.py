"""E115: does a real subpallium let this model select actions?

Populations with Dale-correct signs, tonic pallidal inhibition, striatal collaterals for
competition, and selection by disinhibition -- not a multiplicative gate on an output.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import metrics, simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE = int(30 * 60 / CFG.dt), int(2 * 60 / CFG.dt)
SEED0, SEEDS = int(os.environ.get("E115_SEED0", "0")), 8
PLAIN = regions.DEFAULT_REGIONS
BG = regions.Regions(striatum=64, pallidum=32)
CHANNELS = int(os.environ.get("E115_CHANNELS", "0"))   # 0 = random loop, 8 = topographic
LEARN = plasticity.PlasticConfig(enabled=True)
FROZEN = plasticity.PlasticConfig(enabled=True, eta_out=0.0)


@partial(jax.jit, static_argnames=("cfg", "pc", "n", "reg"))
def probe(w, x, p, ps, key, cfg, pc, n, reg):
    n_motor = p.W_out.shape[-1]
    s_lo, s_hi = reg.bounds(regions.STRIATUM)
    p_lo, p_hi = reg.bounds(regions.PALLIDUM)

    def step(c, _):
        c, out = simulate._one_step(c, None, cfg, pc)
        r = neurons.rate(c[1])
        return c, (r[:, -n_motor:], out[0],
                   jnp.mean(r[:, s_lo:s_hi]) if s_hi > s_lo else jnp.zeros(()),
                   jnp.mean(r[:, p_lo:p_hi]) if p_hi > p_lo else jnp.zeros(()))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


def entropy(motor):
    """Normalised entropy of the motor output across channels. 1.0 = every action
    equally driven, which is the opposite of selection."""
    a = np.asarray(motor).reshape(-1, motor.shape[-1])
    q = a / (a.sum(-1, keepdims=True) + 1e-12)
    h = -(q * np.log(q + 1e-12)).sum(-1)
    return float(np.mean(h) / np.log(a.shape[-1]))


t0 = time.perf_counter()
print(f"E115 -- a real subpallium. seeds {SEED0}-{SEED0+SEEDS-1}, "
      f"subpallium_channels={CHANNELS}\n")
print(f"plain brain N={PLAIN.total}, with subpallium N={BG.total} "
      f"(striatum {BG.striatum}, pallidum {BG.pallidum})\n")

rows = {}
for reg_name, reg in (("plain", PLAIN), ("subpallium", BG)):
    for learn_name, pc in (("frozen", FROZEN), ("learning", LEARN)):
        hu, cd, mrate, stab, ent, stri, pall = [], [], [], [], [], [], []
        for s in range(SEED0, SEED0 + SEEDS):
            k = jax.random.key(s)
            p0 = connectome.build(jax.random.fold_in(k, 1), reg,
                                  n_hens=CFG.n_hens,
                                  subpallium_channels=CHANNELS)
            w2, x2, p2, ps2, _k = simulate.rollout_quiet(
                world.reset(k, CFG), brain.initial_state(p0, CFG.n_hens), p0,
                jax.random.fold_in(k, 2), CFG, REAR,
                plasticity.initial_state(p0, CFG.n_hens, pc), pc)
            hu.append(float(np.mean(np.asarray(w2.hunger))))
            cd.append(float(np.sum(np.asarray(w2.n_caught_any))
                            / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
            stub, motor, sr, pr = probe(
                world.reset(k, CFG), brain.initial_state(p2, CFG.n_hens), p2, ps2,
                jax.random.fold_in(k, 5), CFG,
                plasticity.PlasticConfig(enabled=False), PROBE, reg)
            mrate.append(float(np.mean(np.asarray(stub))))
            stab.append(metrics.direction_stability(np.asarray(stub)))
            ent.append(entropy(motor))
            stri.append(float(np.mean(np.asarray(sr))))
            pall.append(float(np.mean(np.asarray(pr))))
        rows[(reg_name, learn_name)] = dict(
            hu=np.array(hu), cd=np.array(cd), mrate=np.mean(mrate),
            stab=np.mean(stab), ent=np.mean(ent),
            stri=np.mean(stri), pall=np.mean(pall))
        r = rows[(reg_name, learn_name)]
        print(f"{reg_name:>11} {learn_name:>9}  hunger {np.mean(hu):.4f}  "
              f"caught/dive {np.mean(cd):.4f}  motor rate {r['mrate']:.4f}  "
              f"stub stab {r['stab']:.4f}  entropy {r['ent']:.4f}  "
              f"striatum {r['stri']:.4f}  pallidum {r['pall']:.4f}")


def paired(a, b, name):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"    {name:<48}{d.mean():+.4f} +/- {se:.4f}  t={d.mean()/(se+1e-12):+.2f}")


print(f"\n  paired, df={SEEDS-1}, crit 2.365 (lower hunger is better):")
for m in ("hu", "cd"):
    lab = "hunger" if m == "hu" else "caught/dive"
    paired(rows[("subpallium", "frozen")][m], rows[("plain", "frozen")][m],
           f"subpallium vs plain, frozen ({lab})")
    paired(rows[("subpallium", "learning")][m], rows[("subpallium", "frozen")][m],
           f"learning on the subpallium ({lab})")
    paired(rows[("plain", "learning")][m], rows[("plain", "frozen")][m],
           f"learning on the plain brain ({lab})")

print("\n--- pre-registered falsifiers (E115 section 4) ---")
bg = rows[("subpallium", "frozen")]
pl = rows[("plain", "frozen")]
print(f"manipulation  pallidum {bg['pall']:.4f} (must be >0.7), "
      f"striatum {bg['stri']:.4f} (must be <0.15) -> "
      f"{'a basal ganglia' if bg['pall'] > 0.7 and bg['stri'] < 0.15 else 'NOT ONE, void'}")
print(f"degeneracy    motor rate {bg['mrate']:.4f} vs plain {pl['mrate']:.4f} "
      f"(fires if <0.10 or <half of plain)")
print(f"selection     stub direction stability {bg['stab']:.4f} vs plain {pl['stab']:.4f} "
      f"(E107 measured 0.9998); entropy {bg['ent']:.4f} vs {pl['ent']:.4f}")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
