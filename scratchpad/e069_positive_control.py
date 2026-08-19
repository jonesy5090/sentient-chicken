"""E069: the positive control T2 has needed since E065.

A -- can the metric see it?  Minimum detectable effect, derived from the real
     run-to-run variance in E065/E066/E068's own caches. No new compute.
B -- can the rule learn it?  sickness_penalty sweep. Measures whether the flock
     gets sick less, whether the weights move at all (|W|, NOT |W_out|: under
     hebbian_readout the readout is not reward-gated and structurally cannot
     respond -- the diagnostic error E068 found running unnoticed through three
     experiments), and whether the connectome survives a large discrete penalty
     (E014's failure mode).
"""
import json, os, time
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.experiment import _t_critical

# ---------------------------------------------------------------------------
# A -- minimum detectable effect, from real variance
# ---------------------------------------------------------------------------
print("=" * 72)
print("A -- minimum detectable effect on sickness-per-rotation (n=8 seeds)")
print("=" * 72)

CRIT = _t_critical(7)
print(f"threshold t({7}) = {CRIT:.3f}\n")
print(f"{'experiment':<12}{'baseline/rot':>14}{'SE(contrast)':>14}"
      f"{'MDE (count)':>14}{'MDE (% of base)':>18}")

for name, path in (("E065", "scratchpad/e065_cache.json"),
                   ("E066", "scratchpad/e066_cache.json"),
                   ("E068", "scratchpad/e068_cache.json")):
    if not os.path.exists(path):
        print(f"{name:<12}{'(cache missing)':>60}")
        continue
    cache = json.load(open(path))
    seeds = sorted({k.split("|")[1] for k in cache})
    n_rot = len(cache[f"L|{seeds[0]}|90.0"][0])
    early, late = slice(0, 4), slice(n_rot - 4, n_rot)

    def diff(cond):
        out = []
        for s in seeds:
            sick = np.asarray(cache[f"{cond}|{s}|90.0"][0], dtype=float)
            out.append(sick[late].mean() - sick[early].mean())
        return np.asarray(out)

    contrast = diff("L") - diff("C?")
    n = len(seeds)
    se = contrast.std(ddof=1) / np.sqrt(n)
    mde = CRIT * se
    baseline = np.mean([np.asarray(cache[f"L|{s}|90.0"][0], dtype=float).mean()
                        for s in seeds])
    print(f"{name:<12}{baseline:>14.2f}{se:>14.3f}{mde:>14.2f}"
          f"{100 * mde / baseline:>17.1f}%")

# Empirical check that the analytic MDE is right: inject a uniform shift into L's
# late window and find where the paired t crosses threshold. A uniform shift moves
# the mean without changing the SE, so this should land on t_crit*SE exactly -- if
# it does not, one of the two calculations is wrong.
if os.path.exists("scratchpad/e068_cache.json"):
    cache = json.load(open("scratchpad/e068_cache.json"))
    seeds = sorted({k.split("|")[1] for k in cache})
    n_rot = len(cache[f"L|{seeds[0]}|90.0"][0])
    early, late = slice(0, 4), slice(n_rot - 4, n_rot)

    def diff_shifted(cond, shift=0.0):
        out = []
        for s in seeds:
            sick = np.asarray(cache[f"{cond}|{s}|90.0"][0], dtype=float)
            out.append((sick[late].mean() + shift) - sick[early].mean())
        return np.asarray(out)

    n = len(seeds)
    found = None
    for shift in np.arange(0.0, 8.0, 0.01):
        c = diff_shifted("L", -shift) - diff_shifted("C?")
        t = abs(c.mean()) / (c.std(ddof=1) / np.sqrt(n) + 1e-12)
        if t > CRIT:
            found = shift
            break
    print(f"\nempirical check (E068): smallest injected reduction in L's late window "
          f"that clears threshold = {found:.2f} events/rotation")

# ---------------------------------------------------------------------------
# B -- can the rule learn it? sickness_penalty sweep
# ---------------------------------------------------------------------------
HENS = 16
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
ROTATION_S = CFG.contamination_period_s
N_ROTATIONS = 12
SEEDS = 4
PENALTIES = (0.0, 1.0, 10.0, 100.0, 1000.0)
BASE = dict(enabled=True, growth_enabled=False, kin_audible=True,
            explore_sigma=0.6, hebbian_readout=True, readout_scaling_strength=0.3)


@partial(jax.jit, static_argnames=("cfg", "pc", "n_chunks", "chunk_steps"))
def run_chunked(w, x, p, ps, key, cfg, pc, n_chunks, chunk_steps):
    def step(carry, _):
        w_prev = carry[0]
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w_next = carry[0]
        return carry, jnp.sum((w_next.sick_on & ~w_prev.sick_on).astype(jnp.float32))

    def chunk(carry, _):
        carry, onsets = jax.lax.scan(step, carry, None, length=chunk_steps)
        return carry, jnp.sum(onsets)

    return jax.lax.scan(chunk, (w, x, p, ps, key), None, length=n_chunks)


print("\n" + "=" * 72)
print(f"B -- sickness_penalty sweep ({SEEDS} seeds, {N_ROTATIONS} rotations, "
      f"{N_ROTATIONS * ROTATION_S / 60:.0f} min/run)")
print("=" * 72)
print(f"{'penalty':>9}{'early/rot':>12}{'late/rot':>11}{'late-early':>13}"
      f"{'|W-W0|':>13}{'synapses':>12}{'vs innate':>11}")

t0 = time.perf_counter()
chunk_steps = round(ROTATION_S / CFG.dt)
for pen in PENALTIES:
    pc = PlasticConfig(**BASE, sickness_penalty=pen)
    earlies, lates, drifts, syns, syn0s = [], [], [], [], []
    for seed in range(SEEDS):
        key = jax.random.key(seed)
        w = world.reset(key, CFG)
        p0 = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                              n_hens=HENS)
        x = brain.initial_state(p0, HENS)
        ps = plasticity.initial_state(p0, HENS, pc)
        (w2, x2, p2, ps2, k2), onsets = run_chunked(
            w, x, p0, ps, jax.random.fold_in(key, 2), CFG, pc,
            N_ROTATIONS, chunk_steps)
        o = np.asarray(onsets, dtype=float)
        earlies.append(o[:3].mean())
        lates.append(o[-3:].mean())
        drifts.append(float(jnp.mean(jnp.abs(p2.W - p0.W))))
        syns.append(float(jnp.mean(jnp.sum(p2.W != 0.0, axis=(1, 2)))))
        syn0s.append(float(jnp.mean(jnp.sum(p0.W != 0.0, axis=(1, 2)))))
    e, l = np.mean(earlies), np.mean(lates)
    syn_frac = 100.0 * np.mean(syns) / np.mean(syn0s)
    print(f"{pen:>9.0f}{e:>12.2f}{l:>11.2f}{l - e:>+13.2f}"
          f"{np.mean(drifts):>13.3e}{np.mean(syns):>12.0f}{syn_frac:>10.1f}%")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
print("\nReading it: a real planted effect shows as late-early going clearly negative")
print("(she gets sick less) at high penalty, with |W-W0| rising. Synapse count far")
print("below 100% of innate at high penalty is E014's erosion failure mode instead.")
