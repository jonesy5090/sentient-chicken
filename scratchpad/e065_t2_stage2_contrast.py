"""E065: T2 Stage 2 -- does a flock with H2f's validated learning rule reduce its
sickness rate over developmental time more under the intact gakel channel (L) than
under a yoked, decorrelated one (C?), and does either differ from a fixed-connectome
baseline (S)? Chunked at exactly one contamination rotation per chunk (300s), so each
chunk's sickness-onset sum *is* that rotation's count -- no epoch-detection needed.

Mandatory diagnostics (CLAUDE.md's own rule, applied before trusting any positive
result): |W_out| drift (is the rule even active?) and a matched control metric (water
intake, no mechanistic route to the gakel channel) on the same early/late contrast.
"""
import argparse, json, os, time
from functools import partial
from typing import NamedTuple

import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed-offset", type=int, default=0)
ap.add_argument("--minutes", type=float, default=90.0)
ap.add_argument("--cache", default="scratchpad/e065_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

HENS = 16
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
ROTATION_S = CFG.contamination_period_s          # 300.0
N_ROTATIONS = round(a.minutes * 60.0 / ROTATION_S)
EARLY, LATE = slice(0, 4), slice(N_ROTATIONS - 4, N_ROTATIONS)

H2F_RULE = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                         explore_sigma=0.6, hebbian_readout=True,
                         readout_scaling_strength=0.3)
FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)

CONDITIONS = {
    "S": (FIXED, "intact"),
    "C?": (H2F_RULE, "yoked"),
    "L": (H2F_RULE, "intact"),
}


class T2Summary(NamedTuple):
    sick_onsets: jax.Array   # (C,) flock-wide sickness onsets this rotation
    n_drunk: jax.Array       # (C,) cumulative flock water pecks at rotation end
    w_out_norm: jax.Array    # (C,) mean |W_out| at rotation end


@partial(jax.jit, static_argnames=("cfg", "pc", "n_chunks", "chunk_steps"))
def _chunked_t2(w, x, p, ps, key, cfg, pc, n_chunks: int, chunk_steps: int):
    def step(carry, _):
        w_prev = carry[0]
        carry, _out = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w_next = carry[0]
        onset = jnp.sum((w_next.sick_on & ~w_prev.sick_on).astype(jnp.float32))
        return carry, onset

    def chunk(carry, _):
        carry, onsets = jax.lax.scan(step, carry, None, length=chunk_steps)
        w, x, p, ps, key = carry
        s = T2Summary(sick_onsets=jnp.sum(onsets), n_drunk=jnp.sum(w.n_drunk),
                     w_out_norm=jnp.mean(jnp.abs(p.W_out)))
        return carry, s

    return jax.lax.scan(chunk, (w, x, p, ps, key), None, length=n_chunks)


def run_one(seed, pc, channel_mode):
    key = jax.random.key(seed)
    cfg = CFG._replace(channel_mode=channel_mode,
                       call_log_steps=(spec.YOKE_LOG_STEPS if channel_mode == "yoked" else 1))
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=HENS)
    x = brain.initial_state(p, HENS)
    ps = plasticity.initial_state(p, HENS, pc)
    chunk_steps = round(ROTATION_S / cfg.dt)
    _carry, summary = _chunked_t2(w, x, p, ps, jax.random.fold_in(key, 2), cfg, pc,
                                  N_ROTATIONS, chunk_steps)
    return [summary.sick_onsets.tolist(), summary.n_drunk.tolist(),
           summary.w_out_norm.tolist()]


cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()

rows = {name: [] for name in CONDITIONS}
for name, (pc, mode) in CONDITIONS.items():
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        ck = f"{name}|{s}|{a.minutes}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            cache[ck] = run_one(s, pc, mode)
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

sick = {name: jnp.array([r[0] for r in rows[name]]) for name in CONDITIONS}   # (seeds, N_ROTATIONS)
drunk = {name: jnp.array([r[1] for r in rows[name]]) for name in CONDITIONS}
wout = {name: jnp.array([r[2] for r in rows[name]]) for name in CONDITIONS}

print(f"E065 -- T2 Stage 2, {a.seeds} seeds, {a.minutes:.0f} min "
      f"({N_ROTATIONS} rotations of {ROTATION_S:.0f}s each)\n")

n = a.seeds
crit = _t_critical(n - 1)


def one_sample(d, label):
    mean, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    print(f"{label:<58}{mean:+.4f} +/- {se:.4f}  t={t:.2f}  "
         f"{'SIGNIFICANT' if t > crit else 'not significant'}")
    return mean, se, t


print(f"threshold(df={n-1})={crit:.3f}\n")

print("=== Primary: sickness-per-rotation, early (rot 1-4) vs late (rot N-3..N) ===")
sick_diff = {}
for name in CONDITIONS:
    early = sick[name][:, EARLY].mean(axis=-1)
    late = sick[name][:, LATE].mean(axis=-1)
    sick_diff[name] = late - early
    print(f"  {name}: early={float(early.mean()):.3f}  late={float(late.mean()):.3f}")

print()
one_sample(sick_diff["L"], "L: late - early")
one_sample(sick_diff["C?"], "C?: late - early")
one_sample(sick_diff["S"], "S: late - early")
print()
contrast = sick_diff["L"] - sick_diff["C?"]   # paired -- same seeds across conditions
one_sample(contrast, "PRIMARY: (L late-early) - (C? late-early)")

print("\n=== Diagnostic 1: is the rule active? mean |W_out| drift ===")
for name in ("C?", "L"):
    print(f"  {name}: final |W_out| = {float(wout[name][:, -1].mean()):.4f}")

print("\n=== Diagnostic 2: matched control -- water intake, same early/late contrast ===")
drunk_diff = {}
for name in CONDITIONS:
    # n_drunk is cumulative; total drunk in a window = cumulative at window end minus
    # cumulative just before the window starts.
    early_total = drunk[name][:, 3] - jnp.where(True, 0.0, 0.0)  # cumulative through rot 4
    late_total = drunk[name][:, -1] - drunk[name][:, N_ROTATIONS - 5]
    drunk_diff[name] = (late_total / 4.0) - (early_total / 4.0)
one_sample(drunk_diff["L"] - drunk_diff["C?"],
          "control: (L late-early) - (C? late-early), water intake")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
