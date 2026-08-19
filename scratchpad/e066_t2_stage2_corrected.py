"""E066: T2 Stage 2, corrected -- E065's exact question, re-run with a genuine reward
signal for sickness (hen/plasticity.py's reward() had none; E065's null was very
likely not a fair test). Also splits each sickness onset by whether another
already-sick hen was within vision_range at that moment ("witnessed", explainable by
the innate anchor reflex alone, identical across all three conditions) versus not
("testimony-only", the only kind of case the auditory channel could plausibly help
with) -- a pre-registered secondary analysis addressing the concern that the innate
reflex could dilute a real but small call-specific effect in the aggregate metric.

Chunked at exactly one contamination rotation per chunk (300s), so each chunk's
sickness-onset sum *is* that rotation's count -- no epoch-detection needed.

Mandatory diagnostics (CLAUDE.md's own rule, applied before trusting any positive
result): |W_out| drift (is the rule even active?) and a matched control metric (water
intake, no mechanistic route to the gakel channel).
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
ap.add_argument("--cache", default="scratchpad/e066_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

HENS = 16
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
ROTATION_S = CFG.contamination_period_s          # 300.0
N_ROTATIONS = round(a.minutes * 60.0 / ROTATION_S)
EARLY, LATE = slice(0, 4), slice(N_ROTATIONS - 4, N_ROTATIONS)

H2F_RULE = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                         explore_sigma=0.6, hebbian_readout=True,
                         readout_scaling_strength=0.3, sickness_penalty=1.0)
FIXED = PlasticConfig(enabled=False, explore_sigma=0.0, sickness_penalty=1.0)

CONDITIONS = {
    "S": (FIXED, "intact"),
    "C?": (H2F_RULE, "yoked"),
    "L": (H2F_RULE, "intact"),
}


class T2Summary(NamedTuple):
    sick_onsets: jax.Array      # (C,) flock-wide sickness onsets this rotation
    witnessed: jax.Array        # (C,) of those, another sick hen was within vision_range
    testimony_only: jax.Array   # (C,) of those, no visible sick hen at onset
    n_drunk: jax.Array          # (C,) cumulative flock water pecks at rotation end
    w_out_norm: jax.Array       # (C,) mean |W_out| at rotation end


@partial(jax.jit, static_argnames=("cfg", "pc", "n_chunks", "chunk_steps"))
def _chunked_t2(w, x, p, ps, key, cfg, pc, n_chunks: int, chunk_steps: int):
    def step(carry, _):
        w_prev = carry[0]
        carry, _out = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w_next = carry[0]
        newly_sick = w_next.sick_on & ~w_prev.sick_on            # (H,)

        d = jnp.linalg.norm(w_next.pos[:, None, :] - w_next.pos[None, :, :], axis=-1)
        d = d + jnp.eye(cfg.n_hens) * 1e6
        # Was a flockmate who was ALREADY sick (before this transition) visually
        # within range at the moment this hen fell sick? Uses w_prev.sick_on so a
        # simultaneous double-onset doesn't count each hen as the other's witness.
        nearby_sick = jnp.any((d < cfg.vision_range) & w_prev.sick_on[None, :], axis=-1)

        witnessed = jnp.sum((newly_sick & nearby_sick).astype(jnp.float32))
        testimony_only = jnp.sum((newly_sick & ~nearby_sick).astype(jnp.float32))
        return carry, (witnessed, testimony_only)

    def chunk(carry, _):
        carry, (witnessed, testimony_only) = jax.lax.scan(
            step, carry, None, length=chunk_steps)
        w, x, p, ps, key = carry
        w_sum, t_sum = jnp.sum(witnessed), jnp.sum(testimony_only)
        s = T2Summary(sick_onsets=w_sum + t_sum, witnessed=w_sum,
                     testimony_only=t_sum, n_drunk=jnp.sum(w.n_drunk),
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
    return [summary.sick_onsets.tolist(), summary.witnessed.tolist(),
           summary.testimony_only.tolist(), summary.n_drunk.tolist(),
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

sick = {name: jnp.array([r[0] for r in rows[name]]) for name in CONDITIONS}
witnessed = {name: jnp.array([r[1] for r in rows[name]]) for name in CONDITIONS}
testimony = {name: jnp.array([r[2] for r in rows[name]]) for name in CONDITIONS}
drunk = {name: jnp.array([r[3] for r in rows[name]]) for name in CONDITIONS}
wout = {name: jnp.array([r[4] for r in rows[name]]) for name in CONDITIONS}

print(f"E066 -- T2 Stage 2 corrected, {a.seeds} seeds, {a.minutes:.0f} min "
      f"({N_ROTATIONS} rotations of {ROTATION_S:.0f}s each)\n")

n = a.seeds
crit = _t_critical(n - 1)


def one_sample(d, label):
    mean, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    print(f"{label:<58}{mean:+.4f} +/- {se:.4f}  t={t:.2f}  "
         f"{'SIGNIFICANT' if t > crit else 'not significant'}")
    return mean, se, t


def diff_in_diff(arr, label_prefix, print_early_late=True):
    d = {}
    for name in CONDITIONS:
        early = arr[name][:, EARLY].mean(axis=-1)
        late = arr[name][:, LATE].mean(axis=-1)
        d[name] = late - early
        if print_early_late:
            print(f"  {name}: early={float(early.mean()):.3f}  late={float(late.mean()):.3f}")
    print()
    one_sample(d["L"], f"{label_prefix} L: late - early")
    one_sample(d["C?"], f"{label_prefix} C?: late - early")
    one_sample(d["S"], f"{label_prefix} S: late - early")
    print()
    contrast = d["L"] - d["C?"]
    one_sample(contrast, f"{label_prefix} (L late-early) - (C? late-early)")
    return contrast


print(f"threshold(df={n-1})={crit:.3f}\n")

print("=== PRIMARY: total sickness-per-rotation, early (rot 1-4) vs late (rot N-3..N) ===")
diff_in_diff(sick, "PRIMARY")

print("\n=== SECONDARY A: witnessed onsets (explainable by the innate anchor alone) ===")
diff_in_diff(witnessed, "witnessed")

print("\n=== SECONDARY B: testimony-only onsets (only the auditory channel could help) ===")
diff_in_diff(testimony, "testimony-only")

print("\n=== Diagnostic 1: is the rule active? mean |W_out| drift ===")
for name in ("C?", "L"):
    print(f"  {name}: final |W_out| = {float(wout[name][:, -1].mean()):.4f}")

print("\n=== Diagnostic 2: matched control -- water intake, same early/late contrast ===")
drunk_diff = {}
for name in CONDITIONS:
    early_total = drunk[name][:, 3]
    late_total = drunk[name][:, -1] - drunk[name][:, N_ROTATIONS - 5]
    drunk_diff[name] = (late_total / 4.0) - (early_total / 4.0)
one_sample(drunk_diff["L"] - drunk_diff["C?"],
          "control: (L late-early) - (C? late-early), water intake")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
