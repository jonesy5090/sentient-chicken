"""E030: the third block, pre-registered. See docs/experiments/E030-*.md.

Three conditions only. The analysis is fixed in the pre-registration: the primary test
is the POOLED 36-seed paired contrast on caught/dive across blocks A (36-47),
B (48-59) and C (60-71), threshold t=2.030. Per-block figures are reported for
transparency and do not move the tree on their own.
"""
import argparse, json, os, time
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from run.experiment import Condition, _t_critical
from run.h4 import EXPANDED, INNATE, H4Result, _cache_load, _cache_save, run_condition

CONDS = (
    Condition("C? yoked", INNATE,
              cfg_patch=(("channel_mode", "yoked"),
                         ("call_log_steps", spec.YOKE_LOG_STEPS)),
              pallium_scale=EXPANDED, scaffold=True),
    Condition("L  language", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True),
    Condition("Lx lesioned", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True, lesion_readout=True),
)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=12)
ap.add_argument("--seed-offset", type=int, default=60)
ap.add_argument("--minutes", type=float, default=10.0)
ap.add_argument("--budget", type=float, default=520.0)
ap.add_argument("--cache", default="scratchpad/e030_cache.json")
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=20.0)
seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
cache = _cache_load(a.cache); t0 = time.perf_counter(); ran = 0
for c in CONDS:
    for sd in seeds:
        k = f"{c.name}|{sd}|{a.minutes}"
        if k in cache:
            continue
        if time.perf_counter() - t0 > a.budget:
            print(f"budget reached; {len(cache)}/{len(CONDS)*len(seeds)} cached")
            raise SystemExit(0)
        cache[k] = list(run_condition(c, sd, cfg, a.minutes * 60.0))
        _cache_save(a.cache, cache); ran += 1
print(f"complete: {len(cache)}/{len(CONDS)*len(seeds)} cells ({ran} this pass)")
