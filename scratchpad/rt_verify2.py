"""E027 verification, checkpointed. Replaces rt_verify.py, which was killed mid-run.

Same two claims, but per-seed results are flushed to JSON after every single run, so
a container reclamation costs one seed rather than the whole block. `rt_verify.py`
died after four of six conditions and took every paired statistic with it -- only the
pooled counts had been printed.

Claims under test (review 3):
  1. the H4 effect survives lesioning `W_out`, i.e. it does not need the pallium
  2. the effect is carried by the head-raise half of the E018 scaffold, not the
     crouch response -- pre-registered in E018 section 8 and never reported

Fresh seeds (24+), so neither claim is re-read off the data that generated it.
"""
import argparse, json, os

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, regions
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed-offset", type=int, default=24)
ap.add_argument("--minutes", type=float, default=5.0)
ap.add_argument("--cache", default="scratchpad/e027_cache.json")
args = ap.parse_args()

AERIAL_CALL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
GROUND_CALL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GROUND)

CONDITIONS = (
    ("deaf",                    dict(channel="none")),
    ("intact",                  dict(channel="intact")),
    ("intact, head-raise only", dict(channel="intact", drop_crouch=True)),
    ("intact, crouch only",     dict(channel="intact", drop_headraise=True)),
    ("deaf,   W_out=0",         dict(channel="none",   lesion=True)),
    ("intact, W_out=0",         dict(channel="intact", lesion=True)),
)

cache = {}
if os.path.exists(args.cache):
    cache = json.load(open(args.cache))


def run_one(kw, s):
    cfg = spec.DEFAULT_COOP._replace(
        n_hens=16, hawk_period_s=20.0, channel_mode=kw["channel"],
        call_log_steps=(spec.YOKE_LOG_STEPS if kw["channel"] == "yoked" else 1))
    w = world.reset(jax.random.key(s), cfg)
    p = connectome.build(jax.random.fold_in(jax.random.key(s), 1),
                         regions.DEFAULT_REGIONS.with_pallium(1.5),
                         n_hens=16, auditory_scaffold=True)
    if kw.get("lesion"):
        p = p._replace(W_out=jnp.zeros_like(p.W_out))
    if kw.get("drop_crouch") or kw.get("drop_headraise"):
        r = np.asarray(p.reflex).copy()
        if kw.get("drop_crouch"):        # no "hearing a call makes her crouch"
            r[spec.M_CROUCH, AERIAL_CALL] = 0.0
            r[spec.M_FLEE, GROUND_CALL] = 0.0
        if kw.get("drop_headraise"):     # no "hearing a call stops her pecking"
            for c in (AERIAL_CALL, GROUND_CALL):
                r[spec.M_PECK, c] = 0.0
                r[spec.M_SCRATCH, c] = 0.0
        p = p._replace(reflex=jnp.asarray(r))
    x = brain.initial_state(p, 16)
    w_end, *_ = simulate.simulate(w, x, p, jax.random.fold_in(jax.random.key(s), 2),
                                  cfg, args.minutes * 60.0, 60.0,
                                  simulate.NO_PLASTICITY)
    return (float(jnp.sum(w_end.n_blind_risk)), float(jnp.sum(w_end.n_blind_caught)))


seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
for name, kw in CONDITIONS:
    for s in seeds:
        key = f"{name}|{s}|{args.minutes}"
        if key in cache:
            continue
        cache[key] = run_one(kw, s)
        json.dump(cache, open(args.cache, "w"))
        print(f"  ran {name:<26} seed {s}", flush=True)

print(f"\nE027 -- seeds {seeds[0]}-{seeds[-1]} x {args.minutes:.0f} min, 16 hens, "
      f"1.5x pallium, hawk every 20 s, no plasticity\n")
hdr = f"{'condition':<26}{'blind risks':>13}{'caught':>9}{'pooled rate':>13}"
print(hdr); print("-" * len(hdr))
res = {n: np.array([cache[f"{n}|{s}|{args.minutes}"] for s in seeds])
       for n, _ in CONDITIONS}
for name, _ in CONDITIONS:
    a = res[name]
    print(f"{name:<26}{a[:,0].sum():>13.0f}{a[:,1].sum():>9.0f}"
          f"{a[:,1].sum()/max(a[:,0].sum(),1):>13.3f}")


def contrast(label, treat, ctrl):
    a, b = res[treat], res[ctrl]
    ra = np.array([bc/br if br else np.nan for br, bc in a])
    rb = np.array([bc/br if br else np.nan for br, bc in b])
    d = (ra - rb)[~(np.isnan(ra) | np.isnan(rb))]
    n = len(d)
    mean, se = float(d.mean()), float(d.std(ddof=1))/(n**0.5)
    t = abs(mean)/(se+1e-12)
    pooled = a[:,1].sum()/max(a[:,0].sum(),1) - b[:,1].sum()/max(b[:,0].sum(),1)
    verdict = "SIGNIFICANT" if t > _t_critical(n-1) else "not significant"
    print(f"  {label:<34}{mean:+.3f} +/- {se:.3f} t={t:.2f} {verdict:<17}"
          f"pooled {pooled:+.3f}")


print("\n--- claim 1: does the effect need the pallium? ---")
contrast("intact - deaf (pallium intact)", "intact", "deaf")
contrast("intact - deaf (W_out lesioned)", "intact, W_out=0", "deaf,   W_out=0")

print("\n--- claim 2: which half of the scaffold does the work? ---")
contrast("full scaffold - deaf", "intact", "deaf")
contrast("head-raise only - deaf", "intact, head-raise only", "deaf")
contrast("crouch response only - deaf", "intact, crouch only", "deaf")

print("\n--- denominator movement vs deaf ---")
base = res["deaf"][:,0].sum()
for name, _ in CONDITIONS:
    n = res[name][:,0].sum()
    print(f"  {name:<26}{n:>7.0f}  {100*(n-base)/max(base,1):+6.1f}%")
