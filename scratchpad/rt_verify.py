"""Independent check of the third review's two headline claims (E027).

Written from scratch rather than reusing the reviewer's scripts, per the red-team
skill's one rule. Fresh seeds (24+) so that no claim is re-read off the data that
generated it.

Two claims under test:

  1. The H4 effect survives lesioning `W_out` -- i.e. it does not need the pallium.
  2. The effect is carried by the head-raise half of the E018 scaffold (peck/scratch
     suppression -> she looks up -> her own visual reflex fires at 8.0), not by the
     crouch-response half. E018 section 8 pre-registered exactly this ablation and it
     was never reported.

Both estimators are printed. `pooled` is total caught / total blind-at-risk, which is
what the E026 table's rate columns show. `mean-of-ratios` is the per-seed paired
statistic E026 quoted as the headline. They are not the same number and the difference
is itself a finding.
"""
import argparse

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, innate, regions
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed-offset", type=int, default=24)
ap.add_argument("--minutes", type=float, default=5.0)
args = ap.parse_args()

AERIAL_CALL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
GROUND_CALL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GROUND)


def patch(p, *, lesion_readout=False, drop_crouch=False, drop_headraise=False):
    """Surgery on a built genome. Nothing here touches the repository."""
    if lesion_readout:
        p = p._replace(W_out=jnp.zeros_like(p.W_out))
    if drop_crouch or drop_headraise:
        r = np.asarray(p.reflex).copy()
        if drop_crouch:      # remove "hearing a call makes her crouch/flee"
            r[spec.M_CROUCH, AERIAL_CALL] = 0.0
            r[spec.M_FLEE, GROUND_CALL] = 0.0
        if drop_headraise:   # remove "hearing a call stops her pecking"
            for c in (AERIAL_CALL, GROUND_CALL):
                r[spec.M_PECK, c] = 0.0
                r[spec.M_SCRATCH, c] = 0.0
        p = p._replace(reflex=jnp.asarray(r))
    return p


CONDITIONS = (
    ("deaf",                     dict(channel="none")),
    ("intact",                   dict(channel="intact")),
    ("deaf,   W_out=0",          dict(channel="none",   lesion_readout=True)),
    ("intact, W_out=0",          dict(channel="intact", lesion_readout=True)),
    ("intact, head-raise only",  dict(channel="intact", drop_crouch=True)),
    ("intact, crouch only",      dict(channel="intact", drop_headraise=True)),
)

seeds = range(args.seed_offset, args.seed_offset + args.seeds)
print(f"E027 verification -- seeds {args.seed_offset}-{args.seed_offset+args.seeds-1}"
      f" x {args.minutes:.0f} min, 16 hens, 1.5x pallium, hawk every 20 s, no plasticity\n")
hdr = f"{'condition':<26}{'blind risks':>13}{'caught|blind':>14}{'pooled rate':>13}"
print(hdr); print("-" * len(hdr))

out = {}
for name, kw in CONDITIONS:
    per_seed = []
    for s in seeds:
        cfg = spec.DEFAULT_COOP._replace(
            n_hens=16, hawk_period_s=20.0, channel_mode=kw["channel"],
            call_log_steps=(spec.YOKE_LOG_STEPS if kw["channel"] == "yoked" else 1))
        w = world.reset(jax.random.key(s), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(s), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=16, auditory_scaffold=True)
        p = patch(p, lesion_readout=kw.get("lesion_readout", False),
                  drop_crouch=kw.get("drop_crouch", False),
                  drop_headraise=kw.get("drop_headraise", False))
        x = brain.initial_state(p, 16)
        w_end, *_ = simulate.simulate(w, x, p, jax.random.fold_in(jax.random.key(s), 2),
                                      cfg, args.minutes * 60.0, 60.0,
                                      simulate.NO_PLASTICITY)
        per_seed.append((float(jnp.sum(w_end.n_blind_risk)),
                         float(jnp.sum(w_end.n_blind_caught))))
    out[name] = per_seed
    a = np.array(per_seed)
    print(f"{name:<26}{a[:,0].sum():>13.0f}{a[:,1].sum():>14.0f}"
          f"{a[:,1].sum()/max(a[:,0].sum(),1):>13.3f}")


def ratios(name):
    return np.array([(bc / br if br else np.nan) for br, bc in out[name]])


def contrast(label, treat, ctrl):
    d = ratios(treat) - ratios(ctrl)
    d = d[~np.isnan(d)]
    n = len(d)
    mean, se = float(d.mean()), float(d.std(ddof=1)) / (n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    a, b = np.array(out[treat]), np.array(out[ctrl])
    pooled = a[:, 1].sum() / max(a[:, 0].sum(), 1) - b[:, 1].sum() / max(b[:, 0].sum(), 1)
    verdict = "SIGNIFICANT" if t > _t_critical(n - 1) else f"not significant"
    print(f"  {label:<34}{mean:+.3f} +/- {se:.3f}  t={t:.2f}  {verdict:<16}"
          f"pooled {pooled:+.3f}")


print("\n--- claim 1: does the effect need the pallium? ---")
print(f"  {'':<34}{'mean-of-ratios':<28}{'pooled'}")
contrast("intact - deaf  (pallium intact)", "intact", "deaf")
contrast("intact - deaf  (W_out lesioned)", "intact, W_out=0", "deaf,   W_out=0")

print("\n--- claim 2: which half of the scaffold does the work? ---")
contrast("full scaffold - deaf", "intact", "deaf")
contrast("head-raise only - deaf", "intact, head-raise only", "deaf")
contrast("crouch response only - deaf", "intact, crouch only", "deaf")

print("\n--- denominator movement (is the metric's premise true?) ---")
base = np.array(out["deaf"])[:, 0].sum()
for name, _ in CONDITIONS:
    n = np.array(out[name])[:, 0].sum()
    print(f"  {name:<26}{n:>7.0f} blind risks   {100*(n-base)/max(base,1):+.1f}% vs deaf")
