"""E120 pre-checks -- is the H0 ladder physically runnable at hawk_period_s=10?

E119 unblocked the ladder by finding predation repeatability of +0.478 at this density.
Three things must still be true before evolving an intact channel against a yoked one, and
CLAUDE.md requires every one of them to be *measured at the configuration the experiment
uses*, not inherited:

A. **Does the manipulated variable vary?**  Mean aerial-alarm amplitude heard, at rest and
   while a hawk is diving. The auditory channel read 0.999 at rest and 1.000 during an
   alarm for eighteen experiments (E019) because nobody looked.

B. **Does the control destroy what it claims to?**  Correlation between the heard aerial
   channel and "a hawk is diving on me right now", intact versus yoked. E026 measured
   +0.56 -> -0.13 at its own configuration; "any future control gets the same treatment
   before the ladder runs, never after."

C. **Is a positive result physically reachable?**  The mandatory positive control. Turn on
   `auditory_scaffold`, which hand-wires hearing-an-alarm to crouching, and measure whether
   it reduces catches at this predator density. If a hand-planted comprehension cannot
   save a hen here, selection cannot discover one either, and a null ladder would say
   nothing about the channel.

Run:  PYTHONPATH=. python scratchpad/e120_ladder_checks.py
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from coop import sensing, spec, world
from hen import brain, connectome, plasticity, regions
from run import evolve, simulate

N_SEEDS = 8
LIFETIME_S = 600.0
HAWK = 10.0
CACHE = "scratchpad/e120_checks_cache.json"

AERIAL_CH = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)


def base_cfg(mode="intact"):
    cfg = spec.DEFAULT_COOP._replace(hawk_period_s=HAWK, channel_mode=mode)
    if mode == "yoked":
        cfg = cfg._replace(call_log_steps=spec.YOKE_LOG_STEPS)
    return cfg


def founders(key, cfg, **build_kw):
    p = connectome.build(key, regions.DEFAULT_REGIONS, n_hens=cfg.n_hens, **build_kw)
    return evolve._mutate(p, jax.random.fold_in(key, 77), 2.0)


def trace_run(cfg, p, key, secs):
    """Returns (obs trace, hawk_on per step) for a short run."""
    pc = evolve.NO_LEARNING
    w0 = world.reset(jax.random.fold_in(key, 1), cfg)
    x0 = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    _, _, _, _, _, tr = simulate.rollout(
        w0, x0, p, jax.random.fold_in(key, 2), cfg, int(secs / cfg.dt), ps, pc)
    return np.asarray(tr.obs)


def hawk_on_trace(cfg, p, key, secs):
    """Re-run the same rollout stepwise enough to recover per-step hawk state.

    `rollout` does not retain world state, so this replays in chunks and reads
    `hawk_on` / `hawk_pos` at each chunk boundary. Chunks of 10 steps (0.1 s) are far
    finer than the 12 s dive being resolved.
    """
    pc = evolve.NO_LEARNING
    w = world.reset(jax.random.fold_in(key, 1), cfg)
    x = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    k = jax.random.fold_in(key, 2)
    chunk = 10
    on, dist, obs = [], [], []
    for _ in range(int(secs / cfg.dt) // chunk):
        o = np.asarray(sensing.observe(w, cfg))
        obs.append(o)
        on.append(float(w.hawk_on))
        d = np.linalg.norm(np.asarray(w.pos) - np.asarray(w.hawk_pos)[None, :], axis=1)
        dist.append(d)
        w, x, _, ps, k = simulate.rollout_quiet(w, x, p, k, cfg, chunk, ps, pc)
    return np.array(on), np.array(dist), np.array(obs)


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    print(f"E120 pre-checks at hawk_period_s={HAWK:.0f}, aerial channel index {AERIAL_CH}\n")

    # ---------------------------------------------------------------- A and B
    if "channel" not in cache:
        rows = {}
        for mode in ("intact", "yoked"):
            cfg = base_cfg(mode)
            rest, alarm, corrs = [], [], []
            for g in range(N_SEEDS):
                key = jax.random.key(2000 + g)
                p = founders(jax.random.fold_in(key, 0), cfg)
                on, dist, obs = hawk_on_trace(cfg, p, key, 180.0)
                aer = obs[:, :, AERIAL_CH]                       # (T, H)
                # "a hawk is diving on me": dive active AND I am within its reach
                on_me = (on[:, None] > 0.5) & (dist < 4.0)
                if on_me.sum() < 5 or (~on_me).sum() < 5:
                    continue
                rest.append(float(aer[~on_me].mean()))
                alarm.append(float(aer[on_me].mean()))
                a = aer.reshape(-1) - aer.mean()
                b = on_me.reshape(-1).astype(float)
                b = b - b.mean()
                den = np.sqrt((a * a).sum() * (b * b).sum())
                corrs.append(float((a * b).sum() / den) if den > 1e-12 else np.nan)
            rows[mode] = dict(rest=float(np.mean(rest)), alarm=float(np.mean(alarm)),
                              corr=float(np.nanmean(corrs)),
                              corr_sd=float(np.nanstd(corrs)),
                              per_seed_corr=corrs)
            print(f"  [{mode}] rest={rows[mode]['rest']:.4f} "
                  f"hawk-on-me={rows[mode]['alarm']:.4f} "
                  f"r={rows[mode]['corr']:+.4f}", flush=True)
        cache["channel"] = rows
        json.dump(cache, open(CACHE, "w"), indent=1)

    print("\n" + "=" * 84)
    print("A. Does the aerial channel vary?   B. Does the yoked control destroy it?")
    print("=" * 84)
    print(f"{'mode':>8} {'heard at rest':>15} {'heard hawk-on-me':>18} "
          f"{'r(heard, hawk on me)':>22}")
    for mode in ("intact", "yoked"):
        r = cache["channel"][mode]
        print(f"{mode:>8} {r['rest']:15.4f} {r['alarm']:18.4f} "
              f"{r['corr']:+.4f} +- {r['corr_sd']:.4f}")
    ci, cy = cache["channel"]["intact"]["corr"], cache["channel"]["yoked"]["corr"]
    print(f"\n  information retained by the control: {abs(cy)/max(abs(ci),1e-9)*100:.1f}%")
    print("  (E026's yoked control went +0.56 -> -0.13; E024's shuffle kept 98% and was")
    print("   not a control at all. Anything above ~25% here means the ladder's control")
    print("   does not destroy what it claims to, and the ladder must not be run.)")

    # ------------------------------------------------------------------- C
    if "scaffold" not in cache:
        rows = {}
        for name, kw in (("scaffold_off", {}),
                         ("scaffold_on", dict(auditory_scaffold=True))):
            cfg = base_cfg("intact")
            caught, per_dive, hunger, crouch = [], [], [], []
            for g in range(N_SEEDS):
                key = jax.random.key(3000 + g)
                p = founders(jax.random.fold_in(key, 0), cfg, **kw)
                pc = evolve.NO_LEARNING
                w0 = world.reset(jax.random.fold_in(key, 1), cfg)
                x0 = brain.initial_state(p, cfg.n_hens)
                ps = plasticity.initial_state(p, cfg.n_hens, pc)
                w_end, *_ = simulate.rollout_quiet(
                    w0, x0, p, jax.random.fold_in(key, 2), cfg,
                    int(LIFETIME_S / cfg.dt), ps, pc)
                c = float(jnp.mean(w_end.n_caught_any))
                dives = float(jnp.sum(w_end.n_dives))
                caught.append(c)
                per_dive.append(float(jnp.sum(w_end.n_caught_any)) / max(dives, 1.0))
                hunger.append(float(jnp.mean(w_end.hunger)))
            rows[name] = dict(caught=caught, per_dive=per_dive, hunger=hunger)
            print(f"  [{name}] caught/hen={np.mean(caught):.2f} "
                  f"hunger={np.mean(hunger):.4f}", flush=True)
        cache["scaffold"] = rows
        json.dump(cache, open(CACHE, "w"), indent=1)

    print("\n" + "=" * 84)
    print("C. Positive control -- can a HAND-WIRED comprehension save a hen here?")
    print("=" * 84)
    off = np.array(cache["scaffold"]["scaffold_off"]["caught"])
    on = np.array(cache["scaffold"]["scaffold_on"]["caught"])
    d = on - off
    se = d.std(ddof=1) / np.sqrt(len(d))
    hoff = np.mean(cache["scaffold"]["scaffold_off"]["hunger"])
    hon = np.mean(cache["scaffold"]["scaffold_on"]["hunger"])
    print(f"  caught/hen  scaffold off {off.mean():.3f}   on {on.mean():.3f}")
    print(f"  difference  {d.mean():+.3f} +- {se:.3f}  t={d.mean()/se:+.2f} "
          f"(df={len(d)-1}, bar 2.365)")
    print(f"  hunger      off {hoff:.4f}   on {hon:.4f}   "
          f"(a scaffold that only makes her crouch more will show here)")
    print("\n  If this is null, the ladder cannot show a positive and must not be run:")
    print("  a null would be about the coop, not about the channel.")


if __name__ == "__main__":
    main()
