"""E061: T2 Stage 1b -- does the contamination/sickness/gakel scaffold work at the
population level, not just in E060's isolated staged scenarios? Three independent
checks: discovery rate, gakel-call audibility (E024/E026's own instrument), and
anchor-driven dispersal (E025/E048's own instrument). No learning anywhere.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

HENS = 16
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
GAKEL_CH = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GAKEL)


def strip_anchor(p):
    """Zero the CLS_SICK -> turn-away reflex weights, isolation from E048's
    strip_channel/strip_wall_escape helpers.
    """
    r = np.asarray(p.reflex).copy()
    for b in range(spec.N_BINS):
        r[:, spec.vis_index(b, spec.CLS_SICK)] = 0.0
    return p._replace(reflex=jnp.asarray(r))


# ---------------------------------------------------------------------------
# Check 1: discovery rate, natural contamination, 20 min, 3 seeds
# ---------------------------------------------------------------------------
STEPS_20MIN = 120_000


@partial(jax.jit, static_argnames=("cfg",))
def run_discovery(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        w_next = world.step(w, motor, k, cfg)
        onset = (w_next.sick_on & ~w.sick_on).astype(jnp.float32)
        return (w_next, x, key), (onset.sum(), w_next.sick_on.any())
    return jax.lax.scan(step, (w, x, key), None, length=STEPS_20MIN)[1]


print("=== Check 1: discovery rate (natural contamination, 20 min) ===")
onset_counts = []
for seed in range(3):
    key = jax.random.key(seed)
    w = world.reset(key, CFG)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=HENS)
    x = brain.initial_state(p, HENS)
    onsets, _ = run_discovery(w, x, p, jax.random.key(99), CFG)
    total = float(jnp.sum(onsets))
    onset_counts.append(total)
    print(f"  seed {seed}: {total:.0f} sickness-onset events over 20 min")
print(f"  mean: {np.mean(onset_counts):.1f} events / 20 min "
      f"(contamination_period_s={CFG.contamination_period_s:.0f}s)")


# ---------------------------------------------------------------------------
# Check 2: gakel-call audibility (E024/E026 instrument)
# ---------------------------------------------------------------------------
STEPS_3MIN = 18_000


@partial(jax.jit, static_argnames=("cfg",))
def trace_audibility(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        d = jnp.linalg.norm(w.pos[:, None, :] - w.pos[None, :, :], axis=-1)
        d = d + jnp.eye(cfg.n_hens) * 1e6
        nearby_sick = jnp.any((d < cfg.hear_range) & w.sick_on[None, :], axis=-1)
        w = world.step(w, motor, k, cfg)
        return (w, x, key), (obs[:, GAKEL_CH], nearby_sick)
    return jax.lax.scan(step, (w, x, key), None, length=STEPS_3MIN)[1]


print("\n=== Check 2: gakel-call audibility (E024/E026 instrument) ===")
print(f"{'mode':<10}{'corr(heard, nearby sick)':>26}{'heard|sick':>12}"
      f"{'heard|not':>11}{'ratio':>8}")
results = {}
for mode in ("intact", "shuffled"):
    cfg = CFG._replace(channel_mode=mode)
    cs, wh, wo = [], [], []
    for seed in range(3):
        key = jax.random.key(seed)
        w = world.reset(key, cfg)
        p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=HENS)
        x = brain.initial_state(p, HENS)
        heard, nearby = trace_audibility(w, x, p, jax.random.key(99), cfg)
        h = np.asarray(heard).ravel()
        n = np.asarray(nearby).ravel()
        if n.sum() and h.std() > 0:
            cs.append(np.corrcoef(h, n.astype(float))[0, 1])
            wh.append(h[n].mean()); wo.append(h[~n].mean())
    a, b = (np.mean(wh), np.mean(wo)) if wh else (float("nan"), float("nan"))
    results[mode] = (np.mean(cs) if cs else float("nan"), a, b)
    print(f"{mode:<10}{results[mode][0]:>26.4f}{a:>12.4f}{b:>11.4f}{a/max(b,1e-9):>8.2f}")

if not np.isnan(results["intact"][0]) and results["intact"][0] != 0:
    retained = results["shuffled"][0] / results["intact"][0]
    print(f"  shuffled retains {100*retained:.0f}% of intact's correlation "
          f"(compare E024's 90%, E026's 98% for the alarm channel)")
else:
    print("  too few nearby-sick samples to compute a correlation -- see Check 1's "
        "discovery rate first")


# ---------------------------------------------------------------------------
# Check 3: anchor-driven dispersal (E025/E048 ablation instrument)
# Sickness is forced periodically on a random hen, decoupling this check from
# Check 1's own discovery-rate question -- this isolates "does the anchor work",
# not "does contamination get found often enough".
# ---------------------------------------------------------------------------
FORCE_PERIOD_STEPS = 6_000   # 60s of chicken time between forced sickness events


@partial(jax.jit, static_argnames=("cfg",))
def run_dispersal(w, x, p, key, cfg):
    def step(carry, t):
        w, x, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        force_now = (t % FORCE_PERIOD_STEPS) == 0
        target = jax.random.randint(k1, (), 0, cfg.n_hens)
        sick_t = jnp.where(force_now, w.sick_t.at[target].set(cfg.sickness_duration_s),
                           w.sick_t)
        sick_on = sick_t > 0.0
        w = w._replace(sick_t=sick_t, sick_on=sick_on)

        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        d_hens = jnp.linalg.norm(w.pos[:, None, :] - w.pos[None, :, :], axis=-1)
        d_hens = d_hens + jnp.eye(cfg.n_hens) * 1e6
        # Distance from every hen to the nearest currently-sick hen. Only meaningful
        # for the *healthy* rows -- a sick hen's own row is her distance to another
        # sick hen if one exists, or 1e6 (self-masked) otherwise, neither of which
        # says anything about dispersal and must not enter the average outside.
        d_to_sick = jnp.min(jnp.where(w.sick_on[None, :], d_hens, 1e6), axis=-1)
        w = world.step(w, motor, k2, cfg)
        return (w, x, key), (d_to_sick, w.sick_on)
    return jax.lax.scan(step, (w, x, key), jnp.arange(STEPS_20MIN))[1]


print("\n=== Check 3: anchor-driven dispersal from a sick hen (E025/E048 instrument) ===")
print(f"{'condition':<22}{'mean NN dist to sick hen':>26}")
for label, strip in (("anchor present", False), ("anchor stripped", True)):
    dists = []
    for seed in range(3):
        key = jax.random.key(seed)
        w = world.reset(key, CFG)
        p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=HENS)
        if strip:
            p = strip_anchor(p)
        x = brain.initial_state(p, HENS)
        d_to_sick, sick_on = run_dispersal(w, x, p, jax.random.key(99), CFG)
        d, sick = np.asarray(d_to_sick), np.asarray(sick_on)   # (T,H) each
        any_sick_step = sick.any(axis=-1)                      # (T,)
        healthy = ~sick                                        # (T,H)
        mask = healthy & any_sick_step[:, None]                 # only healthy hens,
                                                                 # only when someone is sick
        dists.append(d[mask].mean() if mask.sum() else float("nan"))
    print(f"{label:<22}{np.nanmean(dists):>26.3f} m")

print("\nlarger 'anchor present' distance than 'anchor stripped' = real dispersal.")
