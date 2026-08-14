"""Staged predator trials: does hearing an alarm actually save a hen?

**Why this exists.** Risk in a free-running flock turned out to be unmeasurable
(E026). Three metrics gave three different orderings of the same conditions:

    metric                          no channel   full scaffold   response only
    raw strikes per hen                   22.5            10.4             203
    struck / exposure-steps              0.459           0.048           0.280
    P(caught | at risk at onset)         0.250           0.350           0.517

Raw counts are dominated by where the flock happened to be standing. Dividing by
exposure imports a behavioural confound -- crouching zeroes locomotion, so a
partially-crouching hen lingers in the strike radius and inflates her own
denominator. Anchoring to dive onset fixes that and leaves **0.8 to 9.0 at-risk
events per run**, which is too few to divide by, and its denominator still moves 11x
across conditions because flock density changes where a hawk's aim lands.

None of those is a metric problem that a better ratio solves. The problem is asking a
causal question of a handful of chance encounters whose geometry the treatment
perturbs. So: stage the encounter instead. Fix the hen, fix the hawk, fix the
distance, run many identical trials and vary exactly one thing.

This is the design `run/probes.py` already uses for the seven ethogram behaviours,
and it is how the head-raise question should have been asked from the start.

    usage:  python -m run.riskassay --trials 60
"""

import argparse
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, innate, regions
from run import simulate
from run.experiment import _t_critical

ABSENT = 1e4
# The focal hen forages here; the caller stands within earshot but far enough that a
# hawk over one is not automatically over the other.
FOCAL = (10.0, 10.0)
CALLER = (14.0, 10.0)


class Trial(NamedTuple):
    valid: bool           # was she actually head-down when the hawk committed?
    struck: bool          # was the focal hen hit during the dive?
    crouched: bool        # did she ever cross the hiding threshold?
    saw_hawk: bool        # did the aerial channel ever open for her?
    heard: float          # peak alarm-channel input


def _variant(p, keep: str):
    """Split the E018 auditory scaffold into its two halves.

    'respond'   -- crouch/flee only; the call must supply the response itself.
    'interrupt' -- head-raise only; the call can only stop her foraging.
    """
    r = np.asarray(p.reflex).copy()
    aer = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    gnd = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GROUND)
    if keep in ("interrupt", "none"):
        r[spec.M_CROUCH, aer] = 0.0
        r[spec.M_FLEE, gnd] = 0.0
    if keep in ("respond", "none"):
        for c in (aer, gnd):
            r[spec.M_PECK, c] = 0.0
            r[spec.M_SCRATCH, c] = 0.0
    return p._replace(reflex=jnp.asarray(r))


def trial(dist: float, jitter: jax.Array, *, channel: str, keep: str,
          scaffold: bool, settle: int = 300, steps: int = 400) -> Trial:
    """One staged dive, in two phases. The focal hen is head-down and blind.

    **The two phases are not optional.** The first version placed the hawk live at
    step 0 and got identical numbers in all five conditions: `saw hawk` was 100% and
    `struck` was 62.5% everywhere. Two reasons, both fatal, both fixed by settling
    first. `head_down` is derived from the *previous* motor vector and is zero at
    reset, so the aerial gate was wide open and the "blind" hen could see perfectly.
    And a hawk already inside the strike radius at step 0 lands a hit within a few
    milliseconds, long before any response -- heard or seen -- can develop, so the
    assay measured geometry rather than behaviour.

    So: phase one has no hawk at all and lets her settle into foraging until the gate
    is genuinely shut. Phase two introduces the hawk and counts only strikes from
    that moment.

    `dist` is how far the hawk comes down from the focal hen. Below
    `hawk_strike_radius` she is in real danger; the sweep spans both sides of it.

    Food is placed under her beak so she is genuinely foraging -- head-down, and so
    with the aerial channel gated shut. That is H1a made explicit: the caller has
    information she cannot get by looking, which is the whole premise of the project.
    """
    cfg = spec.DEFAULT_COOP._replace(n_hens=2, channel_mode=channel)
    w = world.reset(jax.random.key(0), cfg)
    off = jnp.asarray(jitter, dtype=jnp.float32)
    w = w._replace(
        pos=jnp.asarray([FOCAL, CALLER], dtype=jnp.float32),
        heading=jnp.zeros((2,)),
        # Food under the focal hen only: she forages, the caller watches the sky.
        food_pos=jnp.full((cfg.n_food, 2), ABSENT).at[0].set(
            jnp.asarray([FOCAL[0] + 0.15, FOCAL[1]])),
        water_pos=jnp.full((cfg.n_water, 2), ABSENT),
        hawk_pos=jnp.asarray([ABSENT, ABSENT]),
        hawk_on=jnp.array(0.0),
        hawk_t=jnp.array(0.0),
        fox_pos=jnp.asarray([ABSENT, ABSENT]),
        fox_on=jnp.array(0.0),
        hunger=jnp.full((2,), 0.8),        # hungry, so she really does keep her head down
        thirst=jnp.full((2,), 0.2),
        cold=jnp.full((2,), 0.2),
    )
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                         n_hens=2, auditory_scaffold=scaffold)
    p = _variant(p, keep)
    x = brain.initial_state(p, 2)

    # Phase 1: no hawk, and we wait for an actual peck rather than assuming one.
    #
    # A hen cannot hold a head-down posture here: TONIC_FORWARD plus the hunger->
    # forward drive walk her off the patch within a second or two, after which
    # `head_down` is ~0.5 from scratching alone and the aerial gate is only half
    # shut -- she sees enough of a hawk at 1 m to crouch, and the assay has no
    # asymmetry to test. Measured: after 3 s of settling, peck 0.08, scratch 0.50,
    # 0.73 m from the food she started on.
    #
    # `run/probes.py::head_down_blindness` already handles this by conditioning on
    # posture per step rather than staging a steady state. Same approach: advance in
    # short chunks until she is genuinely pecking, and drop the hawk then. Trials
    # where that never happens are reported and excluded rather than silently kept.
    pecking = False
    for _ in range(max(settle // 20, 1)):
        w, x, _p, _ps, _k, tr0 = simulate.rollout(
            w, x, p, jax.random.key(2), cfg, 20)
        if float(tr0.motor[-1, 0, spec.M_PECK]) > 0.5:
            pecking = True
            break

    # Phase 2: the hawk commits, while her beak is down. Strikes counted from here.
    struck_before = float(w.n_struck[0])
    w = w._replace(hawk_pos=jnp.asarray([FOCAL[0], FOCAL[1]]) + off * dist,
                   hawk_on=jnp.array(1.0), hawk_t=jnp.array(1e4))
    w_end, _x, _p, _ps, _k, tr = simulate.rollout(
        w, x, p, jax.random.key(3), cfg, steps)
    aer = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    return Trial(
        valid=pecking,
        struck=bool(float(w_end.n_struck[0]) - struck_before > 0),
        crouched=bool(jnp.max(tr.motor[:, 0, spec.M_CROUCH]) > 0.5),
        saw_hawk=bool(jnp.max(tr.obs[:, 0, spec.IDX_AERIAL]) > 0.05),
        heard=float(jnp.max(tr.obs[:, 0, aer])),
    )


CONDITIONS = (
    ("deaf (no channel)",            dict(channel="none",   keep="both",      scaffold=True)),
    ("hears, full scaffold",         dict(channel="intact", keep="both",      scaffold=True)),
    ("hears, response only",         dict(channel="intact", keep="respond",   scaffold=True)),
    ("hears, head-raise only",       dict(channel="intact", keep="interrupt", scaffold=True)),
    ("hears, no scaffold at all",    dict(channel="intact", keep="both",      scaffold=False)),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60,
                    help="jittered dive directions per distance")
    args = ap.parse_args()

    dists = (0.5, 1.0, 1.5, 2.0)
    key = jax.random.key(11)
    angles = jax.random.uniform(key, (args.trials,), maxval=2 * jnp.pi)
    jitters = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    print(f"staged predator trials: {args.trials} dive directions x {len(dists)} "
          f"distances = {args.trials * len(dists)} per condition")
    print("focal hen settles onto food for 3 s first, so the aerial gate is genuinely")
    print("shut when the hawk commits; caller 4 m away is head-up and can see.")
    print(f"strike radius {spec.DEFAULT_COOP.hawk_strike_radius} m\n")

    hdr = (f"{'condition':<26}{'struck %':>10}{'crouched %':>12}{'saw hawk %':>12}"
           f"{'heard':>8}{'valid':>10}")
    print(hdr); print("-" * len(hdr))

    out = {}
    for name, kw in CONDITIONS:
        all_rs = [trial(float(d), jitters[i], **kw)
                  for d in dists for i in range(args.trials)]
        rs = [r for r in all_rs if r.valid]
        out[name] = rs
        if not rs:
            print(f"{name:<26}  NO VALID TRIALS -- she was never head-down at onset")
            continue
        f = lambda g: 100 * float(np.mean([g(r) for r in rs]))
        print(f"{name:<26}{f(lambda r: r.struck):>10.1f}{f(lambda r: r.crouched):>12.1f}"
              f"{f(lambda r: r.saw_hawk):>12.1f}"
              f"{np.mean([r.heard for r in rs]):>8.3f}"
              f"{100*len(rs)/len(all_rs):>9.0f}%")

    base = np.array([r.struck for r in out["deaf (no channel)"]], dtype=float)
    n = len(base)
    if n < 2:
        print("\nno usable trials; the staging failed, not the hypothesis")
        return
    print(f"\n--- vs the deaf hen, on {n} matched trials each ---")
    for name, _ in CONDITIONS[1:]:
        other = np.array([r.struck for r in out[name]], dtype=float)
        if len(other) != n:
            print(f"  {name:<26} skipped: {len(other)} valid vs {n} baseline")
            continue
        d = other - base
        mean = float(d.mean())
        se = float(d.std(ddof=1)) / (n ** 0.5)
        t = abs(mean) / (se + 1e-12)
        crit = _t_critical(n - 1)
        flag = ("SIGNIFICANT" if t > crit
                else f"suggestive (t={t:.2f})" if t > 1.0 else f"noise (t={t:.2f})")
        print(f"  {name:<26} struck {100*mean:+6.1f} pp +/- {100*se:.1f}   {flag}")

    print("\nIf 'response only' matches the deaf hen, the call cannot save her by itself")
    print("and the channel is an interrupt. If it matches the full scaffold, it can.")


if __name__ == "__main__":
    main()
