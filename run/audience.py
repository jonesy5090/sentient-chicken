"""H3: does the audience effect emerge without being programmed?

Real cockerels alarm- and food-call far more readily when a hen is watching, and the
effect is graded by audience type -- a live conspecific beats a bobwhite quail beats
an empty cage. Evans & Marler showed a *video* of a hen is enough.

Chickens are vocal non-learners, so call production is innate and is hardwired in
`hen/innate.py`. Usage is learned and is deliberately not. That leaves the audience
effect sitting in the model as an unclaimed prediction: if it emerges from rearing,
the learning rule is capturing something about chicken development that nobody put
there.

The assay is the Evans & Marler design. Present an identical stimulus to an identical
hen, varying only whether flockmates are within earshot, and measure calling. Applied
to the same flock before and after rearing, with a non-plastic control.

    usage:  python -m run.audience --minutes 30 --seeds 4
"""

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

ABSENT = 1e4
AUDIENCE_DISTANCE = 2.0     # well inside hearing range, outside huddling range

# A hawk directly overhead drives the innate alarm reflex to 0.97 -- saturated, with
# no headroom for an audience effect to appear in. The first run of this assay found
# exactly that: the alarm channel could not move because it was already pinned.
#
# So the hawk is staged at a distance, where the aerial channel is ~0.3 and the call
# sits mid-range. That is also the ecologically interesting case: a hen has a real
# decision to make about a distant threat, and none at all about one on top of her.
HAWK_DISTANCE = 7.0


class AudienceResult(NamedTuple):
    alarm_alone: float
    alarm_audience: float
    food_alone: float
    food_audience: float

    @property
    def alarm_effect(self) -> float:
        return self.alarm_audience - self.alarm_alone

    @property
    def food_effect(self) -> float:
        return self.food_audience - self.food_alone


def _staged(cfg: CoopConfig, n_hens: int, *, audience: bool,
            hawk: bool, food: bool):
    """One focal hen at the centre; flockmates present or absent.

    Absent flockmates are parked far outside vision and hearing rather than removed,
    so both conditions run the same number of hens through the same compiled program.
    Otherwise the two conditions would differ in flock size as well as in audience,
    which is exactly the confound the assay exists to avoid.
    """
    w = world.reset(jax.random.key(0), cfg._replace(n_hens=n_hens))

    pos = jnp.full((n_hens, 2), ABSENT).at[0].set(jnp.array([10.0, 10.0]))
    if audience:
        for i in range(1, n_hens):
            angle = 2 * jnp.pi * i / max(n_hens - 1, 1)
            pos = pos.at[i].set(jnp.array([
                10.0 + AUDIENCE_DISTANCE * jnp.cos(angle),
                10.0 + AUDIENCE_DISTANCE * jnp.sin(angle)]))

    food_pos = jnp.full((cfg.n_food, 2), ABSENT)
    if food:
        food_pos = food_pos.at[0].set(jnp.array([10.3, 10.0]))

    return w._replace(
        pos=pos,
        heading=jnp.zeros((n_hens,)),
        food_pos=food_pos,
        water_pos=jnp.full((cfg.n_water, 2), ABSENT),
        hawk_pos=(jnp.array([10.0 + HAWK_DISTANCE, 10.0]) if hawk
                  else jnp.array([ABSENT, ABSENT])),
        hawk_on=jnp.array(1.0 if hawk else 0.0),
        hawk_t=jnp.array(1e4 if hawk else 0.0),
        fox_pos=jnp.array([ABSENT, ABSENT]),
        fox_on=jnp.array(0.0),
        hunger=jnp.full((n_hens,), 0.4),
        thirst=jnp.full((n_hens,), 0.2),
        cold=jnp.full((n_hens,), 0.2),
    )


def assay(p, cfg: CoopConfig, n_hens: int, steps: int = 300) -> AudienceResult:
    """Calling by the focal hen, with and without an audience.

    Run with plasticity off: this measures what she has already learned, and must not
    let her learn during the measurement.
    """
    def call_rate(audience: bool, hawk: bool, food: bool, channel: int) -> float:
        w = _staged(cfg, n_hens, audience=audience, hawk=hawk, food=food)
        x = brain.initial_state(p, n_hens)
        *_, trace = simulate.rollout(
            w, x, p, jax.random.key(11), cfg._replace(n_hens=n_hens), steps)
        return float(jnp.mean(trace.motor[:, 0, channel]))

    return AudienceResult(
        alarm_alone=call_rate(False, True, False, spec.M_CALL_AERIAL),
        alarm_audience=call_rate(True, True, False, spec.M_CALL_AERIAL),
        food_alone=call_rate(False, False, True, spec.M_CALL_FOOD),
        food_audience=call_rate(True, False, True, spec.M_CALL_FOOD),
    )


def comprehension(p, cfg: CoopConfig, n_hens: int, steps: int = 200,
                  pred_gain: float = 0.0) -> float:
    """Playback: does hearing an aerial alarm call make her crouch?

    This is Evans & Marler's playback design — present the call with no predator to
    see and measure the response — and it measures the thing the audience effect
    structurally depends on.

    The chain that has to close for audience-sensitive calling to be learnable is:
    she calls, a flockmate *hears and responds*, the flockmate avoids a strike, and
    the benefit returns to her through the kin term. If flockmates do not respond to
    calls, the chain is broken at step two and no amount of fixing the reward can
    help. Comprehension is learned, not innate — `hen/innate.py` deliberately wires
    no reflex from the auditory channels — so at hatch this should be zero.

    Driven by a hand-built observation rather than a staged world, so the call is
    the only thing that varies. Held static for `steps` to let the recurrent
    dynamics settle.
    """
    def crouch_under(call_level: float, params=p) -> float:
        obs = jnp.zeros((n_hens, spec.OBS_DIM))
        obs = obs.at[:, spec.AUDIO_LO + 2].set(call_level)   # aerial alarm channel
        x = brain.initial_state(params, n_hens)
        total = 0.0
        for _ in range(steps):
            x, motor, _ = brain.step(x, obs, params, cfg.dt,
                                     pred_gain=pred_gain)
            total += float(jnp.mean(motor[:, spec.M_CROUCH]))
        return total / steps

    return crouch_under(1.0) - crouch_under(0.0)


def rear_and_assay(seed: int, cfg: CoopConfig, seconds: float, pc: PlasticConfig):
    """Assay a flock at hatch, rear it, assay it again."""
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)

    before = assay(p, cfg, cfg.n_hens)
    comp_before = comprehension(p, cfg, cfg.n_hens,
                                pred_gain=pc.pred_gain if pc.pred_enabled else 0.0)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, pc)
    after = assay(p_end, cfg, cfg.n_hens)
    comp_after = comprehension(p_end, cfg, cfg.n_hens,
                               pred_gain=pc.pred_gain if pc.pred_enabled else 0.0)
    return before, after, comp_before, comp_after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    ap.add_argument("--no-growth", action="store_true", default=True,
                    help="growth is off by default: it is the weaker condition (H2a)")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    seconds = args.minutes * 60.0
    seeds = list(range(args.seeds))

    conditions = [
        ("audible kin", PlasticConfig(enabled=True, growth_enabled=False,
                                      kin_audible=True)),
        ("flat kin (E005)", PlasticConfig(enabled=True, growth_enabled=False,
                                          kin_audible=False)),
        ("fixed (control)", PlasticConfig(enabled=False)),
    ]

    print(f"H3 -- audience effect after {args.minutes:.0f} min of rearing, "
          f"{len(seeds)} seeds, {cfg.n_hens} hens\n")
    print("audience effect = calling with flockmates in earshot minus calling alone")
    print("a real cockerel shows a large positive effect; a hatchling should show ~0\n")

    hdr = (f"{'condition':<18} {'when':<8} {'alarm alone':>12} {'alarm aud.':>11} "
           f"{'effect':>8} {'food alone':>11} {'food aud.':>10} {'effect':>8}")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.perf_counter()
    results = {}
    for name, pc in conditions:
        pairs = [rear_and_assay(s, cfg, seconds, pc) for s in seeds]
        for label, idx in (("hatch", 0), ("reared", 1)):
            rs = [pr[idx] for pr in pairs]
            m = lambda f: float(jnp.mean(jnp.array([f(r) for r in rs])))
            print(f"{name:<18} {label:<8} "
                  f"{m(lambda r: r.alarm_alone):>12.3f} "
                  f"{m(lambda r: r.alarm_audience):>11.3f} "
                  f"{m(lambda r: r.alarm_effect):>+8.3f} "
                  f"{m(lambda r: r.food_alone):>11.3f} "
                  f"{m(lambda r: r.food_audience):>10.3f} "
                  f"{m(lambda r: r.food_effect):>+8.3f}")
        results[name] = pairs

    print()
    print("comprehension -- does hearing an alarm call make her crouch, with no")
    print("predator to see? the audience effect cannot emerge without this, since")
    print("a call that nobody responds to can never repay its cost.\n")
    print(f"{'condition':<18} {'hatch':>10} {'reared':>10} {'change':>10}")
    print("-" * 50)
    for name, pairs in results.items():
        cb = float(jnp.mean(jnp.array([p[2] for p in pairs])))
        ca = float(jnp.mean(jnp.array([p[3] for p in pairs])))
        print(f"{name:<18} {cb:>10.4f} {ca:>10.4f} {ca - cb:>+10.4f}")

    print()
    from run.experiment import _t_critical
    n = len(seeds)
    for name, pairs in results.items():
        for label, get in (("alarm", lambda r: r.alarm_effect),
                           ("food", lambda r: r.food_effect)):
            delta = jnp.array([get(a) - get(b) for b, a, _cb, _ca in pairs])
            mean = float(jnp.mean(delta))
            if n < 2:
                print(f"{name:<18} {label} effect change {mean:+.3f} (single seed)")
                continue
            se = float(jnp.std(delta, ddof=1)) / (n ** 0.5)
            t = abs(mean) / (se + 1e-12)
            crit = _t_critical(n - 1)
            flag = ("SIGNIFICANT" if t > crit
                    else f"suggestive (t={t:.2f}, need {crit:.2f})" if t > 1.0
                    else f"noise (t={t:.2f})")
            print(f"{name:<18} {label:<5} effect change over rearing "
                  f"{mean:+.3f} +/- {se:.3f} SE   {flag}")

    print("\npositive = the hen learned to call more when someone is listening")
    print("the fixed control must show ~0; if it moves, the assay is measuring "
          "something other than learning")
    if n < 4:
        print(f"WARNING: {n} seeds is not a usable sample whatever the t value says. "
              f"Two seeds that happen to agree give a tiny SE and a huge t.")
    print(f"wall clock: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
