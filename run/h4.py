"""H4: does an intact channel beat a shuffled one on a task requiring private info?

**This is the experiment the project exists to run**, and it does not need plasticity.

That is worth stating because it was assumed otherwise for twenty-one experiments.
`docs/backlog.md` §7 said phase 1 blocked everything below it. It does not block this:
H4's prediction mentions no learning at all. Call *production* is innate and passes 7/7
ethogram assays; calls have been audible since E019; and comprehension can be made
innate with the E018 scaffold. An all-innate flock is a complete, runnable H4 — and it
tests the thesis directly rather than waiting on a learning rule that has been a null
since E001.

The task is T1, shared vigilance (`docs/backlog.md` §3). A foraging hen has her head
down and cannot see a hawk, which is the information asymmetry `coop/sensing.py` was
built around. With a working channel the flock can divide labour: someone watches while
the rest feed. With a shuffled channel it hears just as much and learns nothing.

**The metric is a trade-off, not a scalar.** A flock that never forages is safe and
starving; a flock that never looks up is fed and eaten. So intake and survival are
reported together and the comparison is over the pair. Reporting intake alone would
score starvation as success.

    usage:  python -m run.h4 --minutes 20 --seeds 8
"""

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.experiment import Condition, _t_critical

# Everything is fixed. No plasticity, no exploration noise -- this measures what an
# innate flock does with a channel, and both would only add variance.
INNATE = PlasticConfig(enabled=False, explore_sigma=0.0)

# The pallium the "expanded" conditions get. The ladder's premise is that capacity is
# held constant across every condition that has a channel, so that C- can ask whether
# the neurons alone did the work.
EXPANDED = 1.5

LADDER = (
    # N -- the natural reference. Default brain, innate calls, no scaffold: a hen who
    # calls but does not respond to calls, which is where the project started.
    Condition("N  natural", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=1.0, scaffold=False),
    # C- -- expanded capacity, no channel at all. If this matches L, the neurons did it.
    Condition("C- capacity", INNATE, cfg_patch=(("channel_mode", "none"),),
              pallium_scale=EXPANDED, scaffold=True),
    # C0 -- she calls, nobody hears. Isolates the energetic cost of calling.
    Condition("C0 severed", INNATE, cfg_patch=(("channel_mode", "severed"),),
              pallium_scale=EXPANDED, scaffold=True),
    # C? -- THE CONTROL. Same bandwidth, same cost, wrong sender.
    Condition("C? shuffled", INNATE, cfg_patch=(("channel_mode", "shuffled"),),
              pallium_scale=EXPANDED, scaffold=True),
    # Cs -- she hears only herself. A channel as private memory is not communication.
    Condition("Cs self-only", INNATE, cfg_patch=(("channel_mode", "self"),),
              pallium_scale=EXPANDED, scaffold=True),
    # L -- the hypothesis.
    Condition("L  language", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True),
)

HEADLINE = ("L  language", "C? shuffled")


class H4Result(NamedTuple):
    fed_rate: float        # % of timesteps feeding -- the intake half of the trade-off
    struck: float          # predator contact-steps per hen
    exposed: float         # steps in strike range, hiding or not
    caught_rate: float     # struck / exposed -- THE risk metric. See below.
    head_down: float       # fraction of time foraging, i.e. blind to the sky
    hunger: float
    heard: float           # mean alarm-channel input, a manipulation check


def run_condition(cond: Condition, seed: int, cfg: CoopConfig,
                  seconds: float) -> H4Result:
    cfg = cond.coop(cfg)
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), cond.regions(),
                         n_hens=cfg.n_hens, auditory_scaffold=cond.scaffold)
    x = brain.initial_state(p, cfg.n_hens)
    w_end, _x, _p, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, INNATE)
    steps = cfg.n_hens * seconds / cfg.dt
    aerial = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    struck = float(jnp.sum(w_end.n_struck))
    exposed = float(jnp.sum(w_end.n_exposed))
    return H4Result(
        fed_rate=float(jnp.sum(w_end.n_fed) / steps) * 100,
        struck=struck / cfg.n_hens,
        exposed=exposed / cfg.n_hens,
        # Of the moments a hawk was actually on her, how often did she fail to hide?
        # Raw contact counts are dominated by whether the flock happened to be standing
        # where the hawk came down -- E024's smoke test showed a 17x spread between
        # conditions that was positional luck, and the conditions differ in position
        # precisely because they differ in behaviour, so the confound is not random.
        # Dividing by opportunity removes it and asks the question H4 actually poses.
        caught_rate=struck / max(exposed, 1.0),
        head_down=float(jnp.mean(s.head_down)),
        hunger=float(jnp.mean(s.hunger)),
        # Received, not emitted. The first version of this read `s.calls` and so
        # reported the severed channel as carrying a signal -- C0 hens call exactly as
        # much as anyone, and nobody hears them. Measuring production to check a
        # manipulation of reception is the same mistake E019's guard test made.
        heard=float(jnp.mean(s.audio[:, aerial])),
    )


def _paired(a, b, seeds: int):
    d = jnp.array(a) - jnp.array(b)
    mean = float(jnp.mean(d))
    if seeds < 2:
        return mean, 0.0, "single seed"
    se = float(jnp.std(d, ddof=1)) / (seeds ** 0.5)
    t = abs(mean) / (se + 1e-12)
    crit = _t_critical(seeds - 1)
    return mean, se, ("SIGNIFICANT" if t > crit
                      else f"suggestive (t={t:.2f}, need {crit:.2f})" if t > 1.0
                      else f"noise (t={t:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    ap.add_argument("--hawk-period", type=float, default=60.0,
                    help="seconds between hawk passes; the default coop's 900 gives "
                         "~1.3 passes in 20 min, so the whole H4 signal would rest on "
                         "~16 s of a 1200 s run")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens,
                                     hawk_period_s=args.hawk_period)
    seconds = args.minutes * 60.0
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    print(f"H4 -- intact channel vs shuffled, {args.minutes:.0f} min, "
          f"{len(seeds)} matched seeds ({seeds[0]}-{seeds[-1]}), {cfg.n_hens} hens")
    print(f"NO PLASTICITY in any condition. Hawk every {args.hawk_period:.0f} s.\n")
    print("intake and risk are a trade-off, not a score: a flock that never forages")
    print("is safe and starving. Read the two columns together.\n")

    hdr = (f"{'condition':<14}{'fed %':>8}{'caught rate':>13}{'exposed':>10}"
           f"{'head down':>11}{'hunger':>8}{'alarm heard':>13}")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.perf_counter()
    table = {}
    for cond in LADDER:
        rs = [run_condition(cond, s, cfg, seconds) for s in seeds]
        table[cond.name] = rs
        m = lambda f: float(jnp.mean(jnp.array([f(r) for r in rs])))
        print(f"{cond.name:<14}{m(lambda r: r.fed_rate):>8.2f}"
              f"{m(lambda r: r.caught_rate):>13.3f}{m(lambda r: r.exposed):>10.0f}"
              f"{m(lambda r: r.head_down):>11.3f}"
              f"{m(lambda r: r.hunger):>8.3f}{m(lambda r: r.heard):>13.4f}")

    n = len(seeds)
    col = lambda name, g: [g(r) for r in table[name]]
    L, Cq = HEADLINE

    print(f"\n--- HEADLINE: {L} vs {Cq} ---")
    print("identical brain, identical bandwidth, identical calling cost. The only")
    print("difference is whether what she hears is about her own surroundings.\n")
    for label, get, good in (("fed %", lambda r: r.fed_rate, "higher"),
                             ("caught rate", lambda r: r.caught_rate, "lower"),
                             ("struck/hen", lambda r: r.struck, "lower"),
                             ("hunger", lambda r: r.hunger, "lower")):
        mean, se, verdict = _paired(col(L, get), col(Cq, get), n)
        print(f"  {label:<12} {mean:+8.3f} +/- {se:.3f} SE   {verdict}"
              f"   ({good} is better for L)")

    print(f"\n--- The falsifiers H4 named in advance ---")
    for label, treat, base, meaning in (
            ("L vs C- (capacity)", L, "C- capacity",
             "if ~0, the extra neurons did the work, not the channel"),
            ("L vs Cs (self-only)", L, "Cs self-only",
             "if ~0, the channel is private memory, not communication"),
            ("L vs C0 (severed)", L, "C0 severed",
             "if ~0, hearing nothing costs nothing -- no information was flowing"),
            ("L vs N (natural)", L, "N  natural",
             "does the language flock beat a naturalistic bird, or a lobotomised one?")):
        mean, se, verdict = _paired(col(treat, lambda r: r.fed_rate),
                                    col(base, lambda r: r.fed_rate), n)
        print(f"  {label:<22} fed % {mean:+7.3f} +/- {se:.3f} SE  {verdict}")
        print(f"  {'':<22} {meaning}")

    print(f"\nmanipulation check: 'alarm heard' must be ~0 for C- and C0, and "
          f"non-zero elsewhere.")
    if n < 4:
        print(f"WARNING: {n} seeds is not a usable sample whatever the t says.")
    print(f"wall clock: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
