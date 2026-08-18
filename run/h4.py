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
import json
import os
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
    # C- pays nothing for calling either -- otherwise it is byte-identical to C0,
    # which is what E024 shipped (all 8 seeds identical on every field, E026). The
    # difference between them IS the energetic cost of calling, and it only exists if
    # one of them is charged and the other is not.
    Condition("C- capacity", INNATE, cfg_patch=(("channel_mode", "none"),
                                                ("call_vigour_drain", 0.0)),
              pallium_scale=EXPANDED, scaffold=True),
    # C0 -- she calls, nobody hears. Isolates the energetic cost of calling.
    Condition("C0 severed", INNATE, cfg_patch=(("channel_mode", "severed"),),
              pallium_scale=EXPANDED, scaffold=True),
    # C? -- THE CONTROL. Same bandwidth, same cost, wrong sender.
    # The yoked control. `call_log_steps` is set here rather than globally because the
    # buffer costs throughput and no other condition needs it (E026).
    Condition("C? yoked", INNATE,
              cfg_patch=(("channel_mode", "yoked"),
                         ("call_log_steps", spec.YOKE_LOG_STEPS)),
              pallium_scale=EXPANDED, scaffold=True),
    # Cs -- she hears only herself. A channel as private memory is not communication.
    Condition("Cs self-only", INNATE, cfg_patch=(("channel_mode", "self"),),
              pallium_scale=EXPANDED, scaffold=True),
    # L -- the hypothesis.
    Condition("L  language", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True),
    # Lx -- L with the pallium's route to the muscles cut. Added by E028 as a permanent
    # rung. If this matches L, whatever the ladder is measuring does not run through the
    # brain, and the result is about the reflex arc and the world. E027 found exactly
    # that after the fact; a standing rung means the next such result announces itself.
    Condition("Lx lesioned", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True, lesion_readout=True),
)

# The registered headline, per `docs/backlog.md` section 1: L vs C?, not L vs anything
# else. E026 reported everything against deaf and never computed this contrast, though
# its own pairing made it a subtraction away.
HEADLINE = ("L  language", "C? yoked")


class H4Result(NamedTuple):
    fed_rate: float        # % of timesteps feeding -- the intake half of the trade-off
    struck: float          # predator contact-steps per hen
    exposed: float         # steps in strike range, hiding or not
    caught_rate: float     # struck / exposed -- CONFOUNDED, see below. Kept for E024.
    at_risk: float         # (hen, dive) pairs where she began inside the radius
    caught: float          # of those, how many she was struck in
    caught_per_event: float   # caught / at_risk -- CONFOUNDED too, see E027
    head_down: float       # fraction of time foraging, i.e. blind to the sky
    hunger: float
    heard: float           # mean alarm-channel input, a manipulation check
    # --- E028. Denominators are reported as raw counts, not folded into ratios, so
    # that a denominator moving with the treatment is visible in the table instead of
    # having to be inferred two experiments later.
    dives: float           # (hen, dive) pairs, full stop. Treatment cannot touch this.
    blind_risk: float      # of those, she began the dive in the radius AND blind
    blind_caught: float    # of those, she was caught
    caught_any: float      # (hen, dive) pairs in which she was caught at all
    caught_itt: float      # caught / dives -- THE metric. Intent to treat.


def run_condition(cond: Condition, seed: int, cfg: CoopConfig,
                  seconds: float) -> H4Result:
    cfg = cond.coop(cfg)
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), cond.regions(),
                         n_hens=cfg.n_hens, auditory_scaffold=cond.scaffold,
                         scaffold_gain=cond.scaffold_gain)
    if cond.lesion_readout:
        p = p._replace(W_out=jnp.zeros_like(p.W_out))
    x = brain.initial_state(p, cfg.n_hens)
    w_end, _x, _p, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, cond.pc)
    steps = cfg.n_hens * seconds / cfg.dt
    aerial = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    struck = float(jnp.sum(w_end.n_struck))
    exposed = float(jnp.sum(w_end.n_exposed))
    at_risk = float(jnp.sum(w_end.n_at_risk))
    caught = float(jnp.sum(w_end.n_caught))
    dives = float(jnp.sum(w_end.n_dives))
    caught_any = float(jnp.sum(w_end.n_caught_any))
    blind_risk = float(jnp.sum(w_end.n_blind_risk))
    blind_caught = float(jnp.sum(w_end.n_blind_caught))
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
        # CONFOUNDED. Introduced mid-E024 to fix a positional confound in raw strikes,
        # and it imported a behavioural one: crouching zeroes locomotion, so a
        # partially-crouching hen lingers in the radius and the denominator moves with
        # the treatment. Exposure varied 15x across E026's ablation conditions, and this
        # ratio disagreed with raw strikes about the sign. Retained only so E024's
        # numbers remain reproducible; do not compare conditions on it.
        caught_rate=struck / max(exposed, 1.0),
        at_risk=at_risk,
        caught=caught,
        # THE metric. Denominator is fixed at the instant the hawk commits, before any
        # response can alter it, so the treatment cannot move it.
        # CONFOUNDED, and it took E027 to see it. The claim attached to this line in
        # four separate files was "the denominator is fixed the instant the hawk
        # commits, so the treatment cannot move it". The denominator *within* a dive is
        # fixed; the number of dives that find a hen at risk is not, because crouching
        # zeroes locomotion and a crouching hen is still there when the next hawk comes.
        # Measured spread across conditions: up to 63%. Kept so E026 stays reproducible.
        caught_per_event=caught / max(at_risk, 1.0),
        head_down=float(jnp.mean(s.head_down)),
        hunger=float(jnp.mean(s.hunger)),
        # Received, not emitted. The first version of this read `s.calls` and so
        # reported the severed channel as carrying a signal -- C0 hens call exactly as
        # much as anyone, and nobody hears them. Measuring production to check a
        # manipulation of reception is the same mistake E019's guard test made.
        heard=float(jnp.mean(s.audio[:, aerial])),
        dives=dives,
        blind_risk=blind_risk,
        blind_caught=blind_caught,
        caught_any=caught_any,
        # Intent to treat: every hen, every dive, whether or not she was near it or
        # could see it. Smaller than the conditional rate and harder to move, which is
        # the point -- it is the only denominator no behaviour can reach.
        caught_itt=caught_any / max(dives, 1.0),
    )


# --- Checkpointing -----------------------------------------------------------
#
# A full ladder is ~10 minutes per condition, and this environment reclaims the
# container between turns -- two runs were killed mid-ladder before this existed, one
# of them after five of six conditions. Every (condition, seed) cell is therefore
# written to disk the moment it completes, and a restart skips what is already there.
# Nothing about the experiment changes; the runs are deterministic, so a resumed
# ladder is bit-identical to one that ran straight through.

def _cache_load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _cache_save(path: str, cache: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)          # atomic: a kill mid-write cannot corrupt the cache


def _key(cond_name: str, seed: int, minutes: float, hens: int, hawk: float,
         cfg=None) -> str:
    """Cell identity, including a hash of the whole world config.

    Without the config hash a cell computed under one world is served as current under
    another. E024's 48 cells were cached before E025 added food depletion, and rerunning
    the same command afterwards reprinted them unchanged (E026). The world is part of
    the measurement; it belongs in the key.
    """
    import hashlib
    tag = ""
    if cfg is not None:
        tag = hashlib.sha1(repr(tuple(cfg)).encode()).hexdigest()[:10]
    return f"{cond_name}|{seed}|{minutes}|{hens}|{hawk}|{tag}"


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
    ap.add_argument("--cache", default="scratchpad/e024_cache.json",
                    help="per-cell results, so a killed run resumes instead of restarting")
    ap.add_argument("--budget", type=float, default=480.0,
                    help="stop starting new cells after this many seconds and exit "
                         "cleanly, so a turn-bounded environment always makes progress")
    ap.add_argument("--hawk-period", type=float, default=60.0,
                    help="seconds between hawk passes; the default coop's 900 gives "
                         "~1.3 passes in 20 min, so the whole H4 signal would rest on "
                         "~16 s of a 1200 s run")
    ap.add_argument("--food-deplete-rate", type=float, default=spec.DEFAULT_COOP.food_deplete_rate,
                    help="E037/E038 found this confounds H2's harness at 16 hens/20 min; "
                         "pass 0.0 to check whether L vs C? is exposed too. The cache key "
                         "already hashes the whole cfg, so this needs no cache changes.")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens,
                                     hawk_period_s=args.hawk_period,
                                     food_deplete_rate=args.food_deplete_rate)
    seconds = args.minutes * 60.0
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    print(f"H4 -- intact channel vs shuffled, {args.minutes:.0f} min, "
          f"{len(seeds)} matched seeds ({seeds[0]}-{seeds[-1]}), {cfg.n_hens} hens")
    print(f"NO PLASTICITY in any condition. Hawk every {args.hawk_period:.0f} s.\n")
    print("intake and risk are a trade-off, not a score: a flock that never forages")
    print("is safe and starving. Read the two columns together.\n")

    hdr = (f"{'condition':<14}{'fed %':>8}{'caught/dive':>13}{'dives':>8}"
           f"{'at risk':>9}{'blind risk':>12}{'caught/event':>14}"
           f"{'head down':>11}{'alarm heard':>13}")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.perf_counter()
    cache = _cache_load(args.cache)
    table = {}
    for cond in LADDER:
        rs = []
        for sd in seeds:
            k = _key(cond.name, sd, args.minutes, cfg.n_hens, args.hawk_period,
                     cond.coop(cfg))
            if k in cache:
                rs.append(H4Result(*cache[k]))
                continue
            if time.perf_counter() - t0 > args.budget:
                continue
            r = run_condition(cond, sd, cfg, seconds)
            cache[k] = list(r)
            _cache_save(args.cache, cache)
            rs.append(r)
        if len(rs) < len(seeds):
            continue
        table[cond.name] = rs
        m = lambda f: float(jnp.mean(jnp.array([f(r) for r in rs])))
        print(f"{cond.name:<14}{m(lambda r: r.fed_rate):>8.2f}"
              f"{m(lambda r: r.caught_itt):>13.3f}{m(lambda r: r.dives):>8.0f}"
              f"{m(lambda r: r.at_risk):>9.1f}{m(lambda r: r.blind_risk):>12.1f}"
              f"{m(lambda r: r.caught_per_event):>14.3f}"
              f"{m(lambda r: r.head_down):>11.3f}{m(lambda r: r.heard):>13.4f}")

    total = len(LADDER) * len(seeds)
    done = sum(1 for c in LADDER for sd in seeds
               if _key(c.name, sd, args.minutes, cfg.n_hens, args.hawk_period,
                       c.coop(cfg)) in cache)
    if done < total:
        print(f"\nINCOMPLETE: {done}/{total} cells cached, {total - done} to go.")
        print(f"Re-run the same command to continue; finished cells are not repeated.")
        print(f"wall clock this pass: {time.perf_counter() - t0:.0f} s")
        return

    n = len(seeds)
    col = lambda name, g: [g(r) for r in table[name]]
    L, Cq = HEADLINE

    # Denominators first, because a denominator that moves with the treatment is what
    # invalidated the previous two metrics and it was visible in the printed table both
    # times. `dives` is fixed by construction; anything else drifting is a warning.
    print("\n--- denominator check: can the treatment move what it is divided by? ---")
    base_d = float(jnp.mean(jnp.array(col(Cq, lambda r: r.dives))))
    base_r = float(jnp.mean(jnp.array(col(Cq, lambda r: r.blind_risk))))
    print(f"  {'condition':<14}{'dives':>8}{'vs C?':>9}{'blind risk':>12}{'vs C?':>9}")
    for name in table:
        d = float(jnp.mean(jnp.array(col(name, lambda r: r.dives))))
        b = float(jnp.mean(jnp.array(col(name, lambda r: r.blind_risk))))
        print(f"  {name:<14}{d:>8.0f}{100*(d-base_d)/max(base_d,1):>8.1f}%"
              f"{b:>12.1f}{100*(b-base_r)/max(base_r,1):>8.1f}%")
    print("  dives must be flat. blind risk drifting by tens of percent is E027's")
    print("  finding, and it is why caught/dive is the headline and caught/event is not.")

    print(f"\n--- HEADLINE: {L} vs {Cq} ---")
    print("The registered contrast (backlog section 1), not L vs deaf. Identical brain,")
    print("identical bandwidth, identical calling cost. The only difference is whether")
    print("what she hears is about her own surroundings.\n")
    for label, get, good in (("caught/dive", lambda r: r.caught_itt, "lower"),
                             ("fed %", lambda r: r.fed_rate, "higher"),
                             ("caught/event*", lambda r: r.caught_per_event, "lower"),
                             ("caught rate*", lambda r: r.caught_rate, "lower"),
                             ("struck/hen", lambda r: r.struck, "lower"),
                             ("hunger", lambda r: r.hunger, "lower")):
        mean, se, verdict = _paired(col(L, get), col(Cq, get), n)
        print(f"  {label:<14}{mean:+8.3f} +/- {se:.3f} SE   {verdict}"
              f"   ({good} is better for L)")
    print("  * confounded denominators, see the check above. Reported, not relied on.")

    # Pooled counts alongside the paired mean. E026 quoted the per-seed mean-of-ratios
    # (-0.198) in prose while the pooled rate over the same events was -0.150; both are
    # legitimate estimators and the write-up should not silently pick the larger.
    pool = lambda name, num, den: (sum(num(r) for r in table[name])
                                   / max(sum(den(r) for r in table[name]), 1.0))
    print("\n  pooled over events, for comparison with the paired means above:")
    for label, num, den in (
            ("caught/dive", lambda r: r.caught_any, lambda r: r.dives),
            ("caught/event", lambda r: r.caught, lambda r: r.at_risk),
            ("caught|blind", lambda r: r.blind_caught, lambda r: r.blind_risk)):
        print(f"    {label:<14}{L} {pool(L, num, den):.3f}   "
              f"{Cq} {pool(Cq, num, den):.3f}   "
              f"diff {pool(L, num, den) - pool(Cq, num, den):+.3f}")

    # T1's own registered metric. `docs/backlog.md` section 96 specifies "food intake at
    # matched predation risk"; E026 reported a risk metric instead and never recorded
    # that this one was null. Printed unconditionally so it cannot be quietly skipped.
    print("\n--- T1's registered metric: food intake at matched risk ---")
    mean, se, verdict = _paired(col(L, lambda r: r.fed_rate),
                                col(Cq, lambda r: r.fed_rate), n)
    print(f"  fed % {mean:+.3f} +/- {se:.3f} SE   {verdict}")
    print("  backlog section 3 predicted L forages MORE than C? at equal risk.")
    print("  In E026 this was 3.06 vs 3.07 and then 2.54 vs 2.41 -- null, and worse in")
    print("  the replication. Reported here whatever it says.")

    print(f"\n--- The falsifiers H4 named in advance ---")
    for label, treat, base, meaning in (
            ("L vs C- (capacity)", L, "C- capacity",
             "if ~0, the extra neurons did the work, not the channel"),
            ("L vs Cs (self-only)", L, "Cs self-only",
             "if ~0, the channel is private memory, not communication"),
            ("L vs C0 (severed)", L, "C0 severed",
             "if ~0, hearing nothing costs nothing -- no information was flowing"),
            ("L vs N (natural)", L, "N  natural",
             "does the language flock beat a naturalistic bird, or a lobotomised one?"),
            ("L vs Lx (lesioned)", L, "Lx lesioned",
             "if ~0, the pallium is not in the causal path and this is a result "
             "about the reflex arc")):
        mean, se, verdict = _paired(col(treat, lambda r: r.caught_itt),
                                    col(base, lambda r: r.caught_itt), n)
        print(f"  {label:<22} caught/dive {mean:+7.3f} +/- {se:.3f} SE  {verdict}")
        print(f"  {'':<22} {meaning}")

    print("\n  NOTE on C-: with plasticity off in every condition, the capacity control")
    print("  is vacuous by construction -- an untrained pallium is a random projection")
    print("  that cannot use extra neurons. H0's 'at any capacity' clause is untested")
    print("  until a learning rule works. Kept in the ladder so it is not re-dropped.")

    print(f"\nmanipulation check: 'alarm heard' must be ~0 for C- and C0, and "
          f"non-zero elsewhere.")
    if n < 4:
        print(f"WARNING: {n} seeds is not a usable sample whatever the t says.")
    print(f"wall clock: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
