"""Neonatal ethogram assays: is this thing behaving like a newly hatched chick?

Each probe stages a specific situation and checks a documented chick behaviour. They
are the phase 0 acceptance criteria, and they are all *innate* behaviours -- nothing
here requires learning, because nothing here has learned anything yet.

The important one is `head_down_blindness`. It is a paired assay: identical hawk,
identical hen, differing only in whether she happens to be pecking at the moment it
arrives. If the foraging hen crouches anyway, the vigilance/foraging trade-off is not
actually wired up, and the entire premise for language emerging in phase 4 is absent
from the simulation regardless of what the neurons do.

    usage:  python -m run.probes
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, regions
from run import simulate

ABSENT = 1e4        # park unwanted entities far outside the vision range
GENOME_SEED = 1     # every probe uses the same genome, only the flock size varies

_CONNECTOME_CACHE: dict = {}

# Adoption gate for E072's balanced E/I (E074). Module-level rather than a per-probe
# argument because the whole point is to re-run the *entire* ethogram under a different
# connectome without touching a single probe's own staging. Set by `--balanced-ei`, or
# directly by a diagnostic script.
BALANCED_EI = False


class Probe(NamedTuple):
    name: str
    passed: bool
    detail: str


def _connectome(n_hens: int, gakel_scaffold: bool = False):
    """Same genome for every probe; width follows the staged flock size."""
    ck = (n_hens, gakel_scaffold, BALANCED_EI)
    if ck not in _CONNECTOME_CACHE:
        _CONNECTOME_CACHE[ck] = connectome.build(
            jax.random.key(GENOME_SEED), regions.DEFAULT_REGIONS, n_hens=n_hens,
            gakel_scaffold=gakel_scaffold, balanced_ei=BALANCED_EI)
    return _CONNECTOME_CACHE[ck]


def _staged(cfg: CoopConfig, *, pos, heading, food=None, water=None,
            hawk=None, fox=None, hunger=0.5, thirst=0.2, cold=0.2):
    """A coop with everything placed by hand and nothing left to chance."""
    w = world.reset(jax.random.key(0), cfg)

    def place(spots, n):
        arr = jnp.full((n, 2), ABSENT)
        if spots is not None:
            arr = arr.at[: len(spots)].set(jnp.asarray(spots, dtype=jnp.float32))
        return arr

    return w._replace(
        pos=jnp.asarray(pos, dtype=jnp.float32).reshape(cfg.n_hens, 2),
        heading=jnp.full((cfg.n_hens,), float(heading)),
        food_pos=place(food, cfg.n_food),
        water_pos=place(water, cfg.n_water),
        hawk_pos=jnp.asarray(hawk if hawk is not None else (ABSENT, ABSENT),
                             dtype=jnp.float32),
        hawk_on=jnp.array(1.0 if hawk is not None else 0.0),
        hawk_t=jnp.array(1e4 if hawk is not None else 0.0),
        fox_pos=jnp.asarray(fox if fox is not None else (ABSENT, ABSENT),
                            dtype=jnp.float32),
        fox_on=jnp.array(1.0 if fox is not None else 0.0),
        fox_t=jnp.array(1e4 if fox is not None else 0.0),
        hunger=jnp.full((cfg.n_hens,), hunger),
        thirst=jnp.full((cfg.n_hens,), thirst),
        cold=jnp.full((cfg.n_hens,), cold),
    )


def _run(cfg: CoopConfig, w, steps: int = 120, gakel_scaffold: bool = False):
    p = _connectome(cfg.n_hens, gakel_scaffold)
    x = brain.initial_state(p, cfg.n_hens)
    w_end, _x, _p, _ps, _k, trace = simulate.rollout(
        w, x, p, jax.random.key(7), cfg, steps)
    return w_end, trace


def _peak(trace, channel, hen=0):
    return float(jnp.max(trace.motor[:, hen, channel]))


def _mean(trace, channel, hen=0):
    return float(jnp.mean(trace.motor[:, hen, channel]))


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

# --- How a modulation assay must be written (E090) --------------------------
#
# An assay of the form "X changes Y" must test the SIZE of the change, not its sign.
#
# E089 is why. The gakel scaffold suppresses pecking by 3.5% at full amplitude, because
# food drives `M_PECK` at +7.0 against a scaffold weight of 1.5 and both sit deep in
# sigmoid saturation. That is behaviourally inert -- E089 planted a correct association,
# validated every other link in the chain, and measured occupancy move by +0.3% against
# an instrument resolving 5.1%. And the 3.5% was printed by every ethogram run for seven
# experiments, under an assay whose own docstring warned that "a scaffold that damped
# everything on the audio bus would pass a bare sign test while being useless" -- and
# which then applied a bare sign test.
#
# `MIN_MODULATION` is a judgement, and stated as one. Its grounds: 3.5% demonstrably
# produces nothing, so the bar must sit far above it; and every other modulation assay
# in this file clears 25% comfortably (contact call 72% relative; sick-flockmate 0.63 of
# separation), so it is not a threshold invented to fail one thing.
MIN_MODULATION = 0.25          # relative change required of a modulated motor channel
MIN_BIAS = 0.05                # absolute bias required where the read-out is a bias
# Hunger levels a conditional response is tested at (E090/E091). The free-running flock
# lives at p10-p90 = 0.285-0.433, so `SATED` sits at its lower edge and `STARVING` well
# outside it -- the response must be strong where the hens actually are, and weaker where
# need is extreme.
SATED, STARVING = 0.2, 0.8


def _modulation(base: float, modulated: float) -> float:
    """Relative reduction of `modulated` against `base`; negative if it went up."""
    return (base - modulated) / max(abs(base), 1e-9)


def peck_at_food(cfg: CoopConfig) -> Probe:
    """Chicks peck at small objects from hatching, hungry or not."""
    cfg = cfg._replace(n_hens=1)
    w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0,
                food=[[10.25, 10.0]], hunger=0.1)
    _, tr = _run(cfg, w)
    v = _peak(tr, spec.M_PECK)
    return Probe("peck at food", v > 0.5, f"peck={v:.2f} (want >0.5)")


def crouch_at_hawk(cfg: CoopConfig) -> Probe:
    """Aerial alarm response: crouch and freeze, head up."""
    cfg = cfg._replace(n_hens=1)
    w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, hawk=(10.0, 10.0))
    _, tr = _run(cfg, w)
    v = _peak(tr, spec.M_CROUCH)
    return Probe("crouch at hawk", v > 0.5, f"crouch={v:.2f} (want >0.5)")


def head_down_blindness(cfg: CoopConfig) -> Probe:
    """The vigilance/foraging trade-off, tested as a within-run contrast.

    One hen, one hawk parked overhead, and food to peck at. As she alternates between
    pecking and walking she supplies both conditions herself, so the comparison is
    made against the same bird moments apart rather than across two staged worlds.

    We read the aerial channel out of the observation directly, which tests the gate
    rather than inferring it from behaviour. Posture at step t is set by the motor
    vector at t-1, hence the offset.
    """
    cfg = cfg._replace(n_hens=1)
    w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0,
                food=[[10.3, 10.0]], hawk=(10.0, 10.0), hunger=0.6)
    _, tr = _run(cfg, w, steps=400)

    pecked = jnp.asarray(tr.motor[:-1, 0, spec.M_PECK]) > 0.5
    aerial = jnp.asarray(tr.obs[1:, 0, spec.IDX_AERIAL])
    crouch = jnp.asarray(tr.motor[1:, 0, spec.M_CROUCH])

    n_down, n_up = int(jnp.sum(pecked)), int(jnp.sum(~pecked))
    if n_down < 10 or n_up < 10:
        return Probe("head-down blindness", False,
                     f"inconclusive: {n_down} head-down / {n_up} head-up steps")

    def avg(v, m):
        return float(jnp.sum(v * m) / jnp.sum(m))

    air_down, air_up = avg(aerial, pecked), avg(aerial, ~pecked)
    cr_down, cr_up = avg(crouch, pecked), avg(crouch, ~pecked)

    ok = air_down < 0.05 and air_up > 0.5 and cr_down < 0.5 < cr_up
    return Probe(
        "head-down blindness", ok,
        f"aerial seen {air_down:.2f} pecking vs {air_up:.2f} head-up; "
        f"crouch {cr_down:.2f} vs {cr_up:.2f} ({n_down}/{n_up} steps)")


def flee_from_fox(cfg: CoopConfig) -> Probe:
    """Ground predator: run, do not freeze. The opposite of the hawk response."""
    cfg = cfg._replace(n_hens=1)
    w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, fox=(12.0, 10.0))
    _, tr = _run(cfg, w)
    flee = _peak(tr, spec.M_FLEE)
    crouch = _peak(tr, spec.M_CROUCH)
    return Probe("flee from fox", flee > 0.5 and crouch < 0.5,
                 f"flee={flee:.2f} crouch={crouch:.2f}")


def referential_alarm(cfg: CoopConfig) -> Probe:
    """Aerial and terrestrial threats must drive *different* calls.

    This is the functional reference Evans & Marler documented in fowl, and it is the
    natural baseline any phase 4 language work has to beat.
    """
    cfg = cfg._replace(n_hens=1)
    w_air = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, hawk=(10.0, 10.0))
    _, tr_air = _run(cfg, w_air)
    w_gnd = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, fox=(12.0, 10.0))
    _, tr_gnd = _run(cfg, w_gnd)

    air_a, air_g = _peak(tr_air, spec.M_CALL_AERIAL), _peak(tr_air, spec.M_CALL_GROUND)
    gnd_a, gnd_g = _peak(tr_gnd, spec.M_CALL_AERIAL), _peak(tr_gnd, spec.M_CALL_GROUND)

    ok = air_a > 0.5 and air_a > air_g and gnd_g > 0.5 and gnd_g > gnd_a
    return Probe("referential alarm calls", ok,
                 f"hawk->(aerial {air_a:.2f}, ground {air_g:.2f})  "
                 f"fox->(aerial {gnd_a:.2f}, ground {gnd_g:.2f})")


def contact_call_when_isolated(cfg: CoopConfig) -> Probe:
    """An isolated chick calls until someone answers."""
    cfg1 = cfg._replace(n_hens=1)
    alone = _staged(cfg1, pos=[[10.0, 10.0]], heading=0.0, cold=0.5)
    _, tr_alone = _run(cfg1, alone)
    v_alone = _mean(tr_alone, spec.M_CALL_CONTACT)

    cfg4 = cfg._replace(n_hens=4)
    huddled = _staged(cfg4, pos=[[10.0, 10.0], [10.3, 10.0],
                                 [10.0, 10.3], [10.3, 10.3]],
                      heading=0.0, cold=0.5)
    _, tr_huddled = _run(cfg4, huddled)
    v_huddled = _mean(tr_huddled, spec.M_CALL_CONTACT)

    drop = _modulation(v_alone, v_huddled)
    return Probe("contact call when isolated", drop >= MIN_MODULATION,
                 f"alone={v_alone:.2f} vs in-flock={v_huddled:.2f} "
                f"({100*drop:.0f}% quieter in flock, want >={100*MIN_MODULATION:.0f}%)")


def approach_flockmates(cfg: CoopConfig) -> Probe:
    """A cold chick orients toward flockmates; huddling is the only heat source."""
    cfg = cfg._replace(n_hens=3)
    # Two flockmates off to her left; she starts facing along +x.
    w = _staged(cfg, pos=[[10.0, 10.0], [10.0, 13.0], [10.5, 13.0]],
                heading=0.0, cold=0.9)
    w_end, tr = _run(cfg, w, steps=200)
    left_bias = _mean(tr, spec.M_TURN_L) - _mean(tr, spec.M_TURN_R)
    closed = float(jnp.linalg.norm(w.pos[0] - w.pos[1])
                   - jnp.linalg.norm(w_end.pos[0] - w_end.pos[1]))
    return Probe("approach flockmates when cold",
                left_bias >= MIN_BIAS and closed >= MIN_BIAS,
                 f"left-bias={left_bias:+.2f} closed={closed:+.3f} m")


def food_call_on_arrival_not_continuous(cfg: CoopConfig) -> Probe:
    """Real cockerels food-call on *finding* food, not continuously while standing
    at it (E053). Stage a hen already at a patch: she should call near the start (the
    discovery pulse, `IDX_FOOD_ARRIVAL`) and fall silent well before 5 s despite
    remaining at the food the entire run.
    """
    cfg = cfg._replace(n_hens=1)
    w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, food=[[10.1, 10.0]], hunger=0.5)
    _, tr = _run(cfg, w, steps=500)
    early = float(jnp.max(tr.motor[:50, 0, spec.M_CALL_FOOD]))
    late = float(jnp.mean(tr.motor[400:, 0, spec.M_CALL_FOOD]))
    passed = early > 0.5 and late < 0.1
    return Probe("food call on arrival, not continuous", passed,
                f"early peak={early:.2f} (want >0.5), late mean={late:.2f} (want <0.1)")


def contamination_causes_sickness_onset(cfg: CoopConfig) -> Probe:
    """T2 (E060). Eating from a contaminated patch must make her sick on the very
    next step (rising edge, no delay); the same setup with a clean patch must not.
    """
    cfg = cfg._replace(n_hens=1, n_food=1, contamination_enabled=True)

    def sick_after_one_bite(contaminated):
        w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, food=[[10.05, 10.0]], hunger=0.5)
        w = w._replace(food_contaminated=jnp.array([contaminated]))
        w_end, _ = _run(cfg, w, steps=3)
        return bool(w_end.sick_on[0])

    bad, clean = sick_after_one_bite(True), sick_after_one_bite(False)
    return Probe("sick immediately after eating contaminated food", bad and not clean,
                f"contaminated patch -> sick={bad}, clean patch -> sick={clean}")


def sickness_slows_movement(cfg: CoopConfig) -> Probe:
    """T2 (E060). Sickness is a physiological constraint applied mechanically in
    `actuation.py`, not a reflex or learned choice -- a sick hen should travel much
    less distance than an otherwise-identical healthy one under the same drives.
    """
    cfg = cfg._replace(n_hens=1)

    def distance_travelled(sick):
        w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, hunger=0.9)
        w = w._replace(sick_t=jnp.array([30.0 if sick else 0.0]),
                       sick_on=jnp.array([sick]))
        w_end, _ = _run(cfg, w, steps=300)
        return float(jnp.linalg.norm(w_end.pos[0] - w.pos[0]))

    d_sick, d_healthy = distance_travelled(True), distance_travelled(False)
    passed = d_sick < d_healthy * 0.5
    return Probe("sickness slows movement", passed,
                f"distance sick={d_sick:.3f} m vs healthy={d_healthy:.3f} m")


def gakel_call_on_falling_sick_not_continuous(cfg: CoopConfig) -> Probe:
    """T2 (E060). The gakel call must be a discovery pulse on falling sick, matching
    E053's fix for the food call -- not continuous distress calling for the whole,
    much longer sickness duration.
    """
    cfg = cfg._replace(n_hens=1, n_food=1, contamination_enabled=True)
    w = _staged(cfg, pos=[[10.0, 10.0]], heading=0.0, food=[[10.05, 10.0]], hunger=0.5)
    w = w._replace(food_contaminated=jnp.array([True]))
    w_end, tr = _run(cfg, w, steps=500)
    early = float(jnp.max(tr.motor[:50, 0, spec.M_CALL_GAKEL]))
    late = float(jnp.mean(tr.motor[400:, 0, spec.M_CALL_GAKEL]))
    still_sick = bool(w_end.sick_on[0])   # decay must not just be recovery
    passed = early > 0.5 and late < 0.1 and still_sick
    return Probe("gakel call on falling sick, not continuous", passed,
                f"early peak={early:.2f} (want >0.5), late mean={late:.2f} (want <0.1), "
                f"still sick at step 500={still_sick} (want True)")


def avoid_a_sick_flockmate(cfg: CoopConfig) -> Probe:
    """T2 (E060) innate anchor. A sick, visible flockmate must produce a turning
    bias *away*, reversing the ordinary gregariousness attraction to the same
    flockmate when she is healthy -- checked as a within-setup contrast, not just a
    sign, since CLS_SICK and CLS_FLOCKMATE co-fire on the same bin.
    """
    cfg = cfg._replace(n_hens=2)

    def bias(sick):
        w = _staged(cfg, pos=[[10.0, 10.0], [10.0, 11.5]], heading=0.0)
        w = w._replace(sick_t=jnp.array([0.0, 30.0 if sick else 0.0]),
                       sick_on=jnp.array([False, sick]))
        _, tr = _run(cfg, w, steps=1)
        return _mean(tr, spec.M_TURN_R) - _mean(tr, spec.M_TURN_L)   # positive = away

    away_bias, toward_bias = bias(True), bias(False)
    # Separation, not ordering: "away is bigger than toward" is true of a 0.001
    # difference, which no hen could act on.
    separation = away_bias - toward_bias
    passed = away_bias >= MIN_BIAS and separation >= 4 * MIN_BIAS
    return Probe("avoid a sick flockmate", passed,
                f"right-bias sick={away_bias:+.2f} vs healthy={toward_bias:+.2f} "
                f"(healthy should be negative -- attraction)")


def withdraw_on_hearing_a_gakel_call(cfg: CoopConfig) -> Probe:
    """T2-revised mechanism 1. Hearing the gakel call must suppress *ingestion* -- and
    must do so *specifically*, not as a response to any call -- while leaving locomotion
    alone.

    Checked as a within-setup contrast against a contact call at identical amplitude
    from the identical position, because a scaffold that damped everything on the
    audio bus would pass a bare sign test while being useless: the whole point is that
    the aversive response is tied to this call and not to hearing a flockmate.

    **The forward-drive clause is a guard, not a second effect (E083).** This assay used
    to require forward drive to fall too, and passed happily while the scaffold was
    broken. E082 showed why that was the wrong test: `coop/actuation.py` derives speed
    from `M_FORWARD`, so damping it makes a hen who is already *at* the bad feeder stay
    at it -- lingering, where avoidance needs leaving. So the clause is now inverted.
    Forward drive must **not** fall, and this runs at the configuration where the defect
    appeared, with the hen standing on food, which is the only place it is visible.

    Opt-in, like the alarm scaffold -- this probe is the only place it is switched on.
    """
    cfg = cfg._replace(n_hens=2, n_food=1)
    gakel = spec.CALL_MOTOR_IDX.index(spec.M_CALL_GAKEL)
    contact = spec.CALL_MOTOR_IDX.index(spec.M_CALL_CONTACT)

    def drive(channel, hunger):
        # Food under her beak so peck is genuinely active and has room to fall.
        w = _staged(cfg, pos=[[10.0, 10.0], [10.0, 10.6]], heading=0.0,
                    food=[[10.05, 10.0]], hunger=hunger)
        calls = jnp.zeros((cfg.n_hens, spec.N_CALLS)).at[1, channel].set(1.0)
        w = w._replace(calls=calls)
        _, tr = _run(cfg, w, steps=1, gakel_scaffold=True)
        # head_down is max over HEAD_DOWN_ACTIONS, and sensing.py scales the aerial
        # channel by (1 - head_down) -- so this is literally whether she can look at
        # the place she has just been warned about (E091).
        head_down = max(_mean(tr, c) for c in spec.HEAD_DOWN_ACTIONS)
        return _mean(tr, spec.M_FORWARD), _mean(tr, spec.M_PECK), head_down

    # Both ends, per the specification E090 §8 recorded *before* this was measured: a
    # conditional response cannot be validated at one staged hunger. Sated is where the
    # warning is supposed to win; starving is where she is supposed to override it.
    fwd_g, peck_g, hd_g = drive(gakel, SATED)
    fwd_c, peck_c, hd_c = drive(contact, SATED)
    _f, peck_g_hungry, _h = drive(gakel, STARVING)
    _f, peck_c_hungry, _h = drive(contact, STARVING)
    # 0.01 of motor range: the two conditions differ only in which audio channel is hot,
    # so anything larger than numerical wobble here is a real path onto M_FORWARD.
    suppression = _modulation(peck_c, peck_g)
    supp_hungry = _modulation(peck_c_hungry, peck_g_hungry)
    # A warning she cannot look up from is useless: E091 measured the gakel scaffold
    # suppressing pecking while scratching held the head-down gate shut at 0.269, so a
    # hen who had just been told a place was bad still saw 73% rather than 90% of it.
    # The contact clause keeps this specific -- hearing *anything* must not free her.
    passed = (suppression >= MIN_MODULATION and fwd_g >= fwd_c - 0.01
              and hd_g < 0.15 and hd_c > 0.5
              and supp_hungry < suppression)
    return Probe("withdraw on hearing a gakel call", passed,
                f"gakel peck={peck_g:.3f} vs contact peck={peck_c:.3f} "
                f"({100*suppression:.1f}% suppression, want "
                f">={100*MIN_MODULATION:.0f}% -- see E089); "
                f"suppression when starving {100*supp_hungry:.1f}% "
                f"(must be < sated -- the response is conditional, E090); "
                f"head_down {hd_g:.3f} vs contact {hd_c:.3f} "
                f"(gakel must be <0.15 so she can look up -- E091); "
                f"gakel fwd={fwd_g:.3f} vs contact fwd={fwd_c:.3f} "
                f"(must NOT be lower -- see E082)")


# Assays known to fail, with the reason and what would fix them. Registered here rather
# than weakened in place: an assay that has been softened to pass stops being a guard,
# which is the whole finding of E089/E090. `tests/test_phase0.py` marks these xfail
# *strictly*, so if one starts passing the suite fails and someone has to come and
# update this list.
EXPECTED_FAILURES = {
    "withdraw_on_hearing_a_gakel_call":
        "E089: 3.5% peck suppression against a 25% bar. Food drives M_PECK at +7.0 and "
        "SCAFFOLD_WEIGHT is 1.5, both deep in sigmoid saturation, so the response is "
        "behaviourally inert -- a correctly planted association moved occupancy by "
        "+0.3%. Fixing it is a design decision, not a tuning one: the scaffold is "
        "deliberately held below the visual arc's weights so first-hand information "
        "dominates second-hand, and E089 showed those two goals are incompatible "
        "through a saturating sigmoid. The proposed fix is a hunger term on M_PECK, so "
        "risk tolerance scales with need and the call matters conditionally rather "
        "than always or never.",
}


ALL = (peck_at_food, crouch_at_hawk, head_down_blindness, flee_from_fox,
       referential_alarm, contact_call_when_isolated, approach_flockmates,
       food_call_on_arrival_not_continuous, contamination_causes_sickness_onset,
       sickness_slows_movement, gakel_call_on_falling_sick_not_continuous,
       avoid_a_sick_flockmate, withdraw_on_hearing_a_gakel_call)


def run_all(cfg: CoopConfig = spec.DEFAULT_COOP):
    return [fn(cfg) for fn in ALL]


def main() -> None:
    global BALANCED_EI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced-ei", action="store_true",
                    help="run the whole ethogram on an E/I-balanced connectome "
                         "(E072/E074's adoption gate), instead of the default")
    BALANCED_EI = ap.parse_args().balanced_ei
    print("Neonatal ethogram -- a hen that has never learned anything"
          f"{' [balanced E/I]' if BALANCED_EI else ''}\n")
    results = run_all()
    width = max(len(r.name) for r in results)
    for fn, r in zip(ALL, results):
        expected = fn.__name__ in EXPECTED_FAILURES
        tag = "PASS" if r.passed else ("XFAIL" if expected else "FAIL")
        print(f"  {tag:<5} {r.name:<{width}}  {r.detail}")
    n_pass = sum(r.passed for r in results)
    xfail = [fn.__name__ for fn, r in zip(ALL, results)
             if not r.passed and fn.__name__ in EXPECTED_FAILURES]
    surprise = [fn.__name__ for fn, r in zip(ALL, results)
                if not r.passed and fn.__name__ not in EXPECTED_FAILURES]
    fixed = [fn.__name__ for fn, r in zip(ALL, results)
             if r.passed and fn.__name__ in EXPECTED_FAILURES]
    print(f"\n{n_pass}/{len(results)} assays passed"
          + (f", {len(xfail)} known-failing" if xfail else ""))
    for name in xfail:
        print(f"\n  XFAIL {name}\n    " + EXPECTED_FAILURES[name].replace(". ", ".\n    "))
    if fixed:
        print(f"\n  {len(fixed)} assay(s) in EXPECTED_FAILURES now PASS: "
              + ", ".join(fixed) + "\n  Remove them from the registry.")
    if surprise:
        print(f"\n  {len(surprise)} UNEXPECTED failure(s): " + ", ".join(surprise))


if __name__ == "__main__":
    main()
