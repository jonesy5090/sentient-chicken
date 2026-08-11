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


class Probe(NamedTuple):
    name: str
    passed: bool
    detail: str


def _connectome(n_hens: int):
    """Same genome for every probe; width follows the staged flock size."""
    if n_hens not in _CONNECTOME_CACHE:
        _CONNECTOME_CACHE[n_hens] = connectome.build(
            jax.random.key(GENOME_SEED), regions.DEFAULT_REGIONS, n_hens=n_hens)
    return _CONNECTOME_CACHE[n_hens]


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


def _run(cfg: CoopConfig, w, steps: int = 120):
    p = _connectome(cfg.n_hens)
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

    return Probe("contact call when isolated", v_alone > v_huddled + 0.1,
                 f"alone={v_alone:.2f} vs in-flock={v_huddled:.2f}")


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
    return Probe("approach flockmates when cold", left_bias > 0.0 and closed > 0.0,
                 f"left-bias={left_bias:+.2f} closed={closed:+.3f} m")


ALL = (peck_at_food, crouch_at_hawk, head_down_blindness, flee_from_fox,
       referential_alarm, contact_call_when_isolated, approach_flockmates)


def run_all(cfg: CoopConfig = spec.DEFAULT_COOP):
    return [fn(cfg) for fn in ALL]


def main() -> None:
    print("Neonatal ethogram -- a hen that has never learned anything\n")
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:<{width}}  {r.detail}")
    print(f"\n{sum(r.passed for r in results)}/{len(results)} assays passed")


if __name__ == "__main__":
    main()
