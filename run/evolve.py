"""Selection across generations, instead of learning within a lifetime (E116).

Six explanations for H2's null have been proposed and all six failed — five about the
learning rule or the readout, and E115's about the anatomy. What they share is that they
all tried to make *within-lifetime* plasticity work.

This does not. Plasticity is off throughout. A flock lives, is scored on how well it
stayed alive and comfortable, and the best hens' connectomes are copied with mutation into
the next generation. There is no credit-assignment problem to solve: selection does not
need to know which synapse helped, only which hen did.

`docs/backlog.md` §4 has been asking for this since before E001, for a different reason —
compositional structure needs a transmission bottleneck, and that is generational turnover
by definition. This module is the machinery both arguments need.

**What is heritable.** Per-hen `W` and `W_out` only. The connectivity `mask`, `W_in`,
`dale`, the resting biases and the reflex arc are *shared across the flock*, so per-hen
selection cannot vary them — this is selection on synaptic weights inside a fixed anatomy,
and that is a real limit rather than an oversight.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec
from coop.spec import CoopConfig
from hen import brain, connectome, plasticity, regions
from run import simulate

# Plasticity off, but `enabled=True` is still needed for the world to advance its own
# state the way every other experiment runs it; `eta` and `eta_out` at zero are what
# make the connectome fixed within a lifetime.
NO_LEARNING = plasticity.PlasticConfig(enabled=True, eta=0.0, eta_out=0.0,
                                       growth_enabled=False)


class EvoConfig(NamedTuple):
    generations: int = 20
    lifetime_s: float = 600.0      # 10 minutes per generation
    n_parents: int = 4             # truncation selection: top 4 of 16
    mutation: float = 0.05         # sigma, as a fraction of each matrix's own scale
    select: bool = True            # False = parents drawn at random, the control
    # Weight on the predation term of `fitness`, chosen by measurement rather than
    # guessed -- see that function. 0.10 equalises the two terms' spreads.
    caught_weight: float = 0.10
    # Standing variation in the founding population (E116 6b). `connectome.build`
    # produces a flock too genetically uniform for selection to see: at its natural
    # spread a hen's fitness repeatability -- same brains, same world, only exploration
    # noise differing -- is ~0, so the top 4 are the lucky 4 and selection concentrates
    # luck. Amplifying the founders to ~2.7x that spread takes repeatability to 0.47.
    # You cannot select without standing variation; this supplies it, once, at
    # generation 0.
    founder_sigma: float = 0.0


def fitness(w_end, cfg: CoopConfig, evo: "EvoConfig") -> jax.Array:
    """How well each hen stayed alive and comfortable. (H,), higher is better.

    Deliberately not a task objective. The same homeostatic basis `plasticity.reward`
    uses — drives are costs, so low is good — minus the cost of having been caught. A
    hen is selected for being fed, warm, watered and uneaten, and any foraging strategy
    has to be discovered from that alone.

    **The weighting was measured, not assumed, and the first version was wrong.** It
    used `strike_penalty * n_strike_events`, and over a 10-minute life that term has a
    standard deviation of **78.7** against the drives' **0.075** — a factor of a
    thousand. Fitness was predation and nothing else, which is E019's "the reward is 87%
    `n_struck`" defect reappearing in a new place, and the smoke test caught it: fitness
    spread went from 0.02 at generation 0 to 36 two generations later.

    So this uses `n_caught_any`, the event-anchored counter E027 established as the
    interpretable one (mean 1.23, sd 0.74 per hen), and `caught_weight` defaults to
    **0.10**, which is the value that equalises the two spreads. Measured at that
    weight: predation accounts for **51%** of fitness variance, and generation-0
    coefficient of variation is **0.094**.
    """
    drives = w_end.hunger + w_end.cold + w_end.thirst
    return -drives - evo.caught_weight * w_end.n_caught_any.astype(jnp.float32)


def _mutate(p, key, sigma: float):
    """Gaussian noise on live synapses of `W` and `W_out`, Dale enforced.

    Scaled by each matrix's own RMS so one sigma means the same thing for both. Dale is
    applied exactly as `consolidate` applies it: mutation must not be able to turn an
    inhibitory neuron excitatory, which is the invariant learning is held to and there is
    no reason evolution should be exempt from it.
    """
    k_w, k_out = jax.random.split(key)
    n_motor = p.W_out.shape[-1]

    scale_w = jnp.sqrt(jnp.mean(p.W ** 2)) + 1e-9
    w = p.W + sigma * scale_w * jax.random.normal(k_w, p.W.shape) * (p.W != 0.0)
    w = plasticity._enforce_dale(w, p.dale, plasticity.PlasticConfig().w_max)

    scale_o = jnp.sqrt(jnp.mean(p.W_out ** 2)) + 1e-9
    w_out = p.W_out + sigma * scale_o * jax.random.normal(k_out, p.W_out.shape)
    w_out = plasticity._enforce_dale(w_out, p.dale[-n_motor:],
                                     plasticity.PlasticConfig().w_max)
    return p._replace(W=w, W_out=w_out)


def _breed(p, scores, key, evo: EvoConfig):
    """Truncation selection, then mutation. Returns (offspring, parent_index_per_child).

    The control (`select=False`) is identical in every respect except that parents are
    drawn uniformly at random instead of by rank, so mutation load and population size
    are matched exactly and the only difference is whether fitness is consulted.
    """
    n_hens = scores.shape[0]
    k_pick, k_mut = jax.random.split(key)
    if evo.select:
        parents = jnp.argsort(-scores)[:evo.n_parents]
    else:
        parents = jax.random.choice(k_pick, n_hens, (evo.n_parents,), replace=False)
    # Each parent contributes an equal share of the next generation.
    child_parent = jnp.repeat(parents, n_hens // evo.n_parents)[:n_hens]
    p = p._replace(W=p.W[child_parent], W_out=p.W_out[child_parent],
                   W_pred=p.W_pred[child_parent], W_gate=p.W_gate[child_parent],
                   W_str=p.W_str[child_parent])
    return _mutate(p, k_mut, evo.mutation), child_parent


def run_lineage(key, cfg: CoopConfig, evo: EvoConfig, reg=None,
                pc: plasticity.PlasticConfig = NO_LEARNING):
    """One lineage. Yields a record per generation.

    The world is reset every generation from the *same* key, so a fitness difference
    between generations is a difference between connectomes and not between coops.
    """
    reg = regions.DEFAULT_REGIONS if reg is None else reg
    steps = int(evo.lifetime_s / cfg.dt)
    k_genome, k_world, k_run = jax.random.split(key, 3)
    p = connectome.build(k_genome, reg, n_hens=cfg.n_hens)
    if evo.founder_sigma > 0.0:
        p = _mutate(p, jax.random.fold_in(k_genome, 77), evo.founder_sigma)

    history = []
    prev_scores = None
    for gen in range(evo.generations):
        from coop import world
        w0 = world.reset(k_world, cfg)
        x0 = brain.initial_state(p, cfg.n_hens)
        ps = plasticity.initial_state(p, cfg.n_hens, pc)
        w_end, *_ = simulate.rollout_quiet(
            w0, x0, p, jax.random.fold_in(k_run, gen), cfg, steps, ps, pc)
        scores = fitness(w_end, cfg, evo)
        history.append(dict(
            gen=gen,
            fitness=jnp.asarray(scores),
            hunger=jnp.mean(w_end.hunger),
            caught=(jnp.sum(w_end.n_caught_any)
                    / jnp.maximum(jnp.sum(w_end.n_dives), 1.0)),
            # Population diversity: mean pairwise distance between hens' readouts. If
            # this collapses there is nothing left for selection to act on and a plateau
            # says nothing about the search.
            diversity=jnp.mean(jnp.abs(p.W_out[:, None] - p.W_out[None, :])),
            parents=prev_scores,
        ))
        p, child_parent = _breed(p, scores, jax.random.fold_in(k_run, 1000 + gen), evo)
        prev_scores = jnp.asarray(scores)[child_parent]
    return history
