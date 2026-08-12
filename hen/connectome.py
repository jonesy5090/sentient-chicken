"""Building the innate connectome.

Two things are genetic and shared by every hen in the flock: the connection *mask*
(which regions wire to which, sampled once from `REGION_CONNECTIVITY`) and each
neuron's excitatory/inhibitory identity. Individual hens differ only in their initial
weight values -- developmental noise, not different blueprints.

The mask is stored explicitly and separately from the weights even though phase 0
never changes it. That is deliberate: phase 1's structural growth is implemented as
flipping a mask bit and initialising a weight, with no reallocation and no
recompilation. Preallocating the maximum connectivity budget now is the one decision
that would be expensive to revisit later.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec
from hen import innate, regions
from hen.regions import Regions


class BrainParams(NamedTuple):
    W: jax.Array          # (H, N, N) recurrent weights, Dale-signed and masked
    mask: jax.Array       # (N, N) bool -- the innate connectome, kept as a reference
    growable: jax.Array   # (N, N) bool -- where an axon could ever reach
    W_in: jax.Array       # (N, OBS_DIM) sensory afferents (shared, fixed)
    W_out: jax.Array      # (H, MOTOR_DIM, n_motor) cortical motor readout
    W_pred: jax.Array     # (H, OBS_DIM, N) top-down associative projection
    pred_src: jax.Array   # (N,) bool -- which neurons may source a prediction
    b: jax.Array          # (N,) resting bias
    tau: jax.Array        # (N,) membrane time constants, seconds
    reflex: jax.Array     # (MOTOR_DIM, OBS_DIM) innate arc, fixed
    b_motor: jax.Array    # (MOTOR_DIM,)
    dale: jax.Array       # (N,) +1 excitatory / -1 inhibitory


def _region_of(reg: Regions) -> np.ndarray:
    """(N,) array giving each neuron's region id."""
    return np.concatenate([np.full(n, r) for r, n in enumerate(reg.sizes)])


def build(key: jax.Array, reg: Regions = regions.DEFAULT_REGIONS,
          n_hens: int = spec.DEFAULT_COOP.n_hens,
          gain: float = 0.70, readout_scale: float = 0.05) -> BrainParams:
    """Sample a newly hatched flock.

    `readout_scale` is small on purpose: at hatch the cortical pathway is near-silent
    and behaviour is dominated by the innate reflex arc. The pallium is present and
    connected, but it has nothing to say yet.

    `gain` was 0.9 for E001-E009 and that was a mistake, found in
    docs/experiments/E009. At 0.9 the pallium runs at a mean rate of 0.83 -- deep in
    the flat part of the sigmoid, slope ~0.12 -- and the states for "heard an alarm
    call" and "saw a hawk" differ by under 1% of the mean. A saturated network cannot
    represent distinctions, and nothing downstream can learn from a representation
    that does not distinguish.

    Measured across genomes, relative separability of those two percepts:

        gain   mean pallial rate   separability (% of mean rate)
        0.60         0.212               3.3%
        0.70         0.271               7.5%
        0.75         0.349              14.2%   <- measured optimum
        0.78         0.497               6.2%
        0.90         0.830               0.9%   <- the old default

    0.70 rather than the peak at 0.75, deliberately. The peak sits about 0.03 from a
    transition: by 0.78 the mean rate has jumped to 0.50 and separability has
    collapsed. Weights move during learning, so a value that has to be held to two
    decimal places is not a value to build on. 0.70 keeps 8x the old separability, and
    its mean rate is tight across genomes (0.26-0.28) where 0.78's is not (0.42-0.59).

    Separability varies a lot between genomes at any gain (3.5%-25.5% at 0.70), which
    is individual variation in how well a given hen's wiring separates her world. That
    is interesting rather than a defect, but it means per-seed results are noisy and
    contrasts need replicates.
    """
    k_mask, k_w, k_in = jax.random.split(key, 3)
    n = reg.total
    rid = _region_of(reg)

    # --- Genetic mask: P[source region, target region], W[i, j] is j -> i ---
    p = np.asarray(regions.REGION_CONNECTIVITY, dtype=np.float32)
    p_ij = p[rid[None, :], rid[:, None]]                     # (N, N)
    mask = jax.random.uniform(k_mask, (n, n)) < jnp.asarray(p_ij)
    mask = mask & ~jnp.eye(n, dtype=bool)                    # no autapses

    # Where a synapse could ever appear. Phase 1 growth is confined to this: axons
    # only reach regions their tract projects to, so a hypothalamic neuron cannot
    # sprout onto the sensory stub however well correlated the two happen to be.
    growable = jnp.asarray(p_ij > 0.0) & ~jnp.eye(n, dtype=bool)

    # --- Dale's law: a neuron's outgoing weights all share its sign ---
    n_exc = int(round(regions.EXCITATORY_FRACTION * n))
    dale = jnp.asarray(np.where(np.arange(n) < n_exc, 1.0, -1.0), dtype=jnp.float32)

    fan_in = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    w_raw = jnp.abs(jax.random.normal(k_w, (n_hens, n, n))) * (gain / jnp.sqrt(fan_in))
    w = w_raw * mask[None] * dale[None, None, :]

    # --- Afferents. Exteroceptive channels reach the sensory stub; interoceptive
    # drives reach the hypothalamus. Nothing else gets direct input.
    w_in = np.zeros((n, spec.OBS_DIM), dtype=np.float32)
    s_lo, s_hi = reg.bounds(regions.SENSORY)
    h_lo, h_hi = reg.bounds(regions.HYPOTHALAMUS)

    rng = np.random.default_rng(int(jax.random.randint(k_in, (), 0, 2**30)))
    extero = [i for i in range(spec.OBS_DIM)
              if not (spec.INTERO_LO <= i < spec.INTERO_HI)]
    w_in[s_lo:s_hi, extero] = rng.gamma(2.0, 0.5, (s_hi - s_lo, len(extero))) \
        * (rng.random((s_hi - s_lo, len(extero))) < 0.3)
    w_in[h_lo:h_hi, spec.INTERO_LO:spec.INTERO_HI] = rng.gamma(
        2.0, 1.0, (h_hi - h_lo, spec.INTERO_HI - spec.INTERO_LO))

    m_lo, m_hi = reg.bounds(regions.MOTOR)
    w_out = jax.random.normal(
        jax.random.fold_in(k_w, 1), (n_hens, spec.MOTOR_DIM, m_hi - m_lo)
    ) * readout_scale

    # Top-down projection onto the sensory representation the reflex arc reads.
    # Starts at exactly zero: a newly hatched hen predicts nothing and perceives only
    # what is in front of her. Every association she ever has is one she formed.
    w_pred = jnp.zeros((n_hens, spec.OBS_DIM, n))

    # Predictions come from the pallium only, never from the sensory stub. E008 found
    # the first version was circular: sourced from the whole brain, it learned "when
    # in hawk-state, predict hawk", because the sensory stub carries the hawk percept
    # directly and dominates the association. Association cortex, one step removed
    # from the relay, is also where top-down predictions come from in a real brain.
    p_lo, p_hi = reg.bounds(regions.PALLIUM)
    pred_src = jnp.zeros((n,), dtype=bool).at[p_lo:p_hi].set(True)

    tau = jnp.asarray(np.asarray(regions.REGION_TAU, dtype=np.float32)[rid])

    return BrainParams(
        W=w,
        mask=mask,
        growable=growable,
        W_in=jnp.asarray(w_in),
        W_out=w_out,
        W_pred=w_pred,
        pred_src=pred_src,
        b=jnp.full((n,), -2.0),
        tau=tau,
        reflex=jnp.asarray(innate.reflex_matrix()),
        b_motor=jnp.asarray(innate.reflex_bias()),
        dale=dale,
    )


def stats(p: BrainParams, reg: Regions = regions.DEFAULT_REGIONS) -> dict:
    """Descriptive numbers for the README and the benchmark banner."""
    n = reg.total
    n_syn = int(jnp.sum(p.mask))
    return {
        "neurons": n,
        "synapses": n_syn,
        "mean_fan_out": n_syn / n,
        "density": n_syn / (n * n),
        "regions": dict(zip(regions.REGION_NAMES, reg.sizes)),
    }
