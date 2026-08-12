"""Phase 1: learning.

A three-factor rule. Two of the factors are local -- traces of pre- and postsynaptic
activity -- and the third is a global neuromodulator standing in for dopamine:

    dW[i,j]  =  eta(age) * m * z_slow[i] * z_fast[j]

The neuromodulator `m` is a reward *prediction error*: homeostatic improvement minus
a running expectation of it. Reward is drive reduction -- eating when hungry, warming
when cold, escaping when threatened. Nothing here is a task objective handed down
from outside; a hen is rewarded for staying alive and comfortable, which is the only
thing a real animal optimises.

Three design points worth stating, because each is a place this could have gone wrong:

**Eligibility is factored, not stored per synapse.** A true pre-post correlation trace
would need an (H, N, N) tensor -- doubling the memory traffic of a simulation that is
already bandwidth bound. Filtering pre and post separately and taking the outer
product at update time is the standard approximation and costs two vectors per hen.

**Weights are written every `interval` steps, not every step.** The forward pass
already reads W each step; writing it too would double bandwidth and halve the
real-time factor. Eligibility accumulates continuously and consolidates periodically,
which is also closer to how synaptic consolidation actually works.

**The reflex arc is untouched.** Plasticity applies to the cortical pathway only.
A chick's innate responses stay fixed for life, as they do in a real bird -- what
develops is the pallium's ability to modulate them.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop.spec import CoopConfig
from hen.connectome import BrainParams


class PlasticConfig(NamedTuple):
    enabled: bool = True

    # Eligibility traces. The asymmetry between the two time constants is what gives
    # the rule a direction: post integrates over a longer window than pre, a
    # rate-coded stand-in for the timing asymmetry of STDP.
    tau_fast: float = 0.02        # presynaptic, seconds
    tau_slow: float = 0.20        # postsynaptic
    tau_motor: float = 0.10

    # Learning rate and the critical period. Plasticity is highest at hatch and
    # decays with age, so early experience shapes the connectome disproportionately.
    eta: float = 4e-3
    # The readout rate is 10x the recurrent one, and that asymmetry is load-bearing.
    # The pallium starts near-silent by design (readout_scale=0.05), so it has to
    # *earn* influence over behaviour before anything it learns can reach a muscle.
    # Measured over 10 min of chicken time (E002): at 2e-3 the readout does not grow
    # at all (1.00x) and cortical drive stays at 0.16 of the reflex arc; at 2e-2 it
    # grows 1.17x and reaches 0.51. At 2e-1 cortical drive overwhelms the innate arc
    # (1.72x) and behaviour gets worse -- a hen who overrides her reflexes with an
    # untrained pallium is worse off than one who does not.
    eta_out: float = 2e-2
    critical_period_s: float = 3 * 86400.0

    # Neuromodulation
    reward_scale: float = 60.0
    strike_penalty: float = 1.0
    baseline_tau_s: float = 20.0

    # Weight on flockmates' welfare in a hen's own reward signal. This is kin
    # selection, which is the standard explanation for why alarm calling evolves in
    # group-living birds at all: warning a relative propagates your own genes. A
    # flock of hens is typically related.
    #
    # It is needed here for a specific reason. Under purely individual reward a call
    # can never pay -- the caller bears the energetic cost and the *listener* gets
    # the benefit -- so a hen would learn to fall silent and H3 could not be tested.
    #
    # Note what this does NOT do: nothing here rewards calling *when an audience is
    # present*. It rewards flockmates doing well, and it charges for calling. That
    # audience-conditional calling is the optimal policy is the hypothesis, not the
    # instruction.
    kin_weight: float = 0.5

    # Whether the kin term is weighted by audibility. See `reward()`: the flat flock
    # mean cannot assign credit to a caller, which is E005's diagnosis of its own
    # null. False reproduces the E005 condition.
    kin_audible: bool = True

    # Consolidation. The forward pass already reads W every step; writing it too is
    # what costs, so eligibility accumulates continuously and consolidates every
    # `interval` steps. Measured: interval=10 costs 3x throughput, 50 costs ~25%,
    # and beyond 50 the returns flatten. 50 steps is 500 ms of chicken time, far
    # finer than any behaviour we care about.
    interval: int = 50            # steps between weight writes
    w_max: float = 0.5
    scaling_strength: float = 0.02  # synaptic scaling back toward innate total

    # Exploration. Young animals are behaviourally variable, and that variability is
    # the substrate reinforcement acts on -- a deterministic policy can only ever be
    # rewarded for actions it already takes. E006 established that without this the
    # hen cannot acquire any stimulus-response pairing her innate reflex arc does not
    # already visit, which blocks every hypothesis about learned communication.
    #
    # Decays with age on the same schedule as the critical period: a chick explores,
    # an adult exploits.
    explore_sigma: float = 0.6      # s.d. on the motor drive, at hatch
    explore_tau_s: float = 3600.0   # halves roughly every hour of chicken time

    # Top-down association (H2c). Learned by a masked delta rule that is NOT
    # reward-modulated: association between co-occurring stimuli needs no reward and
    # no attribution of benefit to anyone, which is exactly the credit-assignment
    # problem that sank E005 and E006.
    pred_enabled: bool = False      # opt-in, so it is a contrast rather than a change
    pred_gain: float = 1.0          # how strongly prediction augments perception
    eta_pred: float = 5e-3
    tau_lag: float = 1.5            # seconds; bridges cue to outcome
    pred_max: float = 0.05          # per-synapse cap on the top-down projection

    # Structural growth
    growth_enabled: bool = True
    growth_interval: int = 10_000   # steps (100 s of chicken time)
    growth_rate: float = 0.02       # P(new synapse) at mean coactivity
    prune_threshold: float = 5e-3   # |w| below this is a candidate for pruning
    prune_rate: float = 0.05
    synapse_budget: float = 1.5     # multiple of the innate synapse count


class PlasticState(NamedTuple):
    z_fast: jax.Array      # (H, N) presynaptic trace
    z_slow: jax.Array      # (H, N) postsynaptic trace
    z_motor: jax.Array     # (H, MOTOR_DIM) motor output trace
    z_err: jax.Array       # (H, OBS_DIM) masked prediction error trace
    z_lag: jax.Array       # (H, N) slow trace: what the brain was doing before
    baseline: jax.Array    # (H,) running expectation of reward
    age_s: jax.Array       # scalar, seconds since hatch
    innate_row_sum: jax.Array   # (H, N) reference total input weight per neuron
    budget: jax.Array      # scalar, maximum live synapses per hen


def initial_state(p: BrainParams, n_hens: int, pc: PlasticConfig) -> PlasticState:
    n = p.b.shape[0]
    row_sum = jnp.sum(jnp.abs(p.W), axis=2)
    return PlasticState(
        z_fast=jnp.zeros((n_hens, n)),
        z_slow=jnp.zeros((n_hens, n)),
        z_motor=jnp.zeros((n_hens, p.W_out.shape[1])),
        z_err=jnp.zeros((n_hens, p.W_pred.shape[1])),
        z_lag=jnp.zeros((n_hens, n)),
        baseline=jnp.zeros((n_hens,)),
        age_s=jnp.array(0.0),
        innate_row_sum=row_sum,
        budget=jnp.sum(p.mask) * pc.synapse_budget,
    )


def reward(w_prev, w_next, cfg: CoopConfig, pc: PlasticConfig) -> jax.Array:
    """Homeostatic improvement per second, minus the cost of being caught.

    Drive *reduction* is positive. There is no task term: the hen is rewarded for
    being fed, watered, warm and uneaten, and any foraging strategy she develops has
    to be discovered from that alone.

    A fraction `kin_weight` of the flock's mean welfare is added to each hen's own.
    See `PlasticConfig.kin_weight` -- without it a call can never repay its energetic
    cost, because the caller pays and the listener benefits.
    """
    # Drives are costs, so reduction is positive. Vigour is a resource, so its
    # *loss* is the cost -- this is what makes calling expensive enough for
    # audience-sensitivity to have anything to emerge from, without charging it to
    # hunger and destroying H2's metric (E012).
    d_drive = ((w_prev.hunger - w_next.hunger)
               + (w_prev.thirst - w_next.thirst)
               + (w_prev.cold - w_next.cold)
               + (w_next.vigour - w_prev.vigour)) / cfg.dt
    # Being caught is a discrete *event*, not a rate, and it must not be divided by
    # dt. It was, until E014. At dt=0.01 that turned a single strike into -100 in a
    # hen's reward -- roughly 150x anything the drive terms contribute (~0.6) -- and
    # the baseline, which tracks over 20 s, cannot absorb a spike like that. The
    # modulator then slams every eligible synapse at once.
    #
    # Measured: seeds where hens were struck lost 50-75% of their connectome; a seed
    # that happened to take zero strikes lost 17% and was fine. Ablating this term
    # restored every seed to ~83%. The units error was the whole of the erosion that
    # E013 attributed to a random walk.
    struck = w_next.n_struck - w_prev.n_struck
    own = d_drive * pc.reward_scale - struck * pc.strike_penalty

    if pc.kin_weight == 0.0:
        return own

    n = own.shape[0]
    if not pc.kin_audible:
        # Flat flock mean over the *other* hens: (sum - self) / (n - 1).
        # E005 showed this cannot work: the quantity is nearly identical for every
        # hen, so it moves the same way for the silent bird as for the caller, and
        # no hen can discover she is responsible for a call. Kept as a condition to
        # contrast against, not as a default.
        others = (jnp.sum(own) - own) / jnp.maximum(n - 1, 1)
        return own + pc.kin_weight * others

    # Audibility-weighted: a hen's reward reflects the welfare of the flockmates who
    # could actually hear her. Now the kin term differs per hen and covaries with who
    # was in earshot, which is the correlation the learning rule needs to credit her
    # own call. It is also better biology -- kin selection operates on the relatives
    # you actually help, not on a population average.
    d = jnp.linalg.norm(w_next.pos[:, None, :] - w_next.pos[None, :, :], axis=-1)
    d = d + jnp.eye(n) * 1e6                       # exclude self
    audible = jnp.clip(1.0 - d / cfg.hear_range, 0.0, 1.0)
    heard = (audible @ own) / (jnp.sum(audible, axis=-1) + 1e-6)
    return own + pc.kin_weight * heard


def update_traces(ps: PlasticState, r: jax.Array, motor: jax.Array,
                  reward_now: jax.Array, cfg: CoopConfig, pc: PlasticConfig,
                  pred_err: jax.Array = None) -> PlasticState:
    """Decay the eligibility traces toward current activity. Called every step."""
    a_f = cfg.dt / pc.tau_fast
    a_s = cfg.dt / pc.tau_slow
    a_m = cfg.dt / pc.tau_motor
    a_b = cfg.dt / pc.baseline_tau_s
    err = ps.z_err if pred_err is None else ps.z_err + a_s * (pred_err - ps.z_err)
    a_l = cfg.dt / pc.tau_lag
    return ps._replace(
        z_err=err,
        z_lag=ps.z_lag + a_l * (r - ps.z_lag),
        z_fast=ps.z_fast + a_f * (r - ps.z_fast),
        z_slow=ps.z_slow + a_s * (r - ps.z_slow),
        z_motor=ps.z_motor + a_m * (motor - ps.z_motor),
        baseline=ps.baseline + a_b * (reward_now - ps.baseline),
        age_s=ps.age_s + cfg.dt,
    )


def _enforce_dale(w: jax.Array, dale: jax.Array, w_max: float) -> jax.Array:
    """Keep every weight on the sign of its presynaptic neuron, and bounded.

    Learning must not be able to turn an inhibitory neuron excitatory. Clamping the
    magnitude at zero also means a weight driven to zero simply leaves the
    connectome, which is what makes pruning fall out of the same operation.
    """
    magnitude = jnp.clip(w * dale[None, None, :], 0.0, w_max)
    return magnitude * dale[None, None, :]


def consolidate(p: BrainParams, ps: PlasticState, m: jax.Array,
                pc: PlasticConfig) -> BrainParams:
    """Write accumulated eligibility into the weights. Called every `interval` steps."""
    eta = pc.eta / (1.0 + ps.age_s / pc.critical_period_s)
    eta_out = pc.eta_out / (1.0 + ps.age_s / pc.critical_period_s)

    # Recurrent weights. Only synapses that already exist are updated; growing new
    # ones is a separate, much rarer operation.
    dw = eta * m[:, None, None] * ps.z_slow[:, :, None] * ps.z_fast[:, None, :]
    w = p.W + dw * (p.W != 0.0)

    # Synaptic scaling: pull each neuron's total input weight back toward what it
    # was born with. Without it, a sustained positive modulator ratchets every
    # weight upward until the network saturates.
    row_sum = jnp.sum(jnp.abs(w), axis=2)
    correction = 1.0 + pc.scaling_strength * (
        ps.innate_row_sum / (row_sum + 1e-6) - 1.0)
    w = w * correction[:, :, None]
    w = _enforce_dale(w, p.dale, pc.w_max)

    # Motor readout: this is how the pallium earns influence over behaviour. Without
    # it the cortical pathway stays near-silent forever and nothing it learns can
    # reach a muscle.
    n_motor = p.W_out.shape[-1]
    dw_out = (eta_out * m[:, None, None]
              * ps.z_motor[:, :, None] * ps.z_slow[:, None, -n_motor:])
    w_out = jnp.clip(p.W_out + dw_out, -pc.w_max, pc.w_max)

    w_pred = p.W_pred
    if pc.pred_enabled:
        # Masked delta rule: move the prediction toward the observation, using only
        # channels that were actually observable. Factored into traces for the same
        # reason as the recurrent rule -- storing a per-synapse error would double the
        # memory traffic of a simulation that is already bandwidth bound.
        # Error now, against what the brain was doing a moment ago, and only from
        # neurons allowed to source a prediction.
        d_pred = (eta_pred_of(ps, pc) * ps.z_err[:, :, None]
                  * (ps.z_lag * p.pred_src[None, :])[:, None, :])
        w_pred = jnp.clip(p.W_pred + d_pred, -pc.pred_max, pc.pred_max)

    return p._replace(W=w, W_out=w_out, W_pred=w_pred)


def eta_pred_of(ps: PlasticState, pc: PlasticConfig):
    """Association rate, on the same critical-period decay as everything else."""
    return pc.eta_pred / (1.0 + ps.age_s / pc.critical_period_s)


def restructure(p: BrainParams, ps: PlasticState, key: jax.Array,
                pc: PlasticConfig) -> BrainParams:
    """Grow and prune synapses. Called every `growth_interval` steps.

    Growth is stochastic and gated on coactivity: a pair of neurons that fire
    together is a candidate for wiring together. It is also anatomically
    constrained -- a synapse can only appear where the innate connectivity priors
    allow an axon to be, so a hypothalamic cell cannot sprout onto the sensory stub
    just because the two happen to correlate.

    Pruning is not a separate mechanism. A weight driven to zero by learning is
    already gone; this only removes the ones loitering near zero.
    """
    live = p.W != 0.0
    n_live = jnp.sum(live, axis=(1, 2), keepdims=True)

    coactive = ps.z_slow[:, :, None] * ps.z_fast[:, None, :]
    mean_co = jnp.mean(coactive, axis=(1, 2), keepdims=True) + 1e-9
    p_grow = jnp.clip(pc.growth_rate * coactive / mean_co, 0.0, 0.5)
    p_grow = jnp.where(n_live >= ps.budget, 0.0, p_grow)   # respect the budget

    # One random tensor serves both decisions. Growth only considers dead synapses
    # and pruning only live ones, so the two draws are over disjoint supports and
    # cannot correlate. At (H, N, N) these allocations are the expensive part of
    # restructuring, so three of them cost real throughput and one does not.
    u = jax.random.uniform(key, p.W.shape)

    born = (~live) & p.growable[None, :, :] & (u < p_grow)
    doomed = live & (jnp.abs(p.W) < pc.prune_threshold) & (u < pc.prune_rate)

    # New synapses start weak, on the sign their presynaptic neuron is allowed.
    seed_w = pc.prune_threshold * 2.0 * p.dale[None, None, :]

    w = jnp.where(born, seed_w, p.W)
    w = jnp.where(doomed, 0.0, w)
    return p._replace(W=w)


def divergence_from_genome(p: BrainParams) -> dict:
    """How far a hen's connectome has drifted from the one she hatched with."""
    live = p.W != 0.0
    innate = p.mask[None, :, :]
    return {
        "live_synapses": jnp.sum(live, axis=(1, 2)),
        "grown": jnp.sum(live & ~innate, axis=(1, 2)),
        "pruned": jnp.sum(~live & innate, axis=(1, 2)),
    }
