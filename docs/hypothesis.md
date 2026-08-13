# The hypothesis tree

The purpose of this file is to stop the project drifting into testing whatever is
convenient. Every experiment must name the hypothesis it feeds, and every result must
come back here and change something — a status, a prediction, or the tree itself.

**Rule: no experiment is run without a parent hypothesis recorded here.** If a
question does not ladder up to H0, either it belongs in the backlog as a new branch,
or it does not belong in the project.

Status values: `SUPPORTED` · `UNDER TEST` · `NOT STARTED` · `REFUTED` · `ABANDONED`

---

## H0 — root

> A neural model of a chicken, in an environment simple enough that its vision and
> motor budget can be spent elsewhere, can be given a communication channel that
> **measurably changes what the flock is able to do**.

Two halves, and they fail independently: the bird has to be a credible bird, and the
channel has to do real work. H1–H3 are the first half. H4–H6 are the second.

**Falsified if:** a flock with an intact channel performs no better than a
capacity-matched flock with a shuffled one, at any capacity, on any task requiring
private information.

---

## H1 — a small rate-coded network with innate reflexes behaves like a hatchling

**Status: SUPPORTED** (phase 0)

**Prediction:** documented neonatal chick behaviours appear without learning —
indiscriminate pecking at contrast, crouching at overhead looming, fleeing ground
threats, distress calling when isolated, approaching flockmates when cold, and
*distinct* aerial vs terrestrial alarm calls.

**Evidence:** 7/7 assays in `run/probes.py`. Referential alarm separation measured at
hawk→(aerial 0.99, ground 0.12), fox→(aerial 0.08, ground 0.94).

**Falsifier:** any assay failing, or passing only under hand-tuned staging.

---

## H1a — the environment creates an information asymmetry the flock cannot resolve individually

**Status: SUPPORTED** (phase 0)

This is the load-bearing precondition for the whole language half. If every hen can
always see every threat, no signal is ever worth making.

**Prediction:** a foraging hen is measurably blind to aerial threats.

**Evidence:** aerial channel 0.01 while pecking vs 0.87 head-up; crouch 0.06 vs 0.98,
in the same bird moments apart. Across an hour of flock time, hens are head-down ~64%
of the time.

**Falsifier:** head-down fraction near zero (nobody is ever blind) or near one
(nobody ever sees anything, so nobody has information to share).

---

## H2 — three-factor plasticity produces measurable behavioural improvement

**Status: REFUTED at this timescale** —
[E013](experiments/E013-clean-test-of-h2.md), the first test with no confound left.

Learning does not merely fail to help; it makes hens **significantly worse**
(+0.062 ± 0.016, t=3.85, p<0.05) and they feed on 4.7% of timesteps against the
control's 6.2%. Exploration is exonerated: the noise-only control is
indistinguishable from fixed (t=0.32).

~~**The smoking gun is the synapse count.** Learning ends with 19,088 of 36,373
innate synapses, destroying 48% of the connectome. The reward prediction error hovers
near zero, so updates approximate a random walk that the Dale clamp makes
irreversible — a ratchet.~~

**Struck through: E014 disproved this in two ways.** The erosion was a units bug (a
strike contributing −100 to reward), not a random walk — and fixing it recovered the
connectome without moving the behaviour at all. Pruning and harm merely co-occurred.

**E004 is reinterpreted, not withdrawn.** In the saturated regime the readout could
only apply a constant bias to the motor drive, so the same erosion could not reach
behaviour and what learning tuned was an offset that happened to help. A tuned offset
is not a learned policy.

**[E014](experiments/E014-strike-units-bug.md) found the erosion's cause and it did
not rescue H2.** Reward subtracted `struck * strike_penalty / dt` — but being caught
is a discrete *event*, not a rate, so at dt=0.01 a single strike contributed **−100**,
about 150x what the drive terms contribute. Erosion tracked strikes exactly (seeds
with zero strikes lost 17%; seeds with thousands lost 50–75%). Fixed, all seeds
survive 81–86%.

**The behaviour did not move**: +0.081 → +0.082, still significantly worse (t=3.46).
So connectome destruction and behavioural harm were two separate things that merely
appeared together — E013's mechanism story linked them and was wrong twice over.

**[E015](experiments/E015-decomposing-the-harm.md) decomposes the harm between the
two learned pathways:**

| what learns | harm vs fixed | synapses |
|---|---|---|
| recurrent only | +0.010 | 28,383 |
| readout only | +0.021 | 36,369 |
| **both** | **+0.052** | 30,109 |

The readout is about twice as harmful as the recurrent weights — and **the two
together are superadditive**, +0.052 against the +0.031 independence would predict.

~~Plausibly a moving-target problem: the readout chases a pallial representation being
rewritten underneath it.~~ **Withdrawn by
[E016](experiments/E016-staged-learning.md)**, which staged the two pathways and got
the opposite of what that predicts: letting the pallium settle *first* does not help
at all (t=0.25), while doing the readout first cuts the harm 69% (+0.052 → +0.016,
significant). **New reading — labelled a hypothesis:** harm is dominated by whichever
pathway learns *last*, and the readout is the harmful one. Staging does not fix an
interaction; it decides who gets the last word.

**Pruning is nearly free**: the recurrent-only condition loses 22% of the connectome
for +0.010. That retires the last of E013's original story — the harm is not the loss
of synapses. It is not exploration either (t=0.32).

~~**Which makes this a consequence of H2d.** The pallium cannot represent distinctions,
`eta_out` grows a readout from that degenerate representation, and the cortical
pathway transmits structured state-dependent noise into an already-competent motor
system. Exploration noise is harmless because it is zero-mean; this is not.~~

**Withdrawn by [E019](experiments/E019-three-verified-defects.md).** The cortical
pathway does not transmit state-dependent anything. `ΔW_out` is **rank one** (top
singular value share 0.9981), and the cortical contribution to the motor drive varies by
**0.7% of its own magnitude** over three seconds of behaviour. It is a constant offset,
and it is negative on peck (+0.02 → −0.52) and on the call channels. The hen learns to
peck less; that is the whole of the measured harm.

Two further caveats on this node, both from E019:

- **E013's status as the clean test of H2 does not hold.** 98.1% of the reward variance
  in that run came from the vigour term — a cost added for a different hypothesis, moved
  out of the metric by E012 and into the teaching signal, where it was never checked.
  `REFUTED at this timescale` stands only as "refuted under a reward that was 98% call
  cost", which is not the claim the tree is making.
- **E015's superadditivity and E016's "last word" may both be artefacts** of a single
  rank-one offset being scaled two ways. Plausible, unverified, and not yet acted on.

---

### Status history (superseded; kept because the route matters)

Previously `UNDER TEST` — downgraded by E010, whose evidence was then found to be
**confounded**.

E010 reported the effect collapsing from t=3.93 to t=0.08 after the gain correction.
That run had also, silently, given every condition — *including the fixed control* —
the exploration noise added in E007 after E004 was run. It compared *(new gain +
noise)* against *(old gain, no noise)* and blamed the gain. See the invalidity notice
on [E010](experiments/E010-rebaseline-at-corrected-gain.md).

**[E012](experiments/E012-corrected-phase-1-contrast.md) isolated the real cause, and
it was neither the gain nor the noise.** With one variable changed at a time:

| gain | call cost | hunger change |
|---|---|---|
| 0.90 | 0 | +0.060 ← the E004 configuration |
| 0.70 | 0 | **+0.033** ← corrected gain is *better* |
| 0.70 | 8e-4 | +0.224 ← current |

The gain is nearly neutral. **`call_energy_cost` — added in E005 for H3 — explains
essentially all of the degradation**, because it triples the rate hunger accumulates
and so swamps the very metric H2 is measured on.

In the corrected 12-seed contrast, learning does not beat the fixed control
(+0.016 ± 0.040, t=0.40) and roughly cancels the cost of its own exploration
(noise-only: +0.018). But that contrast runs *with* the call cost, so its metric is
compromised too.

At that point no run had yet tested H2 cleanly. E013 did, by moving the call cost to
its own budget — and returned the significant negative result recorded at the top of
this section.

**Prediction:** a flock learning under a reward-prediction-error rule regulates its
drives better over a rearing run than a genome-matched, coop-matched flock that
cannot learn. Operationalised as hunger declining across a run relative to control.

**Experiment:** `run/experiment.py`, matched-seed contrast across three conditions
(fixed / learning without growth / learning with growth).

**Falsifier:** no reliable difference from the fixed control across seeds. That would
mean the rule is producing connectome churn without behavioural consequence — which
is a real possibility and the reason this is stated as a hypothesis rather than a
feature.

**Evidence — positive but underpowered.**
[E001](experiments/E001-does-plasticity-help.md) returned a null.
[E002](experiments/E002-can-the-pallium-reach-a-muscle.md) found why: the cortical
readout was frozen, so nothing the pallium learned could reach a muscle.
[E003](experiments/E003-does-the-fixed-readout-rescue-learning.md) reran E001 with
only that fixed, and the effect tripled — mean hunger now *falls* across a run
(0.321 → 0.295) where the fixed control *rises* (0.306 → 0.370), with feeding rate up
from 5.2% to 6.5% of timesteps.

**[E004](experiments/E004-replication-at-twelve-seeds.md) settles it at twelve seeds:
−0.063 ± 0.016 SE, t=3.93 against a 2.23 threshold, p≈0.002.** Both pre-registered
predictions held. The learning hen feeds on 7.4% of timesteps against the control's
6.3%.

The effect size shrank 30% from E003's four-seed estimate (−0.090 → −0.063), which
the pre-registration had flagged as a possibility — E003's estimate *was* inflated by
small-sample luck, and it cleared anyway because precision improved faster than the
estimate fell. E003 also caught a bug in the *analysis*: the harness had used a 2-SE
threshold, which at four seeds would have called p≈0.09 a result.

**What the null turned out to be.** The cortical readout was learning at 2e-3, at
which rate `|W_out|` grew 1.00x over ten minutes — frozen. E001 was measuring a hen
incapable of acting on anything she had learned. E002 also found a ceiling that was
not anticipated: at `eta_out=2e-1` cortical drive overwhelms the innate arc and
behaviour gets *worse*, because a hen who overrides her reflexes with an untrained
pallium is worse off than one who does not. The default is now 2e-2, near the
optimum.

**[E010](experiments/E010-rebaseline-at-corrected-gain.md) re-ran E004 unchanged
except for the gain, and the effect vanished:**

| | E004 (gain 0.9) | E010 (gain 0.70) |
|---|---|---|
| learning, no growth | −0.063 ± 0.016, **t=3.93** | −0.002 ± 0.028, **t=0.08** |
| fixed control, final hunger | 0.330 | **0.655** |

**Every hen also got worse**, control included — final hunger roughly doubled. The
change did not selectively remove learning; it degraded the whole flock and took the
signal with it.

**The mechanism is E002's finding in disguise.** At gain 0.9 the saturated pallium
emitted a near-constant output, so the cortical readout acted as a harmless fixed
*bias*. At 0.70 the pallium is responsive, so an *untrained* readout injects real
variability into the motor drive — and E002 already showed that untrained cortical
influence makes behaviour worse. Saturation had been accidentally protecting
behaviour.

So `readout_scale` and `eta_out`, both tuned by E002 **against a saturated network**,
are now stale. E010 is best read as "the readout parameters are wrong", not "learning
does not work" — learning still feeds more (6.1% vs 5.6%) and takes 21% less predator
exposure, neither significantly.

**Next: E011** re-runs E002's readout sweep at gain 0.70, then H2 is re-tested. Only
after that does a null mean anything about learning.

**Standing lesson**: parameters tuned under a defect inherit the defect. E002's sweep
was correct at the time and silently encoded an assumption about the operating point
that outlived it.

---

## H2a — does structural growth *hurt* learning?

**Status: NOT STARTED** — opened by E004, and it inverts the design's expectation.

Growth was the weaker condition in all three runs of the phase 1 contrast (E001
t=1.00 vs 0.78; E003 0.77 vs 2.50; E004 1.75 vs 3.93), and E004 attached a cost to
it: the no-growth hen reaches significance with **21,148 synapses**, having pruned
42% of her innate connectome, while the growth hen ends with **40,753** and does not
clear.

**Prediction:** the growth rule adds synapses on coactivity, which is orthogonal to
whether those synapses help, so continuous rewiring destabilises what has been
learned. Behavioural improvement should fall monotonically with growth rate above
some small optimum.

**Falsifier:** a growth-rate sweep showing a flat or rising relationship, which would
make the three-run ordering a coincidence of the on/off contrast.

**Note:** the default stays `growth_enabled=True` until this is settled. Three runs
of a binary contrast is not enough to overturn a default with a biological
justification behind it.

---

## H2c — a learned cue can recruit an innate response via top-down association

**Status: NOT STARTED** — opened after E007, and it supersedes the resolution E007
proposed.

E007 left the project at a fork: the innate and learned pathways sum into one motor
drive, so there is no setting where the pallium can both initiate new behaviour and
avoid overriding good reflexes. E007 proposed multiplicative gating of the reflex arc.

**That proposal does not work, and the reason is worth recording.** Gating scales an
existing reflex. In the case that matters — hearing an alarm call with *no hawk
visible* — the reflex input is zero, and any gain applied to zero is zero. Gating
cannot create a response to a cue the innate arc does not already respond to.

**The correction: wire the learned pathway to the sensory representation, not to the
motor output.** Let the pallium project back onto the observation the reflex arc
reads. Then hearing an alarm call does not have to *recreate crouching* — it only has
to recreate the *percept of a hawk*, and the innate reflex, which is already strong
enough, does the rest.

This is Pavlovian conditioning stated architecturally: the conditioned stimulus comes
to activate the representation the unconditioned stimulus would. It is also why the
magnitudes suddenly work. Supplying crouch drive directly needs +2.50 against a
measured cortical capacity of 0.002. Supplying it *through* the aerial channel needs
only ~0.3 in sensory units, because the innate reflex multiplies it by 8.0.

**Where the association comes from, and why it needs no reward.** A hen who is
head-up both sees the hawk and hears the flockmate's call — the two co-occur, and a
plain Hebbian rule associates them. Later, head-down and blind, the call alone
reconstructs the percept. She learns what the call means during the moments she can
check for herself, and can then use it when she cannot.

That resolves the credit-assignment problem that sank E005 and E006 as well, because
association needs no reward signal and no attribution of benefit to a caller.

**Prediction:** comprehension — crouching to a played-back alarm with no predator
visible — rises above zero with a top-down associative projection and stays at zero
without it. Measured on the existing assay, which E006 showed is sound.

**Falsifiers:** comprehension flat with the projection added; or comprehension
appearing but the hen now hallucinating percepts with no cue, which would mean the
projection is unconstrained rather than associative.

**Risk to watch:** a pathway that writes into a hen's own senses can make her
perceive things that are not there. That is the intended mechanism and also the
obvious failure mode, so the assay must check the no-cue baseline, not just the
cued response.

**[E008](experiments/E008-top-down-association.md): implemented, and the first
attempt does not test the hypothesis.** The projection learns — `|W_pred|` saturates
its cap — and comprehension stays at zero. The diagnosis is that the rule as written
maps `rate(t) -> obs(t)`, which is an **autoencoder**, not a predictor. During a hawk
event the brain state is dominated by the hawk percept, so the association it forms is
"when in hawk-state, predict hawk" — circular, and no use for recovering the percept
from a call. Pavlovian association needs `rate(t - delta) -> obs(t)`, mapping a cue to
a later outcome.

Also thin: a 30-minute rearing yields roughly **9 seconds** of usable co-occurrence
(hawks present ~24 s, of which only the head-up fraction counts). E009 must raise
predator density so that a null means the mechanism is wrong rather than the data
absent.

The no-cue baseline did not rise, so the hallucination failure mode has not occurred.

**[E009](experiments/E009-lagged-pallial-association.md): fixed both, still null — and
found the actual blocker.** With a lagged pallial source and up to 90x predator
density, comprehension stayed at zero while baseline crouching with *no cue* nearly
doubled (0.078 → 0.147). The projection learns the **base rate** of aerial threat
rather than a contingency on the call. Measuring the representation says why:

```
                mean|rate|   shift: call     shift: hawk
sensory stub      0.4218     0.0949 (22.5%)  0.0819 (19.4%)
pallium           0.8577     0.0277 ( 3.2%)  0.0235 ( 2.7%)
```

The call reaches the brain with *more* afferent weight than the hawk channel. But the
pallium sits at mean rate 0.86 — deep in the flat part of the sigmoid — and the two
percepts differ from each other there by under 1% of the mean. **No associative rule
can be cue-specific when sourced from a representation that does not distinguish the
cues**, and with nothing to condition on, the base rate is the best available
prediction. See H2d.

---

## H2d — the pallium does not form separable representations of distinct stimuli

**Status: SUPPORTED as a limitation** — measured in
[E009](experiments/E009-lagged-pallial-association.md). **This is the blocker behind
H2b, H2c and everything downstream.**

Hearing an alarm call and seeing a hawk drive pallial states that differ by 0.008 —
under 1% of the mean rate. The network runs saturated (mean pallial rate 0.83, where
the sigmoid slope is ~0.12) ~~and the two stimuli project onto overlapping random
subsets of the sensory stub with nothing downstream to decorrelate them~~.

A gain sweep shows saturation is real but not the whole story: dropping the recurrent
gain from 0.9 to 0.6 moves the mean rate to 0.21 and improves *relative* separability
fourfold, but absolute separability barely moves and collapses below 0.6.

**Mechanism corrected by [E017](experiments/E017-where-separability-is-lost.md).** The
struck clause above was inferred rather than measured, and it is false. The sensory
stub separates the two percepts *cleanly* — relative separability 1.055, sharing only
9% of target units, cosine 0.245. They are close to orthogonal when they arrive.

The 17× loss is the single sensory → pallium projection, and it is **not** recurrent
mixing: zeroing pallial recurrence makes separability slightly worse (0.79×), not
better. What is left is dilution by fan-in — each pallial unit sums ~19 stub inputs of
which one or two carry the distinction, so a clean difference lands as a small
perturbation on a large common-mode drive.

Segregating auditory afferents onto their own pallial target, as a real bird has (Field
L via nucleus ovoidalis, separate from the entopallium via rotundus), recovers 2.06×.
Real, in the right direction, and not sufficient: 2× against a 17× loss.

**A separate finding from E017 that may matter more than any of the above.** The innate
arc has *no* response to hearing a call — every auditory entry in `reflex_matrix()` is
zero, against a weight of 8.0 for crouch on seeing a hawk. Comprehension being learned
in real chickens does not mean it is learned from nothing; naive chicks respond
differentially to conspecific fear calls at hatch, and the learned part is association
off an already-arousing stimulus. So this node may be diagnosing a representation
problem where there is also a missing scaffold.

**E002, E007, E008 and E009 were all attempts to fix the routing of learning.** The
blocker is upstream of routing: there is not enough in the pallial state to route.

**Caveat this raises for an existing result.** H2's supported finding (E004, t=3.93)
was obtained in the saturated regime. Drive regulation evidently only needs coarse
modulation, which a saturated network can still supply — but it should be re-run once
the operating point is fixed, and may well get stronger.

~~**Not yet changed.** The gain default stays 0.9~~ — **stale**. The re-baselining
happened; `hen/connectome.py:48` has `gain = 0.70`. Flagged by external review as a doc
that describes a state the code left behind.

**H2d is demoted from the critical path by
[E019](experiments/E019-three-verified-defects.md), pending re-measurement.** Its whole
diagnosis rests on contrasting "saw a hawk" against "heard an alarm call". In the actual
coop at the default flock size, the call channel is **constant at 1.0** and the aerial
channel averages 0.00 — the contrast this node is built on never occurs. Separability was
measured on hand-injected observations describing a situation the hen never experiences.

Whether the pallium can represent distinctions is still an open and important question.
It is not established that it cannot, and it is no longer the thing blocking everything
else.

---

## H3 — learned usage reproduces the audience effect without being programmed

**Status: UNDER TEST** (phase 2/3) — two nulls recorded, and **blocked by H2b**.

Real cockerels alarm- and food-call far more readily with a hen present, graded by
audience type (conspecific > other species > empty cage). Call *production* is innate
in chickens and is wired in; *usage* is learned and deliberately is not.

**Prediction:** after rearing in a social coop, calling rate becomes conditional on a
flockmate being within earshot, without any term rewarding that.

**Why it matters:** this is the model predicting a documented behaviour it was never
told about. It is the strongest available validation that the learning rule is
capturing something real, and it is cheap — the behaviour is already measurable.

**Falsifier:** calling rate independent of audience after rearing.

**Evidence — [E005](experiments/E005-does-the-audience-effect-emerge.md): null.** The
food-call effect rose in the mean (+0.032) but the between-seed spread swamps it
(t=0.64). The alarm-call effect went *against* prediction (−0.031, t=1.43) and did so
more consistently than the food effect went with it. The assay itself is sound: the
fixed control is flat to three decimals in every cell.

**Leading suspect — structural, same class as E002.** The kin term adds
`kin_weight × mean(other hens' reward)`, which is nearly identical for every hen in
the flock. For a hen to learn that *her call helped*, the modulator must move
*because she called*; a flock average moves the same way for the silent bird beside
her. So the reward gives the flock a reason for calls to exist and gives no
individual a way to discover she is responsible for one — leaving the energy **cost**
as the only component correlated with her own calling, which predicts suppression,
which is the sign observed.

**[E006](experiments/E006-audibility-weighted-kin-reward.md): second null, and it
found the real blocker.** Audibility-weighting the kin term did not rescue the effect
(food t=0.90 vs flat-kin's t=0.64 — indistinguishable). But the comprehension assay
added in the same run returned the decisive number: **hearing an alarm call changes a
hen's behaviour by ~0.0005, before and after rearing, in every condition.** Nothing
responds to calls, so a call cannot help anyone, so it cannot repay its cost — leaving
the private energy cost as the only thing correlated with her own calling, which
predicts suppression, which is the sign seen on the alarm channel in both experiments.

**H3 is blocked by H2b, not by its own design.** See below.

---

## H2b — the learning rule cannot acquire behaviours outside the innate repertoire

**Status: SUPPORTED as a limitation** — established by
[E006](experiments/E006-audibility-weighted-kin-reward.md), refined by
[E007](experiments/E007-exploration-does-not-rescue-comprehension.md). **This blocks
H3, H4 and H5.**

The three-factor rule strengthens synapses that were active when the modulator moved.
It has no way to reinforce an action that never happened — and the model is
deterministic. No motor noise, no stochastic action selection, no exploration of any
kind.

So every behaviour a hen will ever perform is one the innate reflex arc already
produces. Learning can re-weight *when* existing behaviours fire; it cannot acquire a
stimulus-response pairing the reflex arc never visits. To learn "crouch when you hear
an alarm", she must at some point crouch on hearing one — and she never does, because
crouching is driven by *seeing* a hawk and `hen/innate.py` deliberately wires nothing
from the auditory channels.

**Why this was invisible until now.** H2 only required tuning behaviours the hen
already performs — approach food, peck, huddle. The limit binds the moment the
question is about acquiring something genuinely new, which is every hypothesis from
here on.

**[E007](experiments/E007-exploration-does-not-rescue-comprehension.md) refines this
and the refinement matters.** Exploration was added and comprehension did not move —
nor did it with 30x the predator density, nor with both. The measurement that
explains it:

```
drive needed for crouch > 0.5     : +2.50
innate reflex, hawk overhead      : +8.00
cortical drive to crouch, playback: +0.002   -- roughly 170x short
```

**The real limit is that the learned pathway can modulate behaviour but cannot
initiate it.** Exploration cannot help, because the problem is not that the action is
never sampled — it is that once sampled, the learned pathway has no way to reproduce
it. There is nothing for the credit to attach to.

**This is the same tension E002 found, seen from the other side.** E002: cortical
influence too high and behaviour degrades, because an untrained pallium overrides good
reflexes. E007: cortical influence low enough to be safe, and nothing new can be
learned. The two pathways sum into one motor drive and compete on a single axis, so
there is no setting that satisfies both.

**Open architectural decision** — additive competition versus multiplicative gating.
If the cortical pathway set a per-channel *gain on the reflex arc* instead of adding
its own drive, a learned signal could recruit an innate behaviour without generating
it from nothing, and could not run away because it only scales existing responses.
That is closer to how learned modulation of tectal circuits is thought to work. It is
a change to the core architecture and has not been made.

---

## H4 — an intact channel beats a shuffled one on a task requiring private information

**Status: NOT STARTED** (phase 4) — **this is the headline**

**Prediction:** on a task where information is private, costly to acquire alone, and
changes at an intermediate rate, a flock with an intact channel outperforms a
capacity-matched flock whose channel is shuffled between hens.

**Design:** the six-way condition ladder in `docs/backlog.md`. The headline contrast
is **L vs C?** (shuffled), not L vs a smaller brain — a reduced-capacity control
confounds capacity with language and cannot answer this.

**Falsifiers:** L ≈ C? at every capacity; or muting the channel in a *trained* L
flock costs nothing; or C− (extra capacity, no channel) ≈ L, meaning the neurons did
the work.

---

## H5 — compositional structure requires a transmission bottleneck

**Status: NOT STARTED** (phase 4+)

**Prediction:** with generational turnover — naive chicks acquiring the code from a
limited sample of incumbents — emergent signals become compositional (rising
topographic similarity). Without turnover, they stay holistic: one arbitrary symbol
per situation, no reusable parts.

**Counter-prediction worth stating up front:** raw capacity should *increase* channel
use while *decreasing* compositionality, because a large enough network can memorise
an arbitrary mapping instead of building a structured one. If both rise together,
something we do not understand is happening and it is worth stopping to find out.

---

## H6 — a hen's internal state is recoverable from her signals

**Status: NOT STARTED** (phase 4+)

**Prediction:** a post-hoc translator trained on (signal, ground-truth state) pairs
recovers internal state above chance, and its accuracy rises over generations.

**Caveat, which is not optional:** translator output is a learned mapping from a
scalar, not a report. See `docs/ethics.md` §6.

---

## Changelog

| date | change |
|---|---|
| phase 0 | H1, H1a recorded as `SUPPORTED`; tree established |
| phase 1 | H2 opened, experiment harness built |
| E001 | H2 null recorded. Churn without behavioural consequence. |
| E002 | Diagnosed the null: readout frozen. `eta_out` 2e-3 → 2e-2. |
| E003 | Effect appeared (t=2.50), short of threshold. Analysis bug found and fixed. |
| E004 | H2 → `SUPPORTED` (t=3.93, p≈0.002). H2a opened. Predator exposure retired. |
| E005 | H3 null. Kin reward cannot assign credit to the caller; E006 opened. |
| E006 | H3 null again. Comprehension is zero: no exploration. H2b opened, blocking H3-H5. |
| E007 | Exploration added; comprehension still zero. H2b refined: the learned pathway cannot *initiate*, only modulate. Architectural fork opened. |
| E008 | Top-down association built; first version was an autoencoder, tested nothing. |
| E009 | Lagged/pallial association, still null. Found the pallium saturated: H2d opened. |
| E010 | Gain 0.9 -> 0.70; H2 appears to collapse. **Later found confounded — invalid.** |
| E011 | Readout sweep. Control did not improve as predicted; that tell exposed E010's confound. |
| E012 | Isolated it: not the gain, not the noise -- **`call_energy_cost` from E005 swamps H2's metric.** |
| E013 | Call cost moved to its own `vigour` budget. **Clean test: learning is significantly WORSE. H2 refuted at this timescale.** |
| E014 | Units bug found: a strike contributed -100 to reward. Connectome recovers; **behaviour does not**. Harm localised to the learned readout, implicating H2d. |
| E015 | Harm decomposed: readout +0.021, recurrent +0.010, both **+0.052 — superadditive**. Pruning nearly free. |
| E016 | Staging tested. Prediction falsified: pallium-first does nothing, **muscles-first cuts harm 69%**. Moving-target story withdrawn. |
| E017 | H2d's mechanism corrected: the stub separates cleanly (1.055), the loss is fan-in at sensory→pallium, not recurrence. Found the innate arc has **no auditory reflex at all**. |
| E018 | **ABORTED mid-run.** The channel it tests carries no information at n=16. Its pre-registered falsifier would have fired for the wrong reason. |
| E019 | External review, verified here. **Calls are inaudible** (a full alarm moves the receiver by 0.0000), **`W_out` is rank one** (0.7% variability), **reward is 98% call cost**. Withdraws H2's mechanism, caveats E013, demotes H2d. |
