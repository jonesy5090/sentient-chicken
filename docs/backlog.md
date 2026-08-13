# Backlog: the language experiment

The hypothesis: **a flock with a productive communication channel can reach a goal
that an equivalent flock without one cannot.**

This document specifies how to test that without fooling ourselves. It is design
only -- nothing here is built yet.

---

## 1. The control has to be capacity-matched

The obvious control is a flock with reduced neural capacity that behaves like a
normal hen. That is a useful *reference point*, but it is not the control that tests
the hypothesis, because it varies two things at once: capacity and language. If the
small-brained flock fails, we cannot say whether it failed for want of language or
for want of neurons -- and since the extra capacity was added specifically to support
language, that confound is not a technicality. It is the whole question.

The fix is a ladder in which only one thing changes at a time.

| condition | capacity | channel | isolates |
|---|---|---|---|
| **N** natural | default | innate calls only | the "normal hen" reference |
| **C−** capacity control | expanded | none | did the extra neurons alone do it? |
| **C0** severed | expanded | emits, nobody hears | the motor cost of calling |
| **C?** shuffled | expanded | hears a *random* flockmate | **the real control** |
| **Cs** self-only | expanded | hears only herself | channel as private memory |
| **L** language | expanded | intact | the hypothesis |

**The headline comparison is L vs C?, not L vs N.** A shuffled channel delivers the
same bandwidth, the same signal statistics and the same energetic cost, carrying zero
information about the world. If L beats C?, information transfer is doing the work.
If L ≈ C?, the channel is decorative however impressive the transcripts look.

**Cs** matters more than it appears. A channel can help a *single* agent by acting as
external memory -- writing a symbol now and reading it back later -- which is not
communication at all. Without Cs in the ladder, that failure mode is invisible.

Keep **N** in the design anyway. It answers a different and genuinely interesting
question: does the language flock beat a naturalistic bird, or merely beat a
lobotomised one?

---

## 2. Don't design for a wall

The instinct is a task that is impossible without language. A hard binary is worth
avoiding: if the control scores zero, we learn only that the task was impossible, not
how much language helped or at what capacity it started paying off.

A graded task with a large, clean effect is strictly more informative, because it
yields a **dose-response curve** across the capacity ladder. Run every condition at
several pallium sizes (say 0.5x, 1x, 2x, 4x the default 256 units). The interesting
result is not "language wins" but *where the curve bends* -- the capacity at which
the channel starts carrying real information.

Predict two things that may come apart, per the emergent-communication literature:
more capacity should increase channel **use**, while *reducing* structural
**compositionality**, because a large enough network can memorise an arbitrary
holistic code instead of building a compositional one. If both rise together,
something interesting is happening.

---

## 3. Task candidates

The requirement is information that is **private, useful, costly to acquire
individually, and cannot be resolved by looking.**

The critical parameter is how fast the answer changes. Social learning pays only at
*intermediate* rates of environmental change: too stable and every hen just learns it
herself, too fast and nothing helps anyone. **The answer must change faster than an
individual can learn it, but slower than the flock can propagate it.** Finding that
band is a sweep, and finding it is itself a result.

### T1 — Shared vigilance (build first)

Nearly free: the head-down gate, hawks, and innate alarm calls are already built. A
hen who forages cannot watch the sky. With a working channel the flock can divide
vigilance -- one bird watches, the rest feed.

- **Metric**: food intake at matched predation risk (compare Pareto frontiers, not
  raw intake -- a flock that never forages is safe and starving).
- **Grounding**: this is the many-eyes effect, well documented in real flocks.
- **Prediction**: L forages more than C? at equal risk, and per-hen vigilance falls
  as flock size rises.
- **Caveat**: alarm calls are already innate here, so T1 tests *usage*, not
  invention. That is exactly the phase 2/3 target, which makes it the right warm-up
  and the right way to validate the harness — but it is not the headline experiment.

### T2 — The rotating poisoned feeder (the headline experiment)

Several feeders; one is contaminated; which one rotates on a period tuned to the band
above. Eating from it is costly but survivable. Contamination is invisible until
tasted, so the information is genuinely private and genuinely transferable.

- **Metric**: total flock sickness per rotation period.
- **Prediction**: L converges toward one hen's mistake per rotation (the discoverer
  pays, everyone else is warned); C? pays roughly N times that.
- **Why this one**: it tests *reference* -- the signal must carry **which** feeder,
  not merely that something is wrong. And it extends a behaviour chickens genuinely
  have, since their food calls are already functionally referential.

### T3 — The safe corridor (stretch)

Food beyond a region where some ground is dangerous; the safe route changes
periodically. A hen who survives the crossing knows something no one else can see.

- **Tests**: compositionality (direction *and* distance) and **displacement** --
  reference to a place not currently visible, which is the real Rubicon and the thing
  most likely *not* to emerge without explicit pressure.
- Hardest to build and to interpret. Do it last, and treat a negative result as a
  finding rather than a failure.

### Rejected: time-of-day gating

A feeder that opens on a schedule is individually learnable -- every hen can discover
it alone, at low cost, and then it is hers forever. No information asymmetry survives
past the first day, so there is nothing to communicate. Worth stating explicitly so
it does not get proposed again.

---

## 4. Generational turnover is not optional

Compositional structure needs a transmission bottleneck: each generation acquiring
the code from a limited sample of the last. Without it the expected outcome is a
holistic code -- one arbitrary symbol per situation, no reusable parts.

So the protocol needs periodic replacement of a fraction of the flock with naive
chicks who must learn the code from incumbents. This is both the mechanism that
drives structure and, conveniently, just chickens breeding.

---

## 5. Measurement

**Primary**: task performance per condition per capacity.

**Causal efficacy — the one that actually proves it.** Take a *trained* L flock and
mute the channel at test time. The drop in performance is the amount of work language
was doing. Correlational evidence that hens call a lot is not evidence that calling
matters.

**Channel content**: mutual information between emitted symbols and world state
(which feeder, which predator class, where).

**Comprehension via playback**: present a recorded call to a hen with no
corresponding stimulus and measure the behavioural response. This is precisely the
protocol Evans & Marler used on real fowl, which makes the results directly
comparable to the animal literature — a rare opportunity and worth designing for.

**Compositionality**: topographic similarity (Spearman correlation between distances
in world-state space and Levenshtein distances between messages), plus positional and
bag-of-symbols disentanglement.

**Legibility**: a post-hoc translator mapping symbols to English by correlating
against ground-truth state. Do not constrain the language to be readable — let it
evolve, then translate. Translator accuracy over generations is itself a metric: it
measures how much of a hen's internal state is recoverable from her calls.

---

## 6. What would falsify the hypothesis

Worth writing down now, while it is still cheap to be honest:

- **L ≈ C?** at every capacity — the channel carries no usable information.
- **Muting a trained L flock costs nothing** — the channel was correlated with
  behaviour but not driving it.
- **C− ≈ L** — the extra neurons did the work and language was incidental.
- **No rotation period exists where social learning beats individual learning** — the
  task does not actually require communication, and needs redesigning.

---

## 7. Order of work

1. **Phase 1 plasticity — REOPENED, and it blocks everything below.**
   [E013](experiments/E013-clean-test-of-h2.md) is the first clean test and learning
   comes out *significantly worse* than no learning (t=3.85). E004's positive result is
   best explained as an artefact of the saturated regime.

   The reason is **not** connectome damage, though E013 said so at the time: that was a
   units bug, fixed in [E014](experiments/E014-strike-units-bug.md), and
   [E015](experiments/E015-decomposing-the-harm.md) measured pruning as nearly free
   (22% of the connectome for +0.010). The harm is the learned pathways themselves,
   superadditively. Nothing below is worth running until a rule exists that does not
   make the bird worse.
2. Condition harness: the six-way ladder as a config, sharing one simulation path.
   `run/experiment.py` has the matched-seed skeleton; it currently knows three
   conditions, not six.
3. T1 shared vigilance — validates the harness against a known biological effect.
4. Rotation-period sweep to locate the social-learning band.
5. T2 poisoned feeder — the headline experiment.
6. Playback and lesion assays.
7. T3 safe corridor.

## 7a. Blocking everything — the three E019 defects

These come before every item in §7 and every item in §8. All three are verified
([E019](experiments/E019-three-verified-defects.md)) and all three are small fixes.

1. **Make calls audible.** `coop/world.py:192` emits the raw sigmoid, so every hen calls
   continuously at the resting floor of 0.076, and `coop/sensing.py:77` sums that across
   15 flockmates into a clip. At the default flock size the channel sits at 1.0 and a
   full-amplitude alarm from an adjacent bird moves it by **0.0000**. Threshold or
   floor-subtract at emission, and combine audibility non-linearly (max, or an
   energy-sum with log compression) instead of by linear summation. **Until this is
   done, E018, H2b, H2c, H3 and the entire H4 headline are measuring a constant** — and
   L versus C? (shuffled channel) is a guaranteed null, since shuffling a constant gives
   the same constant. Add a test at `n_hens=16`; the suite uses 4, which is the one band
   where the channel still works.

2. **Give `W_out` more than one degree of freedom.** `ΔW_out` is rank one and the
   cortical drive varies by 0.7% of its magnitude. A three-factor outer product of two
   non-negative slow traces cannot express a state-dependent policy — it can only slide
   a constant. Everything H2 through H5 wants from the learned pathway needs this.

3. **Take the vigour term out of `reward()`.** 98.1% of reward variance is the call
   cost. Keep it in the world so calls still attenuate; remove it from the teaching
   signal. Then re-run the E013 contrast — it decides whether `H2 REFUTED` survives.

**And the meta-item.** All three are quantities that were checked in the place they had
just been moved *from*. When a term is relocated, measure it in its new home. Worth a
line in `CLAUDE.md` under design invariants.

## 8. Open items from experiments

- ~~**Does the cortical pathway ever influence behaviour?**~~ **Answered by
  [E002](experiments/E002-can-the-pallium-reach-a-muscle.md):** it does, but only
  once the readout can learn. At the E001 setting `|W_out|` grew 1.00x — frozen.
  `eta_out` raised 2e-3 → 2e-2.
- ~~**Is synaptic scaling cancelling the learning signal?**~~ Moot — E003/E004 found
  the effect once the readout could learn, so scaling was not the blocker.
- **H2a: does structural growth hurt learning?** Weaker in *all five* runs that have
  compared them, now including E013 (+0.077 vs +0.062, both significantly worse than
  no learning). Growth has never helped in any regime. Largely subsumed by the
  erosion question above.
- **The innate/learned control balance is a parameter worth studying, not tuning.**
  E002 found that too much cortical influence makes behaviour *worse* — an untrained
  pallium overriding good reflexes. There is an optimum, and it plausibly maps onto
  the trade-off real precocial birds face between hatching competent and staying
  plastic. Worth a proper sweep rather than a single tuned value.
- ~~**Why does the rule erode?**~~ **Answered (E014):** a units error made one
  predator strike worth −100 in reward. Fixed; the connectome recovers. It was **not**
  the cause of the behavioural harm, and pruning turns out to be nearly free (E015:
  22% of the connectome lost for +0.010).
- ~~**Is the superadditivity a moving-target problem?**~~ **No (E016).** Staging was
  tested and the prediction was falsified: letting the pallium settle first does
  nothing (t=0.25), while doing the readout first cuts harm 69% (+0.052 → +0.016,
  significant). The moving-target story is withdrawn.
- **Does the pathway that learns *last* dominate the harm?** (E016) That is the new
  reading, and it makes a sharp cheap test: append a short pallium-only stage to an
  otherwise simultaneous schedule and see whether it absorbs the harm regardless of
  what came before. Worth one run — but note this is a **workaround**, not a fix.
  Even at its best, staging leaves the hen worse than not learning at all.
- **Fix the representation (H2d).** Both pathways learn from or into a pallium whose
  states for "heard an alarm" and "saw a hawk" differ by ~6% of mean rate. The gain
  correction helped 8x and was not enough. ~~Likely needs something that decorrelates
  the sensory projections rather than the random overlapping ones we have.~~
  **[E017](experiments/E017-where-separability-is-lost.md) relocated the problem:** the
  sensory projections are already near-orthogonal (separability 1.055 at the stub,
  9% shared units). The loss is fan-in dilution at sensory → pallium, and it is not
  recurrence — zeroing recurrence makes it slightly worse. **Still the critical path**
  — H2, H2b, H2c and H3 all trace back to it — but the fix is a projection problem,
  not a decorrelation problem.

- **Wire an innate auditory reflex arc.** (E017) `hen/innate.py` has *no* response to
  hearing any call: every auditory entry in `reflex_matrix()` is zero, against 8.0 for
  crouch on seeing a hawk. That was a deliberate reading of "comprehension is learned",
  and it over-read the biology — parentally naive chicks already respond differentially
  to conspecific fear calls, and the learned part is association off a stimulus that is
  already arousing, not discovery from scratch. Propose: weak crouch on hearing an
  aerial alarm, weak flee/vigilance on a ground alarm, both well below the visual
  weights so they scaffold rather than solve. **This is also the cleanest available fix
  for the E006/E007 exploration null** — she cannot learn to crouch at a call she has
  never once crouched at. Needs its own hypothesis node and a falsifier that
  distinguishes "the scaffold works" from "we wired in the answer".

- **Modality-segregated afferents.** (E017) Audition currently shares the sensory stub
  and its pallial targets with vision. Real birds keep them apart — Field L via nucleus
  ovoidalis, entopallium via rotundus, two separate thalamic relays. A hand-cut
  segregation measured **2.06x** separability. Cheap, biologically motivated, and not
  sufficient on its own (2x against a 17x loss). Should be done via the connectivity
  prior in `regions.py`, not a slice.

- **Is the learning rule the wrong *kind*?** (E017, open — no node yet.) `plasticity.py`
  is reward-modulated three-factor, i.e. instrumental conditioning: act, get rewarded,
  strengthen. The mechanism the biology points at for alarm-call comprehension is
  Pavlovian — Curio's mobbing work has naive birds acquiring enemy recognition by
  *observation*, with no reward and no action of their own, transmitted along a chain
  of six individuals. `W_pred` is much closer to that machinery than `W_out` is. This
  may be why every attempt to route learning (E002, E007, E008, E009) has failed:
  right routing, wrong learning rule. Needs a hypothesis node before anything is built.
- ~~**Does the learning effect grow over a realistic rearing?**~~ Moot until a
  non-destructive rule exists. Running longer with the current rule strips more of
  the connectome, not less.
- ~~**Predator exposure as a metric.**~~ Retired. Uninformative in both E003 and E004
  (SEs of 1460-2469 on means of 13-44). E001's apparent 43% difference was noise, as
  it was flagged at the time.

---

## Sources

- [Many-eyes effect and the vigilance/foraging trade-off](https://royalsocietypublishing.org/doi/10.1098/rsos.150135)
- [Evolutionarily stable vigilance as a function of group size](https://www.sciencedirect.com/science/article/abs/pii/S0003347205810231)
- [Conditions that favour cumulative cultural evolution](https://royalsocietypublishing.org/rstb/article/378/1872/20210400/109147/Conditions-that-favour-cumulative-cultural)
- [Measuring non-trivial compositionality in emergent communication](https://www.researchgate.net/publication/344945098_Measuring_non-trivial_compositionality_in_emergent_communication)
- [Structural inductive biases in emergent communication](https://arxiv.org/pdf/2002.01335)
- [On the meaning of alarm calls: functional reference in an avian vocal system](https://www.sciencedirect.com/science/article/abs/pii/S0003347283711589)
- [Audience effects on alarm calling in chickens](https://pubmed.ncbi.nlm.nih.gov/3396311/)
