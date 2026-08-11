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

1. ~~**Phase 1 plasticity**~~ — **done.** Learning has a demonstrated behavioural
   effect ([E004](experiments/E004-replication-at-twelve-seeds.md): t=3.93, p≈0.002).
   This was the prerequisite for everything below it.
2. Condition harness: the six-way ladder as a config, sharing one simulation path.
   `run/experiment.py` has the matched-seed skeleton; it currently knows three
   conditions, not six.
3. T1 shared vigilance — validates the harness against a known biological effect.
4. Rotation-period sweep to locate the social-learning band.
5. T2 poisoned feeder — the headline experiment.
6. Playback and lesion assays.
7. T3 safe corridor.

## 8. Open items from experiments

- ~~**Does the cortical pathway ever influence behaviour?**~~ **Answered by
  [E002](experiments/E002-can-the-pallium-reach-a-muscle.md):** it does, but only
  once the readout can learn. At the E001 setting `|W_out|` grew 1.00x — frozen.
  `eta_out` raised 2e-3 → 2e-2.
- ~~**Is synaptic scaling cancelling the learning signal?**~~ Moot — E003/E004 found
  the effect once the readout could learn, so scaling was not the blocker.
- **H2a: does structural growth hurt learning?** Weaker in all three runs of the
  contrast, and E004 attached a cost: no-growth reaches significance with 21,148
  synapses (pruning 42% of the innate connectome) while growth ends with 40,753 and
  does not clear. Needs a growth-*rate* sweep, not another on/off contrast.
- **The innate/learned control balance is a parameter worth studying, not tuning.**
  E002 found that too much cortical influence makes behaviour *worse* — an untrained
  pallium overriding good reflexes. There is an optimum, and it plausibly maps onto
  the trade-off real precocial birds face between hatching competent and staying
  plastic. Worth a proper sweep rather than a single tuned value.
- **A longer, better-powered contrast**: 1 day of chicken time with 8 seeds,
  pre-registered on both hunger and predator exposure. ~4 h wall clock. Only worth
  running once E003 shows the effect exists at all.
- **Predator exposure as a metric.** E001 showed a large but unpowered difference
  (1755 vs 3075 exposure-steps). It may be a better-powered readout than hunger,
  since it is driven by discrete costly events rather than a slow scalar. Worth
  promoting to primary if it survives a powered test.

---

## Sources

- [Many-eyes effect and the vigilance/foraging trade-off](https://royalsocietypublishing.org/doi/10.1098/rsos.150135)
- [Evolutionarily stable vigilance as a function of group size](https://www.sciencedirect.com/science/article/abs/pii/S0003347205810231)
- [Conditions that favour cumulative cultural evolution](https://royalsocietypublishing.org/rstb/article/378/1872/20210400/109147/Conditions-that-favour-cumulative-cultural)
- [Measuring non-trivial compositionality in emergent communication](https://www.researchgate.net/publication/344945098_Measuring_non-trivial_compositionality_in_emergent_communication)
- [Structural inductive biases in emergent communication](https://arxiv.org/pdf/2002.01335)
- [On the meaning of alarm calls: functional reference in an avian vocal system](https://www.sciencedirect.com/science/article/abs/pii/S0003347283711589)
- [Audience effects on alarm calling in chickens](https://pubmed.ncbi.nlm.nih.gov/3396311/)
