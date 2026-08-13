# E024 — H4: does an intact channel beat a shuffled one? (no plasticity)

> **Pre-registration.** Sections 1–5 written and committed before the run.
> **This is the experiment the project exists to run.**

## 1. Parent hypothesis

**H4** — an intact channel beats a shuffled one on a task requiring private
information. `NOT STARTED` since the tree was written. The headline.

## 2. Why this can run now, when it has been blocked for 21 experiments

`docs/backlog.md` §7 said phase 1 plasticity "blocks everything below". That was
inferred and never checked, and for H4 it is **false**: H4's prediction mentions no
learning at all.

Everything H4 needs is innate and already built:

- **Production** — innate, hardwired in `hen/innate.py`, 7/7 ethogram assays.
- **Audibility** — fixed in E019; a flockmate's alarm moves the receiver +0.908.
- **Comprehension** — innate via the E018 scaffold, measured at 0.189.
- **The asymmetry** — the head-down gate, H1a, `SUPPORTED`.

So the project has been holding its headline experiment behind a dependency that does
not exist, while running twenty-one experiments on a learning rule that has been a null
since E001.

## 3. Task and design

**T1, shared vigilance** (`docs/backlog.md` §3). A foraging hen has her beak down and
cannot see a hawk. With an informative channel the flock divides labour: someone watches
while the rest feed. This is the many-eyes effect, documented in real flocks.

**The six-way ladder**, no plasticity and no exploration noise in any condition:

| | capacity | channel | isolates |
|---|---|---|---|
| **N** natural | 1.0× | intact, **no scaffold** | the normal-hen reference |
| **C−** capacity | 1.5× | none | did the extra neurons do it? |
| **C0** severed | 1.5× | emits, nobody hears | the motor cost of calling |
| **C?** shuffled | 1.5× | **hears a random flockmate** | **the control** |
| **Cs** self-only | 1.5× | hears only herself | channel as private memory |
| **L** language | 1.5× | intact | the hypothesis |

**Capacity scales the pallium only** (`Regions.with_pallium`), not the whole brain.
Scaling everything would change the sensory and motor interface widths too — two more
things at once, in a design whose entire purpose is to vary one.

**The shuffle is re-drawn every 10 s.** A fixed permutation is a stable mapping and a
stable mapping is in principle invertible; re-drawing keeps the channel genuinely
uninformative rather than scrambled once. Verified: shuffled delivers flock-mean audio
**0.4400** against intact's **0.4399** — same bandwidth, same cost, different sender.

**Hawk period 60 s, not the default 900 s.** At 900 s a 20-minute run gets ~1.3 passes
totalling ~16 s, so the entire H4 signal would rest on 1.3% of the run and whether a hawk
happened to land near the flock. This is a **stated departure from biology** — real
hawks are rarer — made because the alternative is an unmeasurable experiment. It should
be swept later to check the result is not an artefact of predation pressure.

## 4. Metric — a trade-off, not a scalar

**Intake and risk are reported together and compared as a pair.** A flock that never
forages is safe and starving; one that never looks up is fed and eaten. Reporting intake
alone scores starvation as success.

- **Primary:** `fed %` and `struck/hen`, both L vs C?, paired across matched seeds.
- **Secondary:** mean hunger, head-down fraction.
- **Manipulation check:** mean alarm-channel input — must be ~0 for C− and C0, non-zero
  elsewhere, and near-identical between L and C?.

**8 matched seeds**, two-tailed t at 7 df (2.365). Everything is deterministic here — no
plasticity, no noise — so the only variance is between coops.

## 5. Predictions

1. **L beats C? on strikes**, by 15–40%, significant. Mechanism: an intact alarm makes a
   head-down hen look up and crouch *when there is actually a hawk*; a shuffled alarm
   makes her do it at random.
2. **L ≥ C? on fed %**, direction predicted but magnitude uncertain. Both flocks lose
   foraging time to alarms; L loses it when it matters. Could be small.
3. **C? ≈ C0 on strikes.** A channel carrying no information should be worth no more than
   no channel — and might be worth *less*, since it costs foraging time for nothing.
4. **L > C− on fed %.** If not, the extra neurons did the work and H4 is refuted by its
   own stated falsifier.
5. **Cs ≈ C0.** Hearing only yourself is not communication.

**Confidence: moderate on 1 and 3, low on 2.** Stated mechanism aside, my record on
predicting mechanism in this project is poor and this is the first time the ladder has
ever been run.

## 6. Falsifiers

H4's own, written into the tree before today:

- **L ≈ C?** — the channel carries no usable information, however good the transcripts.
- **C− ≈ L** — the neurons did the work and language was incidental.
- **Cs ≈ L** — it is private memory, not communication.

Any of those refutes H4 at this capacity and this task. A null here is a real result: it
would say the coop, not the learning rule, is what stands between this project and its
thesis — and that would redirect everything.

**One caveat recorded in advance.** The flock clumps at ~0.23 m nearest-neighbour. For
T1 that is benign and arguably realistic — a hawk is a shared threat and a tight flock
is what real birds form. For T2 (*which* feeder is poisoned) it would be fatal, since a
signal naming a location is useless when everyone is standing in the same place. That
problem is deferred, not solved.

**Command:** `python -m run.h4 --minutes 20 --seeds 8`

## 7. Result

*To be written after the run.*

## 8. Interpretation

*To be written after the run.*

## 9. Consequence

*To be written after the run.*
