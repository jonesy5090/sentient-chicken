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

48/48 cells, 8 matched seeds, 20 min, 16 hens.

| condition | fed % | caught rate | exposed | head down | alarm heard |
|---|---|---|---|---|---|
| N natural | 5.69 | 0.412 | 2878 | 0.662 | 0.1359 |
| C− capacity | 5.36 | 0.533 | 2331 | 0.676 | 0.0000 |
| C0 severed | 5.36 | 0.533 | 2331 | 0.676 | 0.0000 |
| C? shuffled | 5.13 | **0.282** | 2662 | 0.621 | 0.1442 |
| Cs self-only | 5.76 | 0.406 | 2616 | 0.662 | 0.0420 |
| **L language** | 5.39 | **0.280** | 2680 | 0.637 | 0.1481 |

```
HEADLINE  L vs C?:  caught rate  -0.002 +/- 0.051  t=0.05  noise
                    fed %        +0.256 +/- 0.204  t=1.25  not significant
```

## 8. **The control does not work. This is not a test of H4.**

Before writing the null up, the assumption the control rests on was measured: does the
shuffled channel actually destroy information about a hen's *own* hawk?

| | corr(heard, hawk on me) | heard given hawk | heard given no hawk | ratio |
|---|---|---|---|---|
| **intact** | 0.6239 | 0.9991 | 0.1408 | 7.09 |
| **shuffled** | **0.5610** | 0.9596 | 0.1473 | **6.52** |

**The shuffle retains 90% of the correlation and 92% of the signal-to-background
ratio.** It is not an uninformative channel. It is a slightly degraded copy of the
intact one.

The cause is geometry, measured directly: **when a hawk is live, 38.8% of the flock on
average (max 50%) is inside its strike radius.** Roughly six of sixteen hens share every
hawk, and hens standing together see the same sky. So a randomly reassigned sender is
very often reporting *your* hawk, and the permutation changes almost nothing.

`L ≈ C?` is therefore close to an identity, not a finding. **E024 cannot refute H4 and
does not.**

**The error is one sentence in this file's own §6**, written before the run: *"For T1
that is benign and arguably realistic — a hawk is a shared threat and a tight flock is
what real birds form."* That reasoning is about whether clumping is **realistic**. The
control does not need clumping to be unrealistic; it needs the shuffle to destroy
information. Those are different claims and I substituted one for the other, in a
caveat I wrote specifically to flag the risk.

Worse, the deeper problem was visible in the task choice from the start. **T1's
information is shared by construction** — a hawk over a flock is a common threat, so a
flockmate's alarm is informative about your situation *because* you are in the same
place. A control that scrambles the sender cannot destroy information that is carried
by proximity rather than by identity. `docs/backlog.md` §3 even says T1 "is not the
headline experiment"; I used it as one anyway.

### What E024 does establish

One contrast in the ladder is clean, because C− and C0 have **no** channel at all and so
carry no such confound:

**An audible alarm channel with innate comprehension roughly halves the caught rate** —
0.533 with no channel against 0.280 with one, at essentially unchanged feeding (5.36 vs
5.39). That is a real, measured benefit of having a channel.

It is *not* H4. H4 asks whether the **information** in the channel does the work, and
that question remains open: this design cannot separate information from arousal.

### Predictions, scored

All five wrong, and one instructively so.

1. L beats C? on strikes 15–40% — **wrong**, t=0.05.
2. L ≥ C? on fed % — direction right, not significant.
3. C? ≈ C0 — **wrong, and this was the tell.** A "zero-information" channel nearly
   halved the caught rate. I read that as a surprising result about arousal. It was the
   control announcing that it was not uninformative, and I should have checked then
   rather than after the full run.
4. L > C− on fed % — **wrong**, +0.030, t=0.16.
5. Cs ≈ C0 — **wrong**, 0.406 vs 0.533.

## 9. Consequence

- **H4 stays `NOT STARTED`.** No result is recorded against it. E024 is marked as a
  design failure rather than a null, because a null implies a valid test.
- **The flock clumping item is promoted to the top of the backlog.** It was filed as a
  T2 problem; it is a T1 problem too, and it blocks the headline experiment outright.
- **T1 is retired as the vehicle for H4.** It can still validate the harness and it does
  test *usage*, but its information is shared by geometry and no sender-scrambling
  control can work there. H4 needs a task where the private information is genuinely
  private to one hen.
- **New requirement for any H4 control:** the shuffle-informativeness diagnostic in
  `scratchpad/shuffle_info.py` must be run and reported *before* the ladder, not after.
  A control whose validity is untested is not a control.
- **Retained:** the ladder, the channel modes, capacity scaling, `caught_rate`, hawk
  targeting and the checkpointing all work and are reusable. The instrument is sound;
  the task was wrong.
