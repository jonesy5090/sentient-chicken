# E018 — does an innate auditory reflex unblock learned usage?

> **Pre-registration.** Sections 1–5 written before any code changes, at commit
> `d2c9d60`. Sections 6–8 to be filled in after the run. Nothing below may be edited
> once the first run starts; if the design turns out to be wrong, that is recorded in
> §7 rather than fixed in §5.

## 1. Parent hypothesis

Primary: **[H3](../hypothesis.md#h3)** — learned usage reproduces the audience effect
without being programmed. Currently UNDER TEST with two nulls (E005, E006) and marked
blocked by H2b.

Secondary: **[H2b](../hypothesis.md#h2b)** — the learning rule cannot acquire
behaviours outside the innate repertoire. E018 tests H2b's stated *mechanism*, not H2b
itself.

## 2. Question

E006 and E007 both returned null on comprehension and both explained it the same way:
the chain that H3 requires —

> she calls → a flockmate hears and responds → the flockmate avoids a strike → the
> benefit returns to her through the kin term

— is broken at **step two**, because `hen/innate.py` wires no response to hearing a
call and exploration alone never produces one ([E017](E017-where-separability-is-lost.md)
confirmed every auditory entry in `reflex_matrix()` is exactly zero). E007 added motor
noise to make the response occur by accident; comprehension stayed at ~0.0005.

Real chickens do not learn this response from scratch either. Parentally naive chicks
already respond differentially to conspecific fear calls, and the learned part is
association off an already-arousing stimulus (Curio's mobbing chain), not discovery by
trial and error. So the model has been asking plasticity to solve a harder problem than
the biology poses.

**The question: if step two is closed by construction, does the rest of the chain
close on its own?**

## 3. Prediction

Registered before running, and stated against the control that matters — the scaffold
with learning off, **not** against the current no-scaffold baseline.

1. **Primary.** The audience effect index `A = alarm_audience − alarm_alone` will be
   larger with learning than without, at the same scaffold: `A(S+L) − A(S) > 0`,
   significant at two-tailed p < 0.05 on 8 matched seeds. E005 measured `A ≈ 0` at
   hatch, so any real value is the whole effect.
2. **Manipulation check, not a result.** The comprehension index will be ≈ 0 in the two
   unscaffolded conditions and ≈ 0.25 ± 0.05 in the two scaffolded ones. If it is not,
   the scaffold was not wired as specified and nothing else in the run means anything.
3. **Secondary, exploratory.** Strikes per hen will fall in S against N *without any
   learning*, by 10–30%. This would be an innate benefit and is called out in §5 as the
   result most likely to be misreported.
4. **Secondary, exploratory.** Mean hunger will *rise* in S against N — crouching at
   every call that carries costs foraging time, and the scaffold is context-blind.

**Confidence, stated honestly.** Low. My record on predicting the mechanism behind
these results is 1-for-7, and prediction 1 additionally depends on a learning rule that
E017 raised specific doubts about. Prediction 1 failing is the more informative outcome
and is treated as such in §4.

## 4. Falsifier

**If `A(S+L) − A(S) ≈ 0`, E006's and E007's stated explanation is false.** Those
experiments concluded H3 fails because flockmates do not respond to calls. E018 makes
flockmates respond to calls. If the audience effect still does not emerge, the broken
link was never step two, and every experiment that has tried to fix the *routing* of
learning (E002, E007, E008, E009) has been treating the wrong thing.

That outcome would promote the open item E017 left unnamed — that the rule is the wrong
*kind*, instrumental where the biology is Pavlovian — from speculation to the leading
hypothesis, and it should get its own node in the tree.

Against H2b: a positive result weakens it (the rule *can* acquire a new behaviour, once
given a foothold), a null strengthens it and narrows its mechanism.

**Not a falsifier:** any result in condition S alone. See §5.

## 5. Design

### The trap this design exists to avoid

Wiring `hear alarm → crouch` and then reporting "she responds to alarm calls" would be
reporting the wiring back to ourselves. The whole design is built around one rule:

> **Anything measurable in condition S is wired. Only `S+L − S` is learned.**

The scaffold is therefore run *with learning off* as a first-class condition, not as an
afterthought, and every primary number is a difference against it. The 2×2 exists so
that this subtraction is available for every metric.

### Conditions

Full 2×2, matched seeds, matched genome, matched coop, matched predator arrivals.

| | no scaffold | scaffold |
|---|---|---|
| **no learning** | N — current baseline | **S — the control that matters** |
| **learning** | N+L — current state, known null | S+L — the test |

`explore_sigma` stated explicitly in all four conditions (the E010 confound), and equal
across them: this experiment is not about exploration.

### What the scaffold wires, exactly

Added to `hen/innate.py`. Weights fixed a priori by the rule below and **not tuned on
any metric in this experiment** — a weight sweep chosen on the primary metric would
invalidate the whole thing and is a separate follow-up if wanted.

| from | to | weight |
|---|---|---|
| aerial alarm heard | `M_CROUCH` | **+1.5** |
| aerial alarm heard | `M_PECK`, `M_SCRATCH` | **−1.5** |
| ground alarm heard | `M_FLEE` | **+1.5** |
| ground alarm heard | `M_PECK`, `M_SCRATCH` | **−1.5** |

**Why 1.5.** Against `REST_BIAS = −2.5`, a full-amplitude adjacent call gives
`sigmoid(1.5 − 2.5) = 0.27` — a partial, graded response. That matches the biology:
naive chicks show *longer tonic immobility* to a fear squawk, not full anti-predator
behaviour. It is 19% of the visual crouch weight of 8.0, so seeing a hawk always
dominates hearing about one, which is the right ordering — first-hand information should
beat second-hand.

**Why the peck/scratch suppression is in the scaffold and not left out.** Crouching does
not raise the head: `coop/actuation.py` derives `head_down` from peck and scratch only,
and crouch merely zeroes locomotion. So a hen who crouches while still pecking remains
blind to the sky, and the call restores nothing. Suppressing the head-down actions is
what converts a heard call back into the information the caller had. It is also what
real birds do — they raise their heads at an alarm call.

**This is the part of the scaffold most likely to do the experiment's work for it**, and
it opens a purely innate route to a survival benefit: call → head up → *sees the hawk
herself* → the existing visual reflex fires at weight 8.0, with no learning anywhere.
That route is real, is how it works in nature, and would show up as a strike reduction
in condition S. It is pre-registered here as **innate**, and §7 must not report it as
evidence for H4 or for communication being learned.

**Explicitly not wired**, and each for a reason:

- **No call relay.** Hearing an alarm does not trigger producing one, though real
  chickens do chain alarm calls. A relay makes the acoustic environment self-driving and
  changes it for every hen at once, which would confound the audience assay — the thing
  being measured is exactly whether a hen calls more when others are present. Separate
  question, separate experiment.
- **No posture-, audience- or context-dependence.** The scaffold fires the same whether
  she is head-down or head-up, alone or in a flock, hungry or fed. That conditioning is
  what learning has to add, and wiring any of it would be wiring in the answer.
- **Nothing on the food or contact call channels.** They stay neutral with respect to
  predators, which keeps a second-order-conditioning test (§ below) available later.

**A constraint discovered while designing this, recorded because it limits §3.** The hen
has **no proprioceptive channel for her own posture** — `spec.OBS_DIM` carries vision,
the aerial channel, four drives, wall and speed, and audio, but nothing reporting
head-down. So she *cannot* learn a posture-conditional response to calls, however good
the rule is: the conditioning variable is not observable to her. That was the first
primary metric considered here and it was discarded as unlearnable by construction.
Adding proprioception would change `OBS_DIM` and invalidate the comparison basis for all
seventeen prior experiments, so it is a backlog item, not part of E018.

### Metrics

**Primary, chosen now:** the audience effect index `A = alarm_audience − alarm_alone`
from the existing Evans & Marler assay in `run/audience.py`, unchanged. It is the right
primary for three reasons: the scaffold touches *comprehension* and A measures *usage*,
so nothing in the scaffold acts on it directly; it is a documented real-chicken
behaviour the model was never told about; and it is the exact quantity E005 and E006
returned null on.

Statistic: paired across matched seeds, two-tailed t against the `_t_critical()` table
in `run/experiment.py` (the E003 fix — no 2-SE thresholds).

**Manipulation check:** the comprehension index from `run/audience.py`. Reported as
"did we wire what we said we would", explicitly labelled not-a-result.

**Secondary, all marked exploratory:**
- strikes per hen (survival), with the innate-route caveat above
- mean hunger, the H2 metric — the scaffold could plausibly make foraging worse
- `|W_out|` and live synapse count, to check the scaffold does not interact with the
  connectome erosion E014/E015 characterised

**One planned ablation, only if S shows a strike reduction:** scaffold with the crouch
term but no peck/scratch suppression. That separates "she hides because she was told"
from "she looks up and sees it herself", which are different mechanisms with the same
number attached.

### Replicates

**8 matched seeds.** E009 found per-genome separability varies 3.5%–25.5%, so single
seeds are meaningless here; E004 needed 12 to firm up a t=3.93. 8 is a deliberate
compromise between that and a 2×2 costing four runs per seed. If the primary lands
between p=0.05 and p=0.15, the pre-registered response is to extend to 16 seeds and
report both — not to report the 8-seed result as a trend.

### Commands

```bash
python -m run.audience --minutes 30 --seeds 8 --scaffold-2x2
python -m pytest tests/ -q
```

Base commit `d2c9d60`. A regression test must assert the scaffold is absent in the two
unscaffolded conditions, in the same family as `test_fixed_control_is_actually_fixed`.

### Ethics

Checked before writing, per `docs/ethics.md` §7. This adds a fixed sensorimotor reflex
from an existing auditory channel to existing motor channels. It is **not** tripwire 2:
an aversive channel architecturally distinct from a homeostatic drive. No tripwire moves
and no review is triggered. Noted only because "innate fear response" is the kind of
phrase that sounds like it should trigger one, and the check is cheap.

### Addendum, after implementation and before the run

Appended rather than edited into §3, which stands as registered.

**Prediction 2's number was arithmetically wrong when I wrote it.** I predicted the
comprehension index at 0.25 ± 0.05, but that is the *absolute* crouch level under a
call; the index is a **difference**, crouch-with-call minus crouch-without. The correct
prediction from the registered weight is `sigmoid(1.5 − 2.5) − sigmoid(−2.5) = 0.269 −
0.076 = 0.193`. Measured on the built scaffold: **0.1893** (bare arc: −0.0002).

So the manipulation check passes — the scaffold is wired exactly as specified, and the
predicted band was mis-derived, not the implementation. Recorded here because a
pre-registered number that quietly moves to match a measurement is the failure mode
pre-registration exists to prevent, and because prediction 2 is a manipulation check
rather than a result: nothing about the primary is affected either way.

**§5's justification for the primary metric is wrong, and the design survives it
anyway.** §5 argued the audience effect was a safe primary because "the scaffold
touches *comprehension* and A measures *usage*, so nothing in the scaffold acts on it
directly". A 2-seed 2-minute smoke test says otherwise: condition S shows A = +0.066
against N's +0.013. The scaffold moves the audience effect **innately**, and the
mechanism is the head-up route flagged in §5 under a different heading — with an
audience present, flockmates call, the focal hen hears them, stops pecking, raises her
head, *sees the hawk herself* at 7 m, and the visual reflex drives the alarm call at
weight 7.0. Alone, there is nobody to call, so she stays head-down and never sees it.
An audience effect, entirely innate, with no learning anywhere.

I did not anticipate that pathway when choosing the metric. **The design absorbed it
regardless**, because the primary is `S+L − S` and S contains exactly this. That is the
"anything measurable in S is wired" rule doing the job it was written for, against a
mechanism I failed to predict — which is the point of building the control in rather
than reasoning about why one is unnecessary.

It does raise the bar for the primary: `S+L − S` must now clear a floor that already
sits at +0.066 rather than at zero. Registered predictions unchanged.

**One secondary looks underpowered.** `hawk_period_s = 900`, so a 30-minute rearing
gives ~2 hawk passes; the smoke test recorded 0.00 strikes per hen in every condition.
Prediction 3 (strikes fall 10–30% in S) may simply be unmeasurable at this duration. It
is left as registered and will be reported as unmeasurable if it comes back flat, not
quietly rescued by extending the run.

## 6. Result

*To be written after the run.*

## 7. Interpretation

*To be written after the run.*

## 8. Consequence

*To be written after the run.*
