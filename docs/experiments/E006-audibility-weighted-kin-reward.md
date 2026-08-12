# E006 — does audibility-weighted kin reward rescue the audience effect?

> **Partially pre-registered, and the exception is disclosed.** The comprehension
> assay (§5) was written and run as a 2-seed smoke test *before* this file existed,
> to check the code worked. It returned a decisive mechanical result, so sections 3
> and 4 below were written already knowing it. The audience-effect predictions were
> not informed by any 6-seed data. Recorded plainly rather than dressed up as clean
> pre-registration.

## 1. Parent hypothesis

**H3** — learned usage reproduces the audience effect without being programmed.

## 2. Question

[E005](E005-does-the-audience-effect-emerge.md) diagnosed its own null: the kin term
was a flock average, nearly identical for every hen, so no bird could discover she was
responsible for a call. Does weighting it by audibility — so a hen's reward reflects
the flockmates who could actually hear her — fix it?

## 3. Prediction

**No.** The fix addresses a real defect but not the binding one.

The chain that has to close for audience-sensitive calling to be learnable is: she
calls → a flockmate **hears and responds** → the flockmate avoids a strike → the
benefit returns to her through the kin term. E005 attacked the last link. The
comprehension assay says the chain is broken at the *second*: flockmates do not
respond to calls at all, so a call never helps anyone, so it can never repay its
energy cost no matter how the reward is weighted.

**Prediction:** comprehension stays at zero in all conditions, the audible-kin
condition does not differ meaningfully from the flat-kin condition, and the audience
effect does not emerge in either.

## 4. Falsifier

Audible-kin producing a significant audience effect where flat-kin does not. That
would mean credit assignment was the binding constraint after all and comprehension
is not a prerequisite in the way argued here.

## 5. Design

- **Change under test**: `kin_audible` in `hen/plasticity.py`. True weights the kin
  term by the same distance attenuation `coop/sensing.py` already uses for hearing;
  False reproduces E005's flat flock mean exactly, as a within-experiment control.
- **New measurement — the comprehension assay** (`run/audience.py`): Evans & Marler's
  playback design. Present an alarm call with **no predator to see** and measure
  crouching. Driven by a hand-built observation rather than a staged world, so the
  call is the only thing that varies.
  - Comprehension is *learned*, not innate — `hen/innate.py` deliberately wires no
    reflex from the auditory channels — so at hatch this must be zero. If it is
    non-zero at hatch the assay is broken.
- **Conditions**: audible kin / flat kin (E005) / fixed. **Seeds**: 6.
  **Rearing**: 30 min. Growth off, per H2a.
- **Command**: `python -m run.audience --minutes 30 --seeds 6`

## 6. Result

6 seeds, 30 min rearing, 16 hens. Wall clock 19 min.

| condition | when | alarm alone | alarm aud. | effect | food alone | food aud. | effect |
|---|---|---|---|---|---|---|---|
| audible kin | hatch | 0.321 | 0.321 | −0.000 | 0.390 | 0.396 | +0.006 |
| audible kin | reared | 0.241 | 0.215 | −0.026 | 0.655 | 0.701 | +0.046 |
| flat kin (E005) | reared | 0.390 | 0.358 | −0.031 | 0.580 | 0.617 | +0.038 |
| fixed | reared | 0.321 | 0.321 | −0.000 | 0.390 | 0.396 | +0.006 |

```
audible kin       alarm  -0.026 +/- 0.024 SE   t=1.11   (threshold 2.57)
audible kin       food   +0.040 +/- 0.044 SE   t=0.90
flat kin (E005)   alarm  -0.031 +/- 0.022 SE   t=1.43
flat kin (E005)   food   +0.032 +/- 0.050 SE   t=0.64
fixed             both    0.000 +/- 0.000 SE
```

**Comprehension — the decisive column:**

| condition | hatch | reared | change |
|---|---|---|---|
| audible kin | 0.0001 | −0.0005 | −0.0005 |
| flat kin | 0.0001 | −0.0003 | −0.0004 |
| fixed | 0.0001 | 0.0001 | +0.0000 |

## 7. Interpretation

**The prediction held: audibility weighting does not rescue the audience effect.**
Audible-kin and flat-kin are statistically indistinguishable (food t=0.90 vs t=0.64;
alarm t=1.11 vs t=1.43). E005's diagnosis identified a real defect and fixed it, and
the defect was not the binding one.

**Comprehension is zero, before and after rearing, in every condition.** Hearing an
alarm call with no predator present changes a hen's behaviour by roughly one part in
ten thousand. That is not a small effect; it is nothing.

That number explains both nulls at once. A call that nobody responds to cannot help
anyone, so it can never repay its energy cost, so the only component of the reward
that correlates with a hen's own calling is the private, immediate **cost** — and a
rule that sees only the cost of calling learns to suppress calling. Which is exactly
the sign observed on the alarm channel across both experiments.

### Why comprehension does not emerge: there is no exploration

This is the finding, and it is more consequential than the experiment that produced
it.

To learn "crouch when you hear an alarm call", a hen must **at some point crouch when
she hears an alarm call** and be rewarded for it. The three-factor rule strengthens
synapses that were active when the modulator moved. It has no way to reinforce an
action that never happened.

And crouching never happens on hearing a call. Crouching is driven by the innate
aerial reflex, which requires *seeing* the hawk. `hen/innate.py` deliberately wires no
reflex from the auditory channels, because comprehension is supposed to be learned.
So the behaviour is never sampled, and what is never sampled can never be reinforced.

**The model is deterministic.** Same state, same action, every time. There is no motor
noise, no stochastic action selection, no exploration of any kind. Every behaviour the
hen will ever perform is one the innate reflex arc already produces; learning can only
re-weight the conditions under which existing behaviours fire. It cannot acquire a new
stimulus-response pairing that the reflex arc does not already visit.

That constraint is invisible while testing H2, because drive regulation only requires
tuning behaviours the hen already performs — approach food, peck, huddle. It becomes
binding the moment the question is about acquiring something genuinely new, which is
every hypothesis from H3 onward, and emphatically the language work.

**Real chicks are not deterministic.** Behavioural variability in young animals is
well documented and is generally understood as the substrate reinforcement acts on.
Adding exploration is a correction toward the biology, not a hack around a null.

**What audibility weighting did change.** It is not inert: the audible-kin flock ends
with markedly more food calling (0.655/0.701 vs 0.580/0.617) and markedly less alarm
calling (0.241 vs 0.390) than flat-kin. So the reward change reached behaviour — it
simply did not produce audience-conditionality, because that requires a listener who
responds.

## 8. Consequence

- **H3 stays `UNDER TEST`** with a second null recorded. Two experiments, one real
  defect fixed, hypothesis unmoved.
- **`kin_audible=True` kept as the default.** It is better biology, it demonstrably
  changes behaviour, and E005's argument for it stands on its own merits even though
  it did not rescue H3.
- **H2b opened**: the learning rule cannot acquire behaviours outside the innate
  repertoire's reach, because it has no exploration. This now blocks H3, H4 and H5 —
  everything about language.
- **E007**: add decaying motor exploration noise, re-measure comprehension first.
  Comprehension is the cheap, mechanical readout — if it does not move, nothing
  downstream will, and there is no point running the audience assay at all.
- **Assay design note worth keeping**: the comprehension measure cost almost nothing
  to build and answered a question two full experiments had been circling. When a
  hypothesis depends on a chain of steps, measure the *cheapest link* first rather
  than the endpoint.
- **No ethics review triggered.**
