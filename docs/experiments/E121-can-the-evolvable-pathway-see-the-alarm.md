# E121 — can the evolvable pathway even see the alarm?

> §1–5 written before anything ran.

## 1. Parent hypothesis

**H0**, via [E120](E120-the-h0-ladder.md). The ladder returned an informative null:
selection over 24 lineages produced no advantage for an intact channel, on a substrate
where selection works (E118) and at a density where a *planted* comprehension is worth
−1.195 catches per hen (E120 §6a). E120's leading explanation, predicted in advance:

> `evolve._mutate` touches only `W` and `W_out`, while the one demonstrated route from an
> alarm call to a crouch runs through `p.reflex` — never plastic, never mutated. Selection
> was asked to rebuild, in the pathway it can reach, a benefit only ever shown through one
> it cannot.

That is an explanation, not a measurement. This measures it.

**A correction to E120 §8's own successor list first.** It proposed "plant-then-select":
give founders the scaffold and ask whether selection retains it. **That experiment is
void as specified** — the scaffold lives in `p.reflex`, which `_mutate` never touches, so
nothing could ever degrade it and "does selection retain it" is trivially yes. Recorded
here rather than quietly dropped.

## 2. Question

Does information about a heard alarm reach the motor stub — the only thing `W_out` reads —
and if it does, can a hand-planted `W_out` mapping turn it into a crouch that saves hens?

## 3. Prediction

`brain.step` computes `cortical = einsum(W_out, motor_stub)` where
`motor_stub = rates[:, -n_motor:]`. So `W_out` sees the MOTOR region's rates and nothing
else — not the observation, not the sensory stub, not the pallium directly.

Between the alarm arriving and the motor stub there are three synapses and two documented
losses: E103 measured situation-specific signal falling from 31% to 2.2% in the *first*
synapse because `W_in` is strictly positive, and E105 measured direction stability of
0.9930 at the motor stub with the specific signal intact underneath only after the
population mean is removed.

So, numerically, before looking:

1. **The alarm is decodable at the sensory stub** — AUC above **0.80** for "an aerial
   alarm is audible", among hen-steps where she cannot see a hawk herself.
2. **It degrades through the pallium** — AUC **0.60–0.75**.
3. **It is barely present at the motor stub** — AUC **below 0.60**. This is the prediction
   that matters: if `W_out` cannot read the alarm, no setting of `W_out` can act on it, and
   E120's null is explained by the substrate rather than by the search.
4. **A planted `W_out` mapping therefore recovers little of the ceiling** — under **a third**
   of the reflex scaffold's −1.195 catches per hen, at any gain.

If prediction 3 is wrong and the alarm *is* legible at the motor stub (AUC > 0.70), then
E120's leading explanation is wrong, the substrate was adequate, and the null is about the
search or the criterion — which would send the next work somewhere completely different.

## 4. Falsifier

**For E120's explanation:** if the alarm is decodable at the motor stub *and* a planted
`W_out` mapping recovers a substantial share of the −1.195 ceiling, then the evolvable
pathway could have expressed comprehension and did not — E120's null would then be about
selection's reach, not about anatomy, and "make the reflex arc heritable" would be the
wrong successor.

**Against a decoding artefact:** the decode must be restricted to hen-steps where the hen
**cannot see a hawk herself** (`IDX_AERIAL` below a threshold). Otherwise "an alarm is
audible" is nearly synonymous with "a hawk is overhead", the decoder reads the hen's own
vision, and a high AUC would mean nothing about communication — E024's failure exactly,
where a control retained 98% of its information because every hen already hears every
other.

**Against a planting artefact:** a `W_out` column large enough to force crouching will
reduce catches whether or not it is driven by the alarm, simply by making her crouch all
the time. So the planted arm is compared against a **magnitude-matched scrambled plant** —
the same weights on the same units, permuted across motor channels — and against mean
crouch rate. A plant that only raises baseline crouching is recorded as degenerate, the
same check that caught `readout_1.00` in E117 and `mut_0.30` in E118.

## 5. Design

**Part A — where does the alarm signal die?** Run a normal flock at `hawk_period_s=20`
(E120's density, `channel_mode="intact"`), chunked so internal rates are recoverable.
For each hen-step, record: the heard aerial channel, her own `IDX_AERIAL` vision, and the
rates of the sensory stub, pallium and motor stub.

Restrict to hen-steps with `IDX_AERIAL` below its 25th percentile — she cannot see a hawk.
Among those, label "alarm audible" as heard-aerial above its 75th percentile. Decode with a
**single-unit best-AUC** and a **pooled linear discriminant**, reported separately, at each
of the three regions. 8 genome seeds.

**Part B — plant it.** Take the linear discriminant direction at the motor stub, write it
into `W_out`'s `M_CROUCH` row at gain *g*, and sweep *g* over four values. Measure catches
per hen, crouch rate, and hunger against three references: unplanted, the
magnitude-matched scrambled plant, and the reflex scaffold's −1.195.

**Primary metric.** Motor-stub decode AUC (Part A) — it is what decides whether Part B can
work at all, and what E120's explanation stands or falls on.

**Secondary, pre-declared:** the sensory-stub and pallium AUCs (they localise the loss);
catches per hen at each planted gain; crouch rate at each gain (the degeneracy check).

**Replicates.** 8 genome seeds for Part A, matching E117/E119. Part B is a sweep, reported
as exploratory in magnitude but with the scrambled control at every gain.

**Command.**

```bash
PYTHONPATH=. python scratchpad/e121_can_wout_see_the_alarm.py
```

## 6. Result

### 6a. The evolvable pathway is not blind — prediction 3 was wrong

AUC for "an aerial alarm is audible", among hen-steps where she cannot see a hawk herself.
8 genome seeds, `hawk_period_s=20`.

| region | best single unit | pooled (LDA) |
|---|---|---|
| sensory stub | 0.8611 ± 0.0231 | 0.8457 ± 0.0213 |
| pallium | 0.6629 ± 0.0172 | 0.7376 ± 0.0222 |
| **motor stub** | 0.6090 ± 0.0136 | **0.6871 ± 0.0187** |

Predictions 1 (>0.80 at the stub) and 2 (0.60–0.75 at the pallium) hold. **Prediction 3 —
below 0.60 at the motor stub — is wrong.** It is 0.687, comfortably above chance and just
under the 0.70 I had set as "E120's explanation is wrong".

This is not an abstract information measure. `brain.step` computes
`cortical = einsum(W_out, motor_stub)`, which is a pooled linear readout of the motor
stub — the same class of operation as the LDA that scores 0.687. **So 0.687 is what
`W_out` can actually read, in the units `W_out` works in.** E120's leading explanation, as
stated, is too strong: the substrate is dim, not blind.

Where the signal dies is also informative: **0.846 → 0.738 → 0.687**. Almost the entire
loss is in the *first* synapse, sensory stub → pallium. E103 measured the same thing from
the other side — "situation-specific signal drops from 31% to 2.2% in ONE synapse". The
pallium → motor step costs only 0.05.

### 6b. And it is unusable anyway — plant buys nothing over a matched scramble

Writing the motor-stub discriminant into `W_out`'s `M_CROUCH` row, against a scramble with
identical norm, identical channel and identical DC treatment. Unplanted: 3.375 catches/hen,
crouch 0.399.

| gain | plant caught | scram caught | crouch (plant / scram) | **plant − scram** |
|---|---|---|---|---|
| 0.01 | 3.453 | 3.203 | 0.430 / 0.429 | +0.250 ± 0.270, t=+0.93 |
| 0.02 | 2.953 | 3.109 | 0.455 / 0.455 | −0.156 ± 0.244, t=−0.64 |
| 0.05 | 2.711 | 2.336 | 0.546 / 0.546 | +0.375 ± 0.317, t=+1.18 |
| 0.10 | 1.750 | 1.867 | 0.660 / 0.658 | −0.117 ± 0.258, t=−0.45 |
| 0.25 | 0.984 | 0.930 | 0.796 / 0.790 | +0.055 ± 0.190, t=+0.29 |

**The scramble is matched to three decimal places on crouch rate at every gain**, which is
the control working exactly as designed, and **plant − scram alternates sign with every
|t| < 1.2.** Planting the alarm-aligned direction buys nothing whatsoever over a direction
carrying no alarm information at all. Catches track *how much she crouches*, not *when*.

**Decodability is not usability.** AUC 0.687 implies a discriminability of about 0.69
standard deviations, so the alarm-driven swing in the projection is smaller than its own
noise. To move crouching during alarms by a behaviourally useful amount, the same weights
move it during everything else by more. E081 named this distinction for the pallium; this
is it at the readout.

### 6c. The first sweep was degenerate, and the control is what showed it

Gains 0.5–4.0 were run first. There, catches fell to **8–14%** of unplanted and crouch sat
at **0.85–0.89** — the hen crouching most of her life, catches on the floor with no room to
differ (CLAUDE.md check 5). The scramble captured almost the whole benefit
(−2.60 against the plant's −2.91 at gain 0.5).

Reported rather than discarded, because had only the plant arm been run it would have read
as "planting comprehension into `W_out` cuts predation by 86%" — a spectacular result that
is entirely an artefact of breaking the bird in a direction that happens to score well.
Same shape as E117's `readout_1.00` and E118's `mut_0.30`.

### 6d. The real finding: unconditional vigilance pays, and needs no information

Using the scramble arm — which carries *no* information by construction — and E120's
calibrated `caught_weight=0.1049`:

| crouch rate | catches | hunger | Δfit (predation) | Δfit (hunger) | **net** |
|---|---|---|---|---|---|
| 0.399 | 3.375 | 0.4593 | — | — | — |
| 0.546 | 2.336 | 0.4626 | +0.1090 | −0.0032 | **+0.1058** |
| 0.658 | 1.867 | 0.4698 | +0.1582 | −0.0104 | **+0.1477** |
| 0.790 | 0.930 | 0.4829 | +0.2565 | −0.0235 | **+0.2330** |

**Doubling the crouch rate is worth +0.233 fitness and requires no information at all.**
There is a large, communication-free hill in this fitness landscape. Both arms of E120's
ladder climb it, equally, and it dwarfs anything a channel could contribute.

### 6e. A defect I reported and then disproved

I read `actuation.py`'s comment — *"Crouching is freezing… invisible to the hawk, but not
foraging and not going anywhere"* — against `world.py`'s `fed = at_food_any & pecking`,
concluded that crouch suppresses only `mobility` and therefore a hen could crouch and eat
simultaneously, and drafted this as a defect: vigilance is free, so the trade-off the
many-eyes literature rests on was never in the model.

**Measured, it is false.** P(fed | at food, crouching) = **0.3632** against **0.6896**
upright: a crouching hen feeds at **52.7%** of the upright rate, **−0.3264 ± 0.0195,
t=−16.71**. The comment is approximately right and my reading of the code was not.

The reconciliation with §6d is a third number: **a hen is at a food patch only 4.5% of the
time.** So a 47% local penalty lands on a twentieth of her life, and the aggregate cost of
crouching twice as much is 0.024 hunger against 2.4 catches saved. **The trade-off is real
locally and weak globally** — which is a subtler and more interesting fact than the defect
I thought I had found, and one I would have got wrong twice over by reading rather than
measuring.

## 7. Interpretation

**E120's null now has a mechanism, and it is not the one E120 proposed.** The evolvable
pathway can see the alarm (AUC 0.687 at exactly the stage `W_out` reads). What it cannot do
is *act on it selectively*: at that signal-to-noise, any readout strong enough to make her
crouch when warned makes her crouch when not, and a scrambled direction of equal magnitude
delivers the same predation benefit. So E120's explanation survives only in a weakened
form — the substrate is not blind, it is too noisy for conditional behaviour — and a new,
stronger explanation sits above it.

**The stronger explanation: the task does not require communication.** Unconditional
crouching is worth +0.233 fitness here with no information involved. An alarm call is
valuable precisely when vigilance is expensive enough that you want it *only when
warranted*. In this coop the aggregate cost of vigilance is small — not because crouching
is mechanically free (it halves feeding) but because hens spend 4.5% of their lives on a
patch — so "always crouch" beats "crouch when told", and the channel has nothing to add
that the flock could not get for free. **A communication experiment whose optimal policy is
unconditional is not a communication experiment.**

That indicts the *task*, not the brain, and it is the first explanation in this arc that
does. It also reframes every predation-based result the project has: E120's ladder, and
arguably H4 itself, were run in a world where the behaviour an alarm recommends is worth
doing regardless.

**What would fix it**, and both are changes to `coop/`, not to `hen/`:

1. **Make vigilance expensive in aggregate**, by raising time-on-patch — slower feeding,
   larger patches, or a longer dwell requirement — until unconditional crouching stops
   paying. The target is measurable and stated: the net fitness gain of doubling the crouch
   rate should be ≈ 0, not +0.233. Then re-run E120's ladder.
2. **Give a strike a real homeostatic cost** (E119's standing item), so predation stops
   being a scoring convention chosen by me and the weighting stops being mine to set.

**What I cannot rule out.** That a better readout than a diagonal-covariance LDA would
extract more from the motor stub — the 0.687 is a lower bound on decodability, though it is
the right bound for `W_out`, which is itself a linear readout. And that longer selection
would find the conditional solution even though it is a smaller prize than the
unconditional one; E118 found the search exhausted by generation 2, which argues against it
but does not exclude it.

**A caveat for T1.** T1 is `SUPPORTED as a narrower claim` — no intake benefit, a real
safety benefit. §6d and §6e together suggest why: in a coop where vigilance costs 4.5% of a
hen's life, a safety benefit is cheap and an intake benefit has almost no room to appear.
T1's result is not overturned, but its interpretation should be read against a
vigilance/foraging trade-off that is much weaker than the literature it is checked against.

## 8. Consequence

**`docs/hypothesis.md`.** H0 records the refined mechanism for E120's null: the evolvable
pathway sees the alarm and cannot act on it selectively, *and* the task rewards
unconditional vigilance. T1 gains a pointer to §6d/§6e.

**`docs/backlog.md` §8.** "Make the reflex arc heritable" is **demoted** — E120's stated
rationale for it (the pathway cannot see the alarm) is now measured false. Promoted in its
place: raise the aggregate cost of vigilance until unconditional crouching stops paying,
then re-run the ladder. E119's strike-cost item is promoted alongside it.

**E120 §7 corrected in place** with a pointer here: its leading explanation was too strong.

**No code changes.** `hen/` and `coop/` are untouched; `actuation.py`'s comment is
vindicated and stays as written.

**Code.** `scratchpad/e121_can_wout_see_the_alarm.py`,
`scratchpad/e121c_is_crouching_free.py`, `scratchpad/e121_cache_firstsweep.json` (the
degenerate sweep, kept deliberately).
