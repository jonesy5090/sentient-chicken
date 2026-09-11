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

*(written after the run)*

## 7. Interpretation

*(written after the run)*

## 8. Consequence

*(written after the run)*
