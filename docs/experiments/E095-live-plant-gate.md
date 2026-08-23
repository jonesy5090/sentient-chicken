# E095 — gate the plant on what actually drives behaviour

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**. Fixes the instrument that invalidated
[E093](E093-whole-chain-control-rerun.md), then re-runs it.

---

## 2. Question

Four whole-chain controls have now been invalidated by their own plant, for three distinct
reasons:

| experiment | what the plant did wrong |
|---|---|
| E070 | planted a matched filter, which scores below chance |
| E082/E083 | fitted on **parked** states, read on **moving** ones — anti-selective, 0.53× |
| E093 | fitted and gated in **fit space**, installed in **runtime space** — 0.91× live |

E093's is the worst of the three, because **the correct quantity was already being computed
and printed.** The results table carried `pred@target` and `pred elsewhere` in every run;
the gate consulted a different number and announced PASSED while the printed one read 0.91.

Two specific defects:

1. **The gate aggregates by mean.** Seed 2 returned a selectivity ratio of **5 909 203** —
   degenerate, because the discriminant's scores away from the target are all negative and
   `relu` sends the denominator to ~0. One such seed satisfies `mean(ratios) ≥ 2.0` on its
   own, regardless of the other seven.
2. **The gate measures the wrong space.** It scores the *fitted discriminant* on held-out
   samples. Behaviour is driven by the *installed, normalised* `W_pred` row through the
   runtime einsum, after `pred_src` masking and the `z_lag − z_lag_bar` centring. Those are
   different quantities and nothing checked the second.

**Does a gate on the live, installed quantity — per seed, minimum not mean — produce a
valid whole-chain control?**

---

## 3. Prediction

1. **Some seeds fail the live gate.** E093's live ratio averaged 0.91, so a per-seed
   minimum at 2.0 should exclude several. I predict **2–5 of 8** pass.
2. **Fit-space accuracy stays respectable on failing seeds.** This is the diagnostic value
   of the fix: a seed can separate sampled states well and still install a non-selective
   projection, which is exactly what went unnoticed.
3. **On the seeds that pass, the behavioural result is finally interpretable.** No
   prediction is made about its *direction* — the point of this experiment is the
   instrument, and predicting the outcome would invite reading the instrument through the
   result.

## 4. Falsifier

**Gate-viability falsifier.** Fewer than **6 of 8** seeds pass the live gate. E085
established the metric resolves 5.1% at n=8; below 6 the behavioural arm is underpowered
and **the experiment aborts rather than reporting an underpowered contrast.** This has been
the arc's recurring failure and it is not going to be repeated by reporting n=3.

**Construction falsifier.** The live gate and the fit-space gate agree on every seed. Then
E093's diagnosis was wrong — the two spaces would not be meaningfully different, and the
0.91 would need another explanation before anything is built on it.

**Behavioural falsifiers.** Unchanged from E093 §4 — primary, agitation, starvation,
anchor — evaluated **only on seeds that pass the gate**.

---

## 5. Design

### The gate, rebuilt

**Measured in the live run, with the plant installed**, which costs nothing extra: the
`pred_gain=0.0` arm of the ladder already computes `relu(pred@gakel)` and already splits it
by at-target versus elsewhere. That arm is the baseline the contrast needs anyway, so the
gate runs on data the experiment was already producing — the number that has been printed
and ignored for three experiments.

Per seed:

- **live selectivity** = mean `relu(pred@gakel)` at the target ÷ mean elsewhere, **≥ 2.0**
- reported alongside, as secondary diagnostics only: fit-space accuracy, fit-space ratio,
  and the distance profile

**Aggregated by minimum, never mean.** The statistic is a ratio with a `relu` in the
denominator and can diverge by construction, so a mean is the wrong summary for it — and
was the wrong summary in E093, where one seed at 5.9 million carried the gate.

### Seed exclusion, and why it is legitimate here

Seeds failing the live gate are **excluded from the behavioural analysis and counted in the
report.** This is exclusion on a criterion measured *before* and *independently of* the
behavioural outcome — the manipulation either took or it did not — which is the one form of
exclusion that does not select on the result. The count is reported whether it is 0 or 6.

If it drops below 6, the gate-viability falsifier fires and nothing behavioural is
reported.

### Everything else

Identical to E093: 8 seeds, `pred_gain ∈ {0, 0.5, 1, 2}`, 20 simulated minutes, depletion
at default, E090's anchor weights, E091's scratch suppression, `peck_stops_walking=0.0`,
per-seed target and control from an independent selection run.

### Cost

~30 minutes — the same as E093, since the gate reuses the baseline arm.

---

## 6. Result

| seed | live @T | live elsewhere | **live ratio** | fit acc | fit ratio | verdict |
|---|---|---|---|---|---|---|
| 0 | 1.006 | 1.719 | **0.59** | 68.1% | 3.03 | fail |
| 1 | 1.000 | 0.407 | 2.46 | 85.8% | 2.46 | PASS |
| 2 | 1.009 | 1.726 | **0.58** | 65.2% | **5 909 203** | fail |
| 3 | 1.001 | 0.427 | 2.34 | 86.7% | 2.34 | PASS |
| 4 | 1.022 | 0.123 | 8.32 | 86.2% | 8.32 | PASS |
| 5 | 1.297 | 0.261 | 4.97 | 94.0% | 4.97 | PASS |
| 6 | 1.536 | 4.884 | **0.31** | 83.5% | 4.25 | fail |
| 7 | 1.108 | 0.293 | 3.78 | 88.5% | 3.79 | PASS |

Live ratio: **min 0.31**, median 2.46. **5 of 8 seeds pass.**

**The gate-viability falsifier fires** (5 < 6) and the run **aborted before reporting any
behavioural number.** That is the design working as intended: E093, with the same
connectomes and the same plants, reported a full behavioural contrast here.

**The construction falsifier is clear, decisively.** The two gates disagree on three of
eight seeds, so the fit and runtime spaces are genuinely different and E093's diagnosis
holds.

### 6b. The disagreement has a shape, and it is one-sided

Not pre-registered; visible once the two columns sit side by side.

**On every passing seed, fit ratio and live ratio agree to two decimals** — 2.46/2.46,
2.34/2.34, 8.32/8.32, 4.97/4.97, 3.78/3.79. **On failing seeds they diverge wildly** —
3.03 against 0.59, 5 909 203 against 0.58, 4.25 against 0.31.

So the fit-space measure is **not systematically wrong**. It is a **one-sided error**: it
agrees exactly when the plant is good, and cannot detect when it is not. That is the worst
possible property for a gate, and it explains how it survived four experiments — it was
right every time anyone had reason to look at it.

**A mechanism I proposed and then falsified against this same table, recorded because the
falsification is the useful part.** The obvious suspect is the normalisation: the plant is
installed as `w_h / (w_h · sP)`, so a small or negative inner product would amplify or flip
the vector. **The `live @T` column rules that out.** By construction
`pred@target = (w_h · sP)/(w_h · sP) = 1.0` whatever the denominator does — and the column
duly reads 1.006, 1.000, 1.009, 1.001, 1.022, 1.297, 1.536, 1.108. Every seed normalises
correctly at the target, including all three failures.

**The failures are entirely in the `elsewhere` column** — 1.719, 1.726, 4.884 against
0.123–0.427 for the passing seeds. So the installed plant fires *correctly* at the target
and *also* fires everywhere else. That is a discriminability failure in the live regime,
not a scaling one.

Seed 6 is the sharpest case and rules out "the fit was simply bad": 83.5% fit accuracy,
fit ratio 4.25, live elsewhere **4.884**. It separates sampled states well and generalises
to the live trajectory not at all. Seeds 0 and 2 have poor fit accuracy too (68.1%, 65.2%),
so they are less informative.

**Why is not established.** It is the same family as E083's parked-versus-moving finding —
a discriminant fitted on one distribution of states and read on another — but E095's fit
and test runs are both live, so that explanation does not transfer directly. This needs its
own measurement.

## 7. Interpretation

**The instrument is fixed, and the fix immediately paid for itself.** A gate on the live,
installed quantity rejected three seeds that four previous experiments' gates would have
accepted — including one whose fit-space ratio was 5.9 million and whose actual plant fired
harder away from the target than at it.

**And it stopped the experiment rather than reporting an underpowered contrast.** E084's
finding was that this arc had been reporting effects the metric could not resolve; E095 is
the first time a run has refused to produce a number on those grounds. The abort is the
result.

**The one-sidedness is the part worth carrying forward.** A diagnostic that is exactly
right whenever the thing works, and silently wrong whenever it does not, will pass every
casual check. The only way to catch it was to measure the quantity that actually drives
behaviour — which had been computed and printed since E083 without being consulted.

## 8. Consequence

**Nothing behavioural is claimed, and T2's status is untouched.** Five valid seeds is not
a result; it is a gate reading.

**Two things are needed before the whole-chain control can run, and they are separate.**

1. **Find out why three seeds' plants generalise to the live trajectory and five do not.**
   §6b rules out normalisation and rules out "the fit was bad" (seed 6: 83.5% accuracy,
   live elsewhere 4.884). Until that is understood, a gate rejecting 3 of 8 is *filtering
   for* a defect rather than measuring the hen, and the arc will keep paying for it.
2. **Then widen the seed pool if still needed.** 16 seeds at the observed 62% pass rate
   gives ~10, above the threshold. Increasing n is a legitimate response to an underpowered
   design **provided the gate and its criterion do not move** — and they must not. But this
   is second, not first: more seeds buys a runnable experiment while leaving a defect that
   discards 38% of them unexplained.

**Recorded so it is not re-derived: `relu` in the denominator of a gate statistic.** Both
the mean-aggregation defect (E093) and the 5.9-million reading come from the same source —
a ratio whose denominator can approach zero by construction. Any future gate of this shape
wants a floor, a minimum aggregation, or a different statistic entirely.
