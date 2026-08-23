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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
