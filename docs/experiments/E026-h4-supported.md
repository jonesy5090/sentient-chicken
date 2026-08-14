# E026 — H4: an intact channel beats an uninformative one

> **Diagnostic-and-result, written after the fact.** E025 and E026 ran as a long chain
> of measurement, review and repair without an experiment file, which is a process
> failure against `CLAUDE.md`'s rule that sections 1–5 come first. This file is written
> from the commit record rather than pre-registered, and says so. The *replication* in
> §5 was decided on before its seeds were run, which is the part that matters most.

## 1. Parent hypothesis

**H4** — an intact channel beats a shuffled one on a task requiring private
information. The headline.

## 2. What had to be fixed first

E024 ran the ladder and its control did not work. Four defects stood between the
project and a testable H4; all four were found by measurement, three of them after an
outside review ([E022](E022-second-review-verified.md)) and one by arithmetic.

**The control was not a control.** `shuffled` permutes who-hears-whom within a
timestep. Every hen already hears every other — `hear_range` 15 m in a 20 m arena, mean
15.0 of 15 audible — so a permutation preserves "someone is calling right now", which is
almost the whole signal. Measured correlation with "a hawk is on me": intact **+0.5595**,
permuted **+0.5496** — 98% retained. Replaced by a **yoked** channel: the flock's real
calls, shifted in time per hen by more than a hawk dive. Correlation **−0.1288**.

*The diagnosis this replaced was mine and it was wrong.* I attributed the failure to
flock clumping and spent E025 on dispersal — food depletion (no effect: 23.0% → 21.9%
strike-radius overlap) and a gregariousness sweep (even at weight zero the permutation
kept 91%). The surviving component was temporal, not spatial.

**The risk metric was confounded three ways.** Raw strikes track where the flock stood.
`struck / exposure-steps` imports a behavioural confound — crouching zeroes locomotion,
so a partially-crouching hen lingers in the radius and inflates her own denominator;
exposure varied 15× across conditions and the two metrics disagreed about the *sign*.
Event-anchored `caught / at-risk-at-onset` was too sparse at 0.8–9.0 events per run.
Settled on **P(caught | at risk *and* blind at dive onset)**: the denominator is fixed
the instant the hawk commits, and restricted to hens who could not see it — the only
subset where a call carries information the receiver lacks.

**The world gave no warning interval.** `_step_predators` placed the hawk at its final
position the instant a dive began, so a hen was in danger from the same step she could
first have been told. Measured: P(caught | blind and at risk) was **1.000 deaf, 0.984
intact, 0.981 yoked** — a ceiling in every condition. Fixed with a 2 s visible approach
against a 12 s dive; the ceiling breaks to 0.500. This also explains why six attempts at
a staged assay each had to hand-build a stoop phase: I was supplying the missing
interval manually and reading its absence as a staging bug.

**The scaffold cannot hide a hen by itself.** `sigmoid(1.5 − 2.5) = 0.269` and hiding
requires `crouch > 0.5`. I wrote both halves and never put them together.

## 3. Design

Four conditions, no plasticity and no exploration noise anywhere, 16 hens, 1.5× pallium,
hawk every 20 s, 10 min per run.

| condition | channel | scaffold |
|---|---|---|
| deaf | none | yes |
| **intact** | intact | yes |
| **yoked** | real calls, time-shifted per hen | yes |
| intact, no scaffold | intact | **no** |

**Metric:** P(caught | at risk and blind at dive onset), paired across matched seeds.

## 4. Result

| contrast vs deaf | block A (0–11) | block B (12–23) | **pooled, 24 seeds** |
|---|---|---|---|
| **intact channel** | −0.187 ± 0.071 | −0.208 ± 0.095 | **−0.198 ± 0.059, t=3.33 SIGNIFICANT** |
| **yoked control** | +0.017 ± 0.040 | +0.052 ± 0.077 | +0.035 ± 0.043, t=0.80 noise |
| intact, no scaffold | +0.031 ± 0.046 | +0.103 ± 0.043 | +0.067 ± 0.031, t=2.13 |

Raw rates, block A: deaf 0.725, intact 0.582, yoked 0.753, no-scaffold 0.739.

## 5. Interpretation

**A hen who can hear her flockmates is caught ~20 percentage points less often, in
exactly the moments she could not see the hawk herself.**

**Both pre-registered falsifiers were checked and neither fires.**

- *L ≈ C? refutes.* The yoked control delivers identical calls, rate, amplitude
  distribution and energetic cost, differing only in carrying no contingency with the
  receiver's present. It sits flat at +0.035 in **both** blocks. The benefit is the
  **information**, not the arousal — the distinction E024 could not make.
- *The channel needs comprehension.* Intact-without-scaffold gives no benefit.

**The replication was decided before its seeds ran**, per the E021 rule, and it holds:
−0.187 then −0.208 on independent blocks. Block B misses significance alone (t=2.19 vs
2.201) with a *larger* magnitude — the opposite shape to E021's collapse, where a t=3.84
became t=0.01 on a 4.4× wider SE.

### What this does not license

- **No plasticity anywhere.** Comprehension is innate via the E018 scaffold. This shows
  a working channel helps; it does not show language is *learned*. H4's prediction never
  mentioned learning, so this is H4 as written — but H0 wants more, and H2 is still a
  clean null.
- **T1 only.** The signal means "danger". T2, where it must carry *which* feeder, is the
  referential test and remains blocked behind production being hardwired to stimulus
  classes.
- **A new world.** The approach phase is the third invalidation of the comparison basis
  today, after E023's E/I fix and E025's food depletion. Every other number in the tree
  predates it.
- **One unexplained result.** Intact-without-scaffold is significantly *worse* than deaf
  (+0.067). Hearing calls you cannot act on appears to cost something. No mechanism is
  offered; this project's record on inventing them is poor.

## 6. Consequence

- **H4 moves to `SUPPORTED` at this capacity and this task** — the first status past H1a
  to do so, on 24 seeds across two independent blocks.
- **T1 is vindicated as a harness**, having been retired as an H4 vehicle in E024 on a
  diagnosis that was wrong about the cause.
- **New backlog item:** why does an uninterpretable channel hurt?
- **Owed and unpaid:** E025 has no file; several E022 findings (credit window, audience
  assay confound, inert kin term, `strike_penalty` per-step) remain unverified.
