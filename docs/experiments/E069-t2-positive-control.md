# E069 — the positive control T2 has needed since E065

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2**, and more broadly the reward-gated learning rule itself. E065 flagged that no
positive control specific to the sickness-per-rotation metric had been run, and
called it "the next step if this result is ever treated as more than a first pass."
Three runs later (E065, E066, E068), all withdrawn as tests of T2's claim, it is well
past that point. `CLAUDE.md`: *"Before concluding a rule cannot learn something, show
the harness detecting an effect you have deliberately planted. If it cannot see a
hand-wired success, it cannot see a real one."*

This also serves as the missing end-to-end validation of E067's fix (merged in #39),
which so far is proven only by a unit test showing the signal *reaches*
`consolidate()` — not that learning measurably improved as a result.

## 2. Question

Two separate questions the T2 series has run together, asked apart here:

- **A — can the metric see it?** Given the real run-to-run variance of
  sickness-per-rotation at n=8 seeds, what is the smallest effect the
  early-vs-late difference-in-differences could reliably detect? E065–E068 reported
  contrasts of +1.13, −0.19 and +1.06 against SEs of 0.72–1.25; if the minimum
  detectable effect is larger than any effect T2 could plausibly produce, every one
  of those nulls was uninformative by construction.
- **B — can the rule learn it?** If `sickness_penalty` is raised far beyond the
  point where it competes with ambient reward (E068: currently ~0.007% of total
  reinforcement), does the flock actually learn to get sick less — and does the
  connectome survive it?

## 3. Prediction

**A**: minimum detectable effect ≈ `t_crit × SE` ≈ `2.365 × 0.87` ≈ **2.1
sickness events per rotation** at E068's variance, against a baseline rate of ~8 —
i.e. the harness can only see changes of roughly 25% or larger. Stated before
computing it from the data.

**B**: genuinely uncertain, and that is the point. Three outcomes are all live:
(i) large penalties produce learned avoidance → calibration was the whole problem,
T2 becomes testable; (ii) no penalty produces avoidance at any magnitude → the
obstacle is architectural (E066/E067's other two candidate reasons), and T2 needs a
redesign, not a bigger number; (iii) large penalties destroy the connectome before
they teach anything → the E014 failure mode, and the reward design has no workable
operating point.

## 4. Falsifier

This is a control, so the falsifier applies to the *instrument*, not to T2:

- If **A** shows the minimum detectable effect is comparable to or larger than the
  entire baseline sickness rate, the metric cannot answer T2's question at any
  practical sample size and must be replaced before T2 runs again.
- If **B** shows no response at *any* penalty magnitude, including ones large enough
  to dominate reward entirely, then the reward-gated pathway cannot drive this
  behaviour at all, and no amount of calibration rescues T2's current design.

Either outcome is more useful than a fourth null contrast.

## 5. Design

**A — minimum detectable effect.** Pure re-analysis of `scratchpad/e068_cache.json`,
no new compute. Take L's real per-seed early/late values, inject a synthetic uniform
shift of known size into the late window, and find the smallest shift at which the
paired t-statistic crosses `_t_critical(7) = 2.365`. Reported as both an absolute
count and a fraction of the baseline rate. Repeated on E065's and E066's caches so
the answer reflects the range of variance actually observed, not one lucky block.

**B — penalty sweep.** `sickness_penalty ∈ {0, 1, 10, 100, 1000}`, 4 seeds each, 60
minutes (12 rotations) per run. Single channel condition (`intact`) — this asks
whether the flock learns to avoid at all, not whether the channel helps, so the L/C?
contrast is deliberately absent. H2f's rule otherwise unchanged from E066/E068.

Three measurements per run:
- **sickness per rotation**, early (rotations 1–3) vs late (10–12) — did she learn?
- **mean `|W − W_innate|`** — did the weights move? (`|W|`, not `|W_out|`: under
  `hebbian_readout` the readout is not reward-gated and cannot respond, which is the
  diagnostic error E068 found running unnoticed through three experiments)
- **live synapse count** — did a large discrete penalty erode the connectome, the
  E014 failure mode the backlog flagged as the risk in "just raise it"?

4 seeds and 12 rotations is deliberately coarse: B is looking for a gross,
unmistakable response, and A is doing the precision work. If B shows something at
this resolution it warrants a proper powered run; if it shows nothing at
`sickness_penalty=1000`, no sample size will rescue `sickness_penalty=1`.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e069_positive_control.py
```

## 6. Result

**A — minimum detectable effect**, from the real variance in three seed blocks:

| experiment | baseline/rotation | SE(contrast) | MDE (count) | MDE (% of baseline) |
|---|---|---|---|---|
| E065 | 8.94 | 0.715 | 1.69 | 18.9% |
| E066 | 8.43 | 1.249 | 2.95 | 35.0% |
| E068 | 8.47 | 0.866 | 2.05 | 24.2% |

The pre-registered prediction (≈2.1 events, ≈25%) lands mid-range.

Empirical cross-check on E068: the smallest injected reduction in L's late window
that clears threshold is 3.11, against an analytic MDE of 2.05. Not a discrepancy —
E068's *observed* contrast was +1.06, so reaching significance in the negative
direction requires travelling 1.06 + 2.05 = 3.11. The two numbers answer different
questions ("how large must a true effect be, from zero" vs. "how far must this
particular observed result move") and agree exactly.

**B — `sickness_penalty` sweep**, 4 seeds, 12 rotations, 60 min/run:

| penalty | early/rot | late/rot | late−early | mean \|W−W₀\| | synapses | vs. innate |
|---|---|---|---|---|---|---|
| 0 | 6.92 | 7.92 | +1.00 | 3.745e-04 | 35627 | 98.0% |
| 1 | 6.75 | 6.83 | +0.08 | 3.888e-04 | 35617 | 98.0% |
| 10 | 7.75 | 7.83 | +0.08 | 3.689e-04 | 35628 | 98.0% |
| 100 | 7.25 | 8.50 | +1.25 | 3.841e-04 | 35617 | 98.0% |
| 1000 | 7.33 | 7.00 | −0.33 | 4.711e-04 | 35433 | 97.5% |

## 7. Interpretation

**Outcome (ii) of the three pre-registered possibilities: no penalty magnitude
produces learned avoidance. The obstacle is not calibration.**

Sickness rate does not fall at any penalty. The `late−early` column runs +1.00,
+0.08, +0.08, +1.25, −0.33 across a thousandfold increase in the reward term — no
trend, no monotonicity, and the single negative value (−0.33 at penalty 1000) is
roughly a sixth of the minimum detectable effect Part A establishes for a
better-powered 8-seed block. It is noise.

**The signal does reach the weights; it just does not become behaviour.** Mean
`|W−W₀|` rises 26% between penalty 0 and penalty 1000 (3.745e-04 → 4.711e-04), so a
sufficiently large reward term demonstrably perturbs the recurrent weights — E067's
fix works and the pathway is live. But undirected perturbation is all it is: a large,
sparse, noisy reward term drives *more* weight change without driving *useful* weight
change. The weights move; the hen does not learn where not to eat.

**E014's erosion failure mode did not occur, and that risk can be retired.** The
backlog flagged "just raise it" as unsafe on the precedent of a discrete event scaled
large enough to destroy the connectome. At a thousandfold penalty the connectome
retains 97.5% of its innate synapses against 98.0% at zero — a half-point difference.
Raising `sickness_penalty` is safe. It is simply useless.

**This converges with, and now independently confirms, E066/E067's architectural
explanation.** E058/E059 established that this project's working rule *amplifies an
existing innate anchor* and does not build an association from nothing. E063 then gave
place cells no innate reflex whatsoever — deliberately, and stated as such: *"raw
location carries no innate meaning alone, by design."* Those two facts are jointly
sufficient to predict this result. There is nothing linking a place-cell pattern to
any motor output for the rule to strengthen, so no amount of reward flowing through
`m` can produce place-specific avoidance. Part B is that prediction confirmed
empirically rather than inferred.

**Part A's separate verdict: the metric itself is sound.** It resolves changes of
roughly 19–35% of baseline at n=8. T2's own prediction — L converging toward one
mistake per rotation while C? pays N times that — is far larger than that threshold,
so the metric was never the limiting factor. It was adequate for the question; the
learning was absent.

## 8. Consequence

**Close the backlog's "calibrate `sickness_penalty`" item as answered: calibration is
not the fix, and the E014 safety concern attached to it is retired.** Both were
reasonable given what was known; both are now settled by measurement.

**T2 as currently designed cannot be rescued by tuning.** It requires one of:

1. **A weak innate place-linked anchor for the rule to amplify** — the one route the
   project's own evidence (E058/E059) says works. This is a real design decision with
   a real hazard: an anchor specific enough to make T2 learnable risks hardwiring the
   answer T2 is meant to discover, which is the line `hen/innate.py`'s
   `auditory_scaffold` already walks deliberately and keeps off by default. Any such
   anchor would need the same treatment — opt-in, documented as scaffolding, and
   never on in a headline condition without saying so.
2. **A different learning rule** capable of building a new association rather than
   amplifying an existing one — which is the open problem behind H2c's null
   (E058/E059) and is not specific to T2.
3. **Accepting T2 as answered in the negative for this architecture**, and recording
   that the rotating-poisoned-feeder task is beyond what this rule can learn — a
   legitimate outcome, and the honest reading if neither (1) nor (2) is attempted.

**The wider open item stands unchanged**: E067's `strike_penalty` audit, sharpened by
`CLAUDE.md`'s own note that the term carries 87% of reward variance at the H4
configuration while reaching the weights 2% of the time. Nothing here bears on it —
this experiment measured the sickness term only.

**One methodological note worth keeping.** The positive control was called for by
E065 and deferred through three subsequent experiments, each of which produced a null
that was then explained by a newly discovered defect. Running it first would have
established in a single 30-minute sweep that the task was unlearnable at any signal
strength, which is the fact all three of those experiments were circling. The rule in
`CLAUDE.md` is not ceremonial.
