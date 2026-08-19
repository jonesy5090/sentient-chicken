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

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
