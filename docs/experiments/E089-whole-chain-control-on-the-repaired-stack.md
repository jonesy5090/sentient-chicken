# E089 — the whole-chain positive control, on a stack where every link is now measured

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**. Direct successor to
[E088](E088-frozen-centring-baseline.md).

This is **staging step 3** from `docs/backlog.md`'s T2-revised section, whose own wording
is the reason this experiment exists rather than the contrast:

> *Hand-plant a `W_pred` association (place P → gakel) and confirm a hen avoids P with no
> learning involved. **This is the step E065 skipped and three experiments paid for.** If a
> hand-wired success is undetectable, stop.*

---

## 2. Question

E082 and E083 both attempted this step and both returned no avoidance. Neither was a valid
test, and we now know why in detail:

- **The plant was anti-selective** — it drove the gakel channel at 0.656 at the target and
  1.244 elsewhere, with the innermost distance bin the *lowest* of seven (E083).
- **The metric could not resolve the predicted effect** — 18.3% minimum detectable at n=4,
  against a pre-registered 15% (E084).
- **The place signal was not readable** — position decoded at ~4 points above chance under
  movement (E085).
- **The readout removed most of what was there** — the centring baseline cost ~20 points
  (E086/E087).

Every one of those is now fixed and measured:

| link | state | evidence |
|---|---|---|
| place representable | hippocampus routed, `pred_src` extended | E086: parked 99.5% |
| place readable while moving | frozen centring baseline | E088: **89.0% on fresh seeds** |
| metric resolves the effect | per-seed target selection | E085: **MDE 5.1% at n=8** |
| anchor produces leaving | `M_PECK` only, no freeze | E083, guarded assay |
| chain conducts end to end | forward drive responds to the plant | E082 |

**So: with a correctly selective plant, on a readable representation, measured with an
instrument that can see the effect — does a hen avoid the poisoned feeder?**

If she does not *here*, T2-revised has no remaining instrumental excuse and the backlog's
"stop" applies.

---

## 3. Prediction

1. **Occupancy at the planted target falls by ≥15% relative** at `pred_gain=2.0`, against
   an instrument that resolves 5.1% at n=8 — so a real effect of the predicted size is
   comfortably visible.
2. **Monotonically** across the gain ladder.
3. **Control-cell occupancy does not fall** by more than 5%. This separates avoidance from
   agitation, and it is the discriminator, not the headline.
4. **Peck rate at the target falls**, which is the direct read on the reflex firing where
   it matters.
5. **Hunger rises modestly** — she declines food at one of several feeders — but stays
   below 0.60.

I hold prediction 1 with more confidence than at any previous attempt, and less than the
table above might suggest. Every *link* is measured; the *chain* end-to-end under a
selective plant never has been. E082 showed the chain conducts, but it conducted a signal
that was pointing at the wrong places.

## 4. Falsifier

**Primary.** Occupancy at the target falls by less than **10%** relative (roughly 2× the
metric's resolution), or falls non-monotonically. **This is the "stop" condition**: with the
plant validated, the representation readable, the anchor correct and the metric adequate,
a null here is a statement about the architecture and not about our instruments. T2-revised
would then need re-scoping rather than another repair.

**Agitation falsifier.** Control-cell occupancy falls by ≥10% alongside the target. She is
moving more, not avoiding a place.

**Starvation falsifier.** Mean hunger at gain 2.0 exceeds 0.60.

**Plant falsifier — a hard gate, run before anything else.** The plant must, on the
held-out run: decode at ≥80% balanced accuracy, fire ≥2× harder at the target than
elsewhere, and produce a distance profile decreasing across the first three bins. **The run
aborts if any of these fail.** E082 and E083 each passed an amplitude-only pre-flight while
the plant was useless; that pre-flight is replaced, not supplemented.

---

## 5. Design

### Configuration — every fix from E083–E088, on by name

`place_to_hippocampus=True`, `shared_place_map=True`, `gakel_scaffold=True`,
`place_cells_enabled=True`, `pred_bar_freeze_s=60.0` (E088's recommendation, chosen for
selectivity stability rather than the rule's 40 s), `contamination_enabled=False`.

Contamination stays off deliberately: the plant supplies the association, and a live
rotating contaminant would add a second, uncontrolled teaching signal. This is a control
for whether the anchor *can* produce avoidance, not a learning run.

### The plant — fitted on live states, gated on selectivity

Per seed, using E085/E086's machinery unchanged:

1. **Selection run** (`fold_in(k,2)`, gain 0, no plant) → target = highest-occupancy cell,
   control = second-highest. Chosen before any treatment, from an independent run.
2. Fit the discriminant on that run's **hippocampal** `z_lag − z_lag_bar` states, labelled
   by distance to the target — the configuration E088 measured at 89.0%.
3. **Gate on the held-out test run** against the plant falsifier's three conditions.
4. Only then install it into `W_pred`'s gakel row and run the behavioural ladder.

### The ladder

`pred_gain ∈ {0.0, 0.5, 1.0, 2.0}`, **8 seeds**, 20 simulated minutes, both arms of every
comparison on the test run key so selection bias cancels in the paired difference.

Reported per gain: occupancy at target and control, hunger, forward drive, peck rate at
target, and live `pred@gakel` split by at-target vs elsewhere — the split E083 had to
invent post-hoc, now standard.

### Cost

~30 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
