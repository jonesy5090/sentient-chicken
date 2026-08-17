# E038 — does E032/E033's causal-efficacy interaction survive without food depletion?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2e** — REFUTED by E033, on a pooled 24-seed test finding a trained readout costs
something measurable to remove. This audit checks whether that result is confounded by
`food_deplete_rate`, per `docs/backlog.md`'s item flagged in E037.

## 2. Question

E037 found that `food_deplete_rate` (added by E025 for an unrelated question, left in
`spec.DEFAULT_COOP`'s permanent defaults) collapses foraging over a 20-minute, 16-hen
run, and that H2's own PHASE1 contrast moved substantially in absolute terms once
controlled for (though not enough to flip its qualitative null). E032/E033 used the
identical rearing duration and flock size, with no override, and their design is more
exposed to a depletion-driven confound than PHASE1's: rearing behaviour differs between
`trained` (plasticity on) and `fixed` (plasticity off) flocks, so the two conditions can
plausibly accumulate *different* depletion by the end of rearing — a difference in
starting world state for the outer `trained` vs `fixed` comparison the primary
interaction subtracts, not protected by the intact/lesioned forking the way the inner
comparison is (both forks share the identical end-of-rearing world by construction).

**Does E033's interaction (+0.390 ± 0.153, t=2.55, pooled 24 seeds) survive at
`food_deplete_rate=0`?**

## 3. Prediction

**No strong directional prediction**, matching E037's own honesty about this. The inner
intact-vs-lesioned pairing is protected by forking regardless of depletion, so the
mechanism the interaction measures should be structurally robust; if it is not, that
itself is informative about how much the outer trained-vs-fixed depletion difference
matters. Given E037 found the *direction* of H2's own null was unaffected by depletion,
weak prior toward the interaction's sign also surviving — but E037's *magnitude* moved
substantially, so no claim that the exact interaction size will match.

## 4. Falsifier

**If the interaction collapses to near-zero or reverses sign at `food_deplete_rate=0`**,
E033's `REFUTED` verdict for H2e was reached on a confounded run and needs withdrawing
pending a full clean re-measurement. **If it holds** (same sign, comparable order of
magnitude, even if this smaller sample doesn't clear its own threshold), that is not
proof of no confound, but it substantially de-risks H2e's `REFUTED` status pending the
full audit this file does not have the budget to run at E033's original 24-seed scale.

## 5. Design

Identical to `scratchpad/e032.py`, with `food_deplete_rate=0` (new `--food-deplete-rate`
argument added to the script for this audit). **8 seeds (0–7)**, a subset of E032's own
block-one seeds, chosen so results are directly comparable seed-by-seed against the
already-cached depleted-world numbers in `scratchpad/e032_cache.json` — same genome, same
predator arrivals, same everything except this one config field.

**Command:**
```bash
python -m scratchpad.e032 --seeds 8 --rear 20 --test 5 --food-deplete-rate 0.0 \
  --cache scratchpad/e038_audit_cache.json --budget 100000
```

**Not a full replication.** 8 seeds at a reduced count is a de-risking check, not a
registered replacement for E033's 24-seed pooled result — explicitly not powered to move
H2e's status on its own, only to flag if something is badly wrong.

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
