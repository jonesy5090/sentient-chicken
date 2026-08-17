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

**8 seeds (0–7), `food_deplete_rate=0`, otherwise identical to E032/E033:**

```
MANIPULATION CHECK: mean |dW_out| / |W_out| = 0.0960   PASSES (fixed drift 0.000000)

rearing         intact  lesioned      drop
trained         31.364    30.703     0.661
fixed           29.946    28.332     1.615

PRIMARY interaction  -0.9539 +/- 0.7011   t=1.36   threshold 2.365   NOT SIGNIFICANT
  (E033's pooled 24-seed result: +0.390 +/- 0.153, t=2.55, SIGNIFICANT)

secondary, both intact: fed % +1.4172 +/- 1.0767  t=1.32
```

**The falsifier fires.** §4 registered: sign reversal at `food_deplete_rate=0` means
E033's `REFUTED` verdict needs withdrawing pending a full clean re-measurement. The sign
reversed (+0.390 → −0.954). Not significant at n=8 (this check was explicitly not
powered to establish anything on its own), but a reversal is a stronger signal than a
shrinking-toward-zero would have been, and this is not the outcome the prediction in §3
leaned toward.

**Absolute levels moved as expected, confirming this is the same mechanism E037 found.**
fed % here is 28–31, roughly double E032/E033's 11–14 — consistent with E037's ~2–2.6×
depletion effect on this exact duration and flock size.

**This result does not settle anything, and treating it as if it did would repeat this
project's own most expensive lesson.** E021 spent an entire experiment establishing that
one seed block cannot move a status. An 8-seed sign flip against a 24-seed significant
result is exactly that situation. Extending to a full, properly powered 24-seed
clean re-measurement, below, rather than leaving H2e's status decided by 8 seeds in
either direction.

**Extension: full 24-seed clean re-measurement, `food_deplete_rate=0`, matching E033's
exact block structure (0–11, 12–23), reusing this file's cache:**

```bash
python -m scratchpad.e032 --seeds 24 --rear 20 --test 5 --food-deplete-rate 0.0 \
  --cache scratchpad/e038_audit_cache.json --budget 100000
```

```
MANIPULATION CHECK: mean |dW_out| / |W_out| = 0.0981   PASSES (fixed drift 0.000000)

rearing         intact  lesioned      drop
trained         30.680    30.485     0.195
fixed           31.616    30.531     1.085

PRIMARY interaction  -0.8902 +/- 0.5564   t=1.60   threshold 2.069   NOT SIGNIFICANT
  (E033's depleted-world result: +0.390 +/- 0.153, t=2.55, SIGNIFICANT)

secondary, both intact: fed % -0.9356 +/- 1.3016  t=0.72
```

## 7. Interpretation

**On equal footing — 24 seeds each — the clean world gives a null and the depleted
world gave a significant positive result.** The sign is negative in both the 8-seed
interim check (−0.954) and the full 24-seed run (−0.890), reasonably consistent between
them, and opposite to E033's +0.390. Neither clean measurement clears its threshold.

**E033's `REFUTED` verdict does not replicate under a properly controlled, equally
powered test, and the most likely explanation is the confound this file set out to
check.** This is not proof the confound *caused* E033's result — a true effect could
still exist and simply not have cleared threshold here (t=1.60 is not "zero," it is
"not significant," and the CI still contains meaningfully negative values) — but the
burden has shifted. A result that appears at t=2.55 in one world and vanishes (even
trending in the *opposite* direction) at t=1.60 in a properly controlled version of the
identical seeds and design is not a result that stands on the depleted-world number
alone.

**H2e reverts to `UNDER TEST`, not to `SUPPORTED`.** The clean null does not confirm
H2e's original claim (the pathway is inert) either — t=1.60 is a null, not a
significant result in H2e's predicted direction. What is settled is narrower: **E033's
specific `REFUTED` finding does not hold up**, and H2e is back to being an open question,
now on a connectome and a world both correctly controlled for the first time.

## 8. Consequence

- **H2e moves from `REFUTED` to `UNDER TEST`** in `docs/hypothesis.md`. E032 and E033
  are corrected in place (struck through, not deleted) with pointers here, per this
  project's standing convention.
- **H2's own "the null is informative again" framing, which depended on H2e being
  refuted, reverts too.** H2's null (E037, +0.0003 ± 0.0156) stands on its own — that
  measurement already controlled for `food_deplete_rate` — but whether it is
  *informative* (whether the pathway can carry a trained signal to behaviour at all)
  is open again, exactly where E031 left it.
- **E007's multiplicative-gating question, which H2f's write-up noted was "on hold"
  pending this result, stays open** rather than resolving either way.
- **A properly powered clean measurement of H2e now exists and should be the number
  cited going forward**: interaction −0.890 ± 0.556, t=1.60, 24 seeds,
  `food_deplete_rate=0`. Not `REFUTED`, not `SUPPORTED` — a null, on the first version of
  this experiment run without an uncontrolled confound.
- **The broader audit `docs/backlog.md` flagged (E032/E033/E036 all exposed to
  `food_deplete_rate`) is partially closed by this file** — E032/E033 checked and found
  materially affected. E036 remains reasoned-about-but-unchecked: its primary metric
  comes from a short, deterministic staged assay run after rearing rather than the
  depleting world directly, and its own data already shows stability across a 15×
  duration range (2-min smoke test comprehension 0.1899 vs. the 30-min full run's
  0.1921) — evidence it is less exposed, not a direct test of the same kind this file
  ran for E032/E033.
