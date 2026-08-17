# E025 — does food depletion disperse the flock, and what actually does

> **Retrospective, written after the fact.** This experiment ran (commits `f659745`,
> `8843126`, `71a8cf2`, `c95b4bb`) and its results have been cited in `docs/backlog.md`,
> `docs/hypothesis.md` and [E026](E026-h4-supported.md) ever since, but no file was ever
> written — flagged as the project's oldest process debt since E027 §8. Reconstructed
> here from the preserved commit messages and scratchpad scripts, which already contain
> full tables and reasoning; nothing below is inferred beyond what those record. Sections
> 1–5 describe the question as it was actually approached, not as a clean pre-registration
> — this run predates the project's own discipline about writing sections 1–5 first for
> everything, which is itself worth being honest about.

## 1. Parent hypothesis

**H4** — an intact channel beats a shuffled one on a task requiring private information.
E024 built the H4 ladder and found the shuffled control retained ~90–98% of the intact
channel's information, traced to the flock clumping so tightly that most hens already
share every hawk. E025 was spent on the leading fix.

## 2. Question

Does giving food patches finite capacity — deplete under feeding pressure, regrow when
abandoned — disperse a flock that currently clumps at 0.23–0.40 m nearest-neighbour
distance, restoring the shuffle control's ability to destroy real information?

## 3. Prediction

Implicit in the commit that built it (`f659745`): foraging competition should push hens
toward different patches, spreading them out, because "four patches is still only four
places worth standing" was the predicted failure mode if it *didn't* work — recorded in
the negative-result commit rather than invented after.

## 4. Falsifier

Nearest-neighbour distance and strike-radius overlap unchanged by depletion — would mean
foraging pressure is not what holds the flock together, and the dispersal question needs
a different candidate force.

## 5. Design

`coop/world.py`: `food_amount` depletes at `food_deplete_rate` per second per hen feeding
at a patch and regrows at `(1 - food_amount) / food_regrow_s`. Rates chosen (per the
commit) "so a patch supports a couple of birds for roughly a minute and recovers over
about five." Three linked diagnostics, each a single-seed, no-plasticity, fixed-hen
rollout at 16 hens, 6 minutes (36,000 steps) — quick ablations built to answer one
question each, not pre-registered multi-seed contrasts:

1. **Depletion on vs. the old infinite-patch world** — does clumping (nearest-neighbour
   distance, strike-radius overlap) change at all.
2. **Ablation**: zero thermal pressure (no cold accumulation) vs. zero gregariousness (no
   flockmate-approach reflex) vs. both — which force is actually binding.
3. **Weight sweep** on the gregariousness reflex, chosen against the quantity H4's control
   actually depends on (shuffled-channel information retention), not against the
   strike-radius proxy.

## 6. Result

**Depletion alone does essentially nothing:**

| | nn dist | spread | in strike radius | food left |
|---|---|---|---|---|
| infinite (old) | 0.40 | 6.42 | 23.0% | 1.000 |
| depleting (E025) | 0.39 | 6.50 | 21.9% | 0.701 |

The mechanism works — patches genuinely draw down to 0.70 remaining — and the flock does
not care. 23.0% → 21.9% is noise. (Note for comparability, recorded in the original
commit: this measures the old condition at 23.0%, where E024's own diagnostic measured
38.8% — different windows and seed counts; only the within-run contrast is valid.)

**The real cause, found by ablation:**

| ablation | nn dist | spread | in strike radius | cold | fed/hen |
|---|---|---|---|---|---|
| baseline | 0.39 | 6.50 | 21.9% | 0.157 | 1044 |
| no thermal pressure | 0.33 | 6.36 | 17.9% | 0.200 | 992 |
| no gregariousness | 1.62 | 17.23 | **6.8%** | 0.273 | 1399 |
| neither | 1.65 | 17.55 | 7.9% | 0.200 | 1425 |

Thermotaxis was the predicted culprit (huddling is the coop's only heat source) and
barely moves the number. Removing the flockmate-approach reflex — gregariousness — cuts
strike-radius overlap 21.9% → 6.8%, spreads the flock across the arena, and *raises*
feeding 34% as hens stop crowding one patch.

**The wiring defect underneath it:** vision is proximity-graded
(`prox = 1 - d/vision_range`), so turn-toward strengthens the *closer* a flockmate already
is — attraction with no repulsion, where real fowl have a documented individual distance
(attraction at range, repulsion when crowded). Fixing this properly needs a crowding
channel, which changes `OBS_DIM` and invalidates every prior comparison — the same
objection that has shelved proprioception (E018 §5) — so this was not done.

**Weight sweep**, chosen against H4's control-information metric rather than the
strike-radius proxy:

| gregariousness weight | nn dist | spread | in strike radius | cold | fed/hen |
|---|---|---|---|---|---|
| 1.20 (default) | 0.39 | 6.50 | 21.9% | 0.157 | 1044 |
| 0.60 | 0.43 | 6.63 | 20.1% | 0.180 | 1194 |
| 0.30 | 0.77 | 9.78 | 18.0% | 0.224 | 1296 |
| 0.12 | 1.13 | 14.72 | 11.3% | 0.257 | 1319 |
| 0.00 | 1.62 | 17.23 | 6.8% | 0.273 | 1399 |

A smooth trade-off, no knee — feeding rises monotonically, cold stays modest throughout,
so no welfare cliff forces a choice by eye. At weight 1.20, the shuffled channel retained
**90%** of the intact channel's information about a hen's own hawk (see E026) — still too
high, but the decision was made on that number, not on strike-radius overlap.

## 7. Interpretation

**Food depletion is not the dispersal fix it was built to be, and was kept anyway.** It
does not disperse the flock (falsifier condition fires cleanly: 23.0% → 21.9%, noise).
What actually holds the flock together is the gregariousness reflex, whose vision-graded
strength has no repulsive term at close range — attraction-only wiring that a proper
individual-distance model would need a new observation channel to fix. `food_deplete_rate`
was left in the world at its tuned default anyway, on the reasoning (from the commit) that
"a patch supports a couple of birds for roughly a minute and recovers over about five,"
and its own message states the intent that it "does not run out of food over a
20-minute run."

**That specific intent does not hold, and this was found independently by
[E037](E037-h2-rebaseline.md), by accident, twelve experiments later.** E037 set out to
re-measure H2's clean null on the E023-corrected connectome and found "fixed (innate
only)" hens' `fed %` had collapsed from E020/E021's 6.6% to ~2.5%, with hunger rising
substantially over a 20-minute run — not explained by gain or the cortical readout
(directly ruled out), but by `food_deplete_rate`, run for its full registered 20 minutes
at 16 hens rather than E025's 6-minute diagnostics, driving individual patches down to
**0.01** remaining rather than the 0.70 these shorter runs saw. Depletion compounds with
duration in a way none of E025's own diagnostics were run long enough to see, because
none of them needed to be — they were built to answer a dispersal question, not a
foraging-baseline one.

## 8. Consequence

- **The dispersal question is answered and was recorded at the time**: gregariousness's
  attraction-only wiring is the cause, not thermotaxis, and not food scarcity. Fixing it
  properly needs a crowding/individual-distance channel — filed as a backlog item
  already, unchanged by this write-up.
- **`food_deplete_rate=0.02` stays in `spec.DEFAULT_COOP`**, but the justification for
  leaving it on ("does not run out of food over a 20-minute run") is now known to be
  false at the duration and flock size several other experiments actually use it at. This
  write-up does not change the default — that is E037's call to make, since H2 is the
  hypothesis it silently affected — but flags the assumption as false rather than leaving
  it standing uncontradicted in this file.
- **Every experiment using `spec.DEFAULT_COOP` at `n_hens=16` for 20+ minutes without an
  explicit `food_deplete_rate` override since commit `f659745`** should be treated as
  having run in a food-scarcer world than anything before it, whether or not that was
  the point of the experiment. Worth an audit — not performed here — of which tree
  results this actually touches versus which used short enough windows or different
  overrides to be unaffected. E026–E030's H4 ladder, in particular, uses 10-minute
  windows, closer to E025's own diagnostics than to E037's 20; whether that duration
  keeps depletion negligible there has not been checked either.
- **This file's own existence is the other consequence.** The process debt it closes was
  three experiments' worth of citations resting on a changelog line rather than a record.
