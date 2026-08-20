# E083 — redesigning T2's gakel anchor so it produces *leaving*, not *stopping*

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** (the rotating poisoned feeder), specifically the
**T2-revised** associative route, and within it **mechanism 1** — the innate response
to *hearing* a gakel call (`hen/innate.py`'s `_add_gakel_scaffold`).

Direct successor to [E082](E082-t2-chain-control-redone.md), which established that
T2-revised's chain conducts end to end and fails at exactly one point.

---

## 2. Question

E082 planted the place→gakel association by hand, confirmed it fires (pre-flight 1.000
at the planted place per seed, 0.86–0.96 live), and watched forward drive fall 17%
(0.622 → 0.519). Every link in the chain worked. **Occupancy at the poisoned feeder did
not fall.**

The diagnosis was mechanical, not neural. `_add_gakel_scaffold` suppresses `M_FORWARD`,
and `coop/actuation.py:46` derives speed from it:

```python
speed = mobility * (fwd * cfg.walk_speed + flee * cfg.flee_speed)
```

So a hen already *at* the bad feeder who slows down **stays at it**. The anchor produces
lingering where avoidance requires leaving. Its own docstring declines to borrow the
anti-predator response — "*no crouch or flee. This is bad food, not a predator*" — and
the implementation is a functional freeze regardless.

**Does removing the `M_FORWARD` suppression, leaving only the `M_PECK` suppression,
convert the response from lingering into leaving?**

### Why this is the cheapest candidate, mechanically

Forward drive is never wired to food. Reading `hen/innate.py`, `M_FORWARD` receives
`IDX_HUNGER` (+2.0, line 113), `IDX_THIRST` (+1.5), `IDX_COLD` (+2.5), a tonic bias
(line 317), and `IDX_AERIAL` (−6.0). Food drives `M_PECK` only (line 83). A hen does
not stop at a feeder; she walks through it and pecks as she passes.

That gives a two-step mechanism with no new machinery at all:

1. She keeps walking, because nothing about the gakel call touches her walking any more.
2. She does not eat there, so hunger stays high, so `M_FORWARD` is driven *harder* — she
   leaves faster than a fed hen would.

The aversive response becomes "decline to eat here", and departure falls out of the
foraging dynamics that already exist. Nothing needs a bearing, which was the original
docstring's reason for rejecting a turn and remains true — the audio channels carry no
direction (E064's premise), and E064's location cue gives a *place*, not a heading.

---

## 3. Prediction

Run E082's control **unchanged** — same seeds, same discriminative plant, same
`pred_gain` ladder, same arena, same pre-flight — against the redesigned anchor.

1. **Occupancy at the planted feeder P falls monotonically with `pred_gain`**, and by
   more than E082's 3.6% relative (0.4501 → 0.4339, non-monotonic). I predict a fall of
   **at least 15% relative** at `pred_gain=2.0`, i.e. occupancy ≤ 0.383.
2. **Occupancy at the control feeder P′ does not fall**, and plausibly rises, since a hen
   declining to feed at P has to feed somewhere.
3. **Forward drive no longer falls with gain.** E082's 17% drop was the defect. Under the
   redesign `M_FORWARD` receives nothing from the gakel channel, so any residual movement
   in `fwd` is indirect (via hunger) and should point *upward* if anything.
4. **Peck rate at P falls with gain**, which is now the direct read on the reflex firing.
5. **The ethogram assay still passes**, rewritten to test peck suppression and to assert
   that forward drive is *not* suppressed.

### What would make this uninteresting even if it "works"

If occupancy at P falls only because the hen is starving and moving faster everywhere,
that is not avoidance, it is agitation. Prediction 2 is the discriminator: agitation
lowers occupancy at *both* feeders, avoidance lowers it at P and not at P′.

---

## 4. Falsifier

**Primary.** Occupancy at P does not fall by ≥15% relative at `pred_gain=2.0`, or falls
non-monotonically across the ladder. Mechanism 1 is then not merely mis-signed; declining
to eat is insufficient to produce departure, and the anchor needs a genuinely new motor
consequence rather than a subtraction.

**Agitation falsifier.** Occupancy at P′ falls by ≥10% relative alongside P. She is not
avoiding a place, she is moving more. This is the one I consider most likely to fire,
because the mechanism deliberately routes departure through hunger.

**Starvation falsifier.** Mean hunger at `pred_gain=2.0` exceeds **0.60** (baseline
~0.43). The redesign trades eating for leaving, and a hen who cannot feed at all has not
learned to avoid one feeder, she has learned not to eat. Note this is a *changed*
threshold from E082, where any hunger rise was read as hallucination: here a modest rise
is the mechanism working as designed, and only a large one is disqualifying.

**Reflex falsifier.** The pre-flight fails, or live `pred@gakel` at P drops below 0.80.
The plant must fire as hard as it did in E082 or the comparison is not matched.

---

## 5. Design

### The change

In `hen/innate.py`'s `_add_gakel_scaffold`, delete

```python
w(spec.M_FORWARD, gakel_call, -SCAFFOLD_WEIGHT * gain)
```

and keep the `M_PECK` line at the same weight. Rewrite the docstring's "why suppression
of approach rather than a turn" paragraph, which is now wrong on its own terms, to record
E082's finding and the hunger-mediated departure mechanism.

`gakel_scaffold` stays **off by default**, as does `place_cells_enabled` and
`contamination_enabled`. Nothing outside T2 changes.

### The assay

`run/probes.py`'s `withdraw_on_hearing_a_gakel_call` currently requires `fwd_g < fwd_c`
**and** `peck_g < peck_c`. Rewrite as:

- `peck_g < peck_c` — the response fires, and specifically for this call rather than for
  any call (the contact-call contrast is retained unchanged; it is the part of this assay
  that was always doing the real work).
- `fwd_g >= fwd_c - tol` — a **guard** that the freeze has not come back. This is the
  configuration where the defect appeared, which is where `CLAUDE.md` requires the guard
  to run.

### The run

`scratchpad/e083_leaving_anchor.py`, forked from `scratchpad/e082_chain_control_redone.py`
with the anchor change and nothing else. Retained verbatim from E082 so the comparison is
matched:

- 4 seeds, `pred_gain ∈ {0.0, 0.5, 1.0, 2.0}`, 20 minutes simulated per cell.
- Planted place P = grid cell 2 at (3.3, 10.0); control P′ = cell 10 at (10.0, 3.3);
  food at both; 3.33 m spacing.
- The discriminative plant into `W_pred`, planted after a **300 s tour-settle** so
  `z_lag_bar` has converged, with the pre-flight assertion E082 added — `pred@gakel` at P
  must read ≥0.99 before the run starts, or it aborts loudly rather than producing a
  quiet null.
- Reported per gain: occupancy P, occupancy P′, hunger, forward drive, **peck rate at P**
  (new — the direct read on the reflex), and live `pred@gakel`.

`contamination_enabled` stays **False**: the plant supplies the association, and a live
rotating contaminant would add an uncontrolled second teaching signal. This is a control
for whether the anchor can produce avoidance at all, not a learning run.

### Cost

~12 minutes wall clock, matching E082.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
