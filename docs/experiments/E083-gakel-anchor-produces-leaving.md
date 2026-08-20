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

The anchor change landed as intended. 13/13 ethogram, 86/86 suite. In the assay,
forward drive now reads **0.786 under a gakel call and 0.786 under a contact call** —
identical, the channel is disconnected from locomotion entirely — while pecking still
falls, 0.989 → 0.954.

Pre-flight **1.000 on every seed**. 4 seeds, 20 simulated minutes per cell, 885 s wall.

| `pred_gain` | occupancy P | occupancy P′ | hunger | fwd | peck@P | pred@gakel |
|---|---|---|---|---|---|---|
| 0.0 | 0.4246 | 0.2021 | 0.437 | 0.633 | 0.550 | 0.9079 |
| 0.5 | 0.4030 | 0.2495 | 0.432 | 0.621 | 0.533 | 0.9065 |
| 1.0 | 0.4150 | 0.2374 | 0.441 | 0.622 | 0.512 | 0.9510 |
| 2.0 | 0.4627 | 0.1955 | 0.453 | 0.626 | 0.504 | 0.9511 |

Against the pre-registered falsifiers:

- **Primary — FIRES.** Occupancy at P runs **+9.0%** (0.4246 → 0.4627) where ≤−15% was
  required, and non-monotonically.
- **Agitation — clear.** P′ occupancy −3.3%, threshold −10%.
- **Starvation — clear.** Hunger 0.453 at gain 2.0, threshold 0.60.
- **Reflex — clear.** Live `pred@gakel` 0.951, threshold 0.80.

Two things did work exactly as designed. **The freeze is gone**: forward drive is flat
across the ladder (0.633 → 0.626) where E082 had it falling 17%. And **the reflex has
its intended proximal effect**: pecking at the planted feeder falls monotonically,
0.550 → 0.504.

### 6b. The diagnostic that makes the above uninterpretable

`pred@gakel` came back at **0.90 as a run-wide mean while the hen is at P only ~42% of
the time**. A place-selective plant should average near 0.42. That is the wrong shape,
and it prompted a check that neither E082's pre-flight nor this one performs: the
pre-flight measures that the plant fires **at P** and never measures that it is **silent
anywhere else**.

Splitting live `relu(pred@gakel)` by whether each hen is within one grid spacing of P
(`$CLAUDE_JOB_DIR/tmp/plant_live_selectivity.py`, 4 seeds, gain 1.0):

| seed | pre-flight @P | live @P | live elsewhere | ratio |
|---|---|---|---|---|
| 0 | 1.000 | 1.650 | 3.278 | 0.50 |
| 1 | 1.000 | 0.188 | 0.479 | 0.39 |
| 2 | 1.000 | 0.185 | 0.430 | 0.43 |
| 3 | 1.000 | 0.601 | 0.788 | 0.76 |
| **mean** | **1.000** | **0.656** | **1.244** | **0.53** |

The plant is not weakly selective, it is **anti-selective** — it drives the gakel channel
about twice as hard where the hen is *not* meant to be avoiding. Its live magnitude also
varies **9-fold across seeds** (1.650 / 0.188 / 0.185 / 0.601) while pre-flight reads
exactly 1.000 on all four.

Controlling for the possibility that the "at P" disc is simply wider than the place code
— radius 3.33 m against `place_sigma` 2.0 — by profiling against distance instead
(`plant_distance_profile.py`, per-seed normalised so no single seed sets the shape):

| distance from P (m) | relative `pred@gakel` |
|---|---|
| 0.0–1.0 | **0.655** |
| 1.0–2.0 | 0.626 |
| 2.0–3.3 | 0.632 |
| 3.3–5.0 | 1.230 |
| 5.0–7.0 | **2.128** |
| 7.0–10.0 | 0.907 |
| 10.0–99.0 | 0.822 |

The innermost bin — the most on-target position possible — is the **lowest of all seven**,
and the peak sits in a ring 5–7 m away at 3.2× that. The inversion is real and is not an
artefact of the disc radius.

## 7. Interpretation

**The primary falsifier fired, and it must not be counted against the mechanism, because
the instrument was inverted.** E083 did not test whether declining to eat produces
leaving. It tested what happens when you suppress pecking hardest 5–7 m *away* from the
feeder you are trying to make aversive, and weakest while standing on it.

Read that way, the result is not a null at all — **it is the mechanism working correctly
on a signal with the wrong sign.** Occupancy at P rose 9% because the hen was being
pushed off every location *except* P. The same reading rehabilitates E082: its forward
suppression was likewise strongest away from P, and a hen slowed everywhere-but-P
concentrates at P.

**Why the plant inverts: it was measured in one regime and read back in another.** The
discriminant is built from a hen **parked motionless at a grid centre with heading 0**,
settled for 300 s. It is then applied to a hen moving continuously, turning, pecking,
hungry, hearing flockmates. Those pallial states have almost nothing in common, so the
direction that separates *parked at P* from *parked elsewhere* projects onto live states
in a way unrelated to where she is.

**This is the third instance of one error, and `CLAUDE.md` names it exactly** — "a
quantity verified in the place it had just been moved *from*". E071 found the centring
bar measured on one timescale and read on another. E082's first run planted against raw
`z_lag` while the runtime reads `z_lag − z_lag_bar`. This is the same shape again:
measure the discriminant on parked states, read it on moving ones. Each time the
verification was performed and looked at the wrong place.

**And the pre-flight I added *after* E082 does not catch it**, which is the part worth
sitting with. It asserts the plant fires at P. It asserts nothing about elsewhere. That
is precisely the gap that let E024's "shuffled" control retain 98% of the information it
claimed to destroy: a control or a plant must be measured on **what it is supposed to
suppress**, not only on what it is supposed to produce.

### E081's 84.6% does not license what E082 and E083 asked of it

E081 is the experiment that unblocked this route, and its headline number has the same
scope limit (`scratchpad/e081_decodability.py:16`, `e081_place_discriminant.py:17-24`).
It was measured on hens **parked at five cell centres under 0.35 m of jitter**, holding a
single static observation for 200 steps, reading raw `rate(x)`.

The live regime differs on every axis: continuous movement across the whole arena rather
than five discrete points, varying heading, an observation that changes every step, the
lagged-centred trace rather than raw rate, and a 3.33 m radius rather than 0.35 m.

**84.6% is not withdrawn — it is correct for what it measured.** What is withdrawn is the
inference E082 and E083 both drew from it: that a linear readout which separates five
parked point-locations will separate *where the hen is* during free movement. That was
never measured, and the selectivity profile above is the first evidence on it — pointing
the other way.

### What still stands

**The `M_FORWARD` removal is correct and is not affected by any of this.** Its
justification is a reading of `coop/actuation.py` — speed derives from `M_FORWARD`, so
damping it makes a hen already at the aversive place stay there, and suppressing
locomotion is a functional freeze that the arc's own anti-predator clause forbids by
another route. That argument needs no experiment. The assay guard and the connectome
test are likewise sound, and both now run at the configuration where the defect appeared.

## 8. Consequence

**Withdrawn from E082:** its §7 diagnosis that occupancy failed to fall *because* the hen
slows and stays. The mechanism it describes is real in the code, but E082 could not have
observed it, because its own drive was strongest away from P. E082's §6 measurements
stand, and so does "**the chain conducts**" — the plant demonstrably drives the motor
system end to end, and that was E070's open question.

**Withdrawn from E083:** the primary falsifier as evidence about mechanism 1. Whether
declining to eat produces leaving is **still untested**.

**Not withdrawn:** E081's 84.6%, within its stated scope; the anchor redesign; the guards.

**Next, and it is now a prerequisite rather than an option — build the plant from live
states.** Sample `z_lag − z_lag_bar` from the free-running simulation, labelled by
whether the hen is within the target radius of P, and fit the discriminant on *those*.
This is not a refinement of the current plant, it is the only version of it that
addresses the question.

**And the pre-flight must assert selectivity, not just amplitude.** Fires at P **and**
near-silent elsewhere, with the distance profile monotonically decreasing, checked before
any behavioural contrast runs. An amplitude-only pre-flight has now passed twice while
the plant was useless, and both times it produced a confident, wrong, mechanistic story.

**Do not re-run the L vs. C? contrast**, unchanged from E082 — and note that the reason
has moved. It is no longer "the anchor cannot produce the outcome"; it is "the
association we plant is not the association we think we are planting."

**Open question this raises for T2-revised as a whole.** If a discriminant fitted on live
states also fails to separate *at P* from *elsewhere*, then the place code is not linearly
decodable under free movement, and mechanism 2 — the shared allocentric population — is
insufficient as built. That would be a finding about the representation rather than the
plumbing, and the first one in this arc that is genuinely about the hen rather than the
instrument.
