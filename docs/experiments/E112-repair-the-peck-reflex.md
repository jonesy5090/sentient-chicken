# E112 — repair the peck reflex, then re-ask H2

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, and **H1** (the innate arc). A change to the *hardwired*
arc, not to learning, made because the arc is wrong.

---

## 2. Question

[E111](E111-is-there-headroom.md) measured a defect in the innate wiring:

| | on food | off food |
|---|---|---|
| `M_PECK > 0.5` | **39.65%** | **59.59%** |

**She pecks less when she is standing on food.** `innate.py:84-85` drives `M_PECK` from
the `CLS_FOOD` channels of the **two front bins only** — `_FRONT = (5, 6)`, the pair
straddling the beak — at weight 7.0. That is the right idea for a hen approaching food
from a distance and the wrong one for a hen standing on it: at 0.30 m the bearing to the
patch swings wildly with every step, so the patch spends much of the dwell in a bin the
peck reflex does not read, or behind her entirely. Meanwhile `vision_range = 10 m` in a
20 m coop keeps the same channels near-saturated everywhere else — 0.9430 summed off food
against 0.9100 on it — so the reflex fires constantly and at nothing in particular.

`innate.py:305` already describes half of this: "A hen does not stop at a feeder, she
walks through it and pecks as she passes." What it does not say is that the pecking is not
aimed either.

**And the model already has the signal that would fix it.** `IDX_FOOD_ARRIVAL`, the E053
discovery pulse, is set to 1.0 on the rising edge of arriving at a patch and decays over
`food_call_decay_s = 4 s`. It is bearing-independent, it carries "am I at a feeder" at
**AUC 0.87–0.99** ([E107](E107-red-team-review-2026-08-24.md)), and it is wired to the
food **call** and to nothing else. A hen announces that she has found food and does not
act on it.

**So: does repairing the arc close the 0.21-hunger gap E111 measured — and does anything
remain for learning to do afterwards?**

This is a change to the **hardwired** arc. `CLAUDE.md` says the reflex arc is fixed for
life and never plastic; it does not say it must stay *wrong*. But it moves every baseline
in the project, which is why it is opt-in and pre-registered rather than simply corrected.

## 3. Prediction

1. **The repair works mechanically.** `P(peck | on food)` rises from 39.65% to **above
   80%**, and the sign of the on-food/off-food difference reverses.
2. **It closes much of the gap.** Repaired hunger falls from 0.6332 to **0.50–0.55**,
   against the camped oracle's 0.4223.
3. **Headroom persists but shrinks** — I expect roughly half of E111's 0.21 to remain,
   since the oracle also *stays* on the patch and the repair only makes her peck while she
   is there.
4. **Learning still adds nothing on top.** [E110](E110-postsynaptic-factor.md) found no
   benefit over a frozen readout and I expect that to survive the repair, because
   [E109](E109-what-the-rule-writes.md)'s constraint is unchanged by it.

## 4. Falsifier

**Instrument falsifier — reported before anything else.** `P(peck | on food)` must exceed
80%. If the wiring does not change the behaviour it targets, no hunger number in this
experiment means anything.

**Regression falsifier.** Inertness bit-identical at `arrival_peck_weight = 0.0`; the
`peck_at_food` ethogram assay must still pass; no other assay may change state; the suite
must pass.

**Interpretation guard, written in advance so neither outcome can be spun afterwards.**

- If the repair closes **most** of the gap and learning still adds nothing, then what ten
  experiments were asking the learning rule to discover was **a defect in the hand-written
  wiring**. That is the uncomfortable outcome and it is the one I would report first.
- If the repair closes **little** of the gap, the arc was not the limitation, E111's
  headroom stands undiminished, and the nulls remain squarely about the rule.
- If the repair **hurts** — plausible, since a hen who stops to peck is a hen not
  travelling to the next patch — that is a real finding about the foraging trade-off and
  the flag stays off.

**Replication rule.** E021: nothing moves the tree on one seed block.

## 5. Design

**`arrival_peck_weight: float = 0.0`** in `connectome.build`, passed through to
`innate.reflex_matrix`, adding `w(M_PECK, IDX_FOOD_ARRIVAL, weight)`. Off by default, so
every existing result is untouched.

**The weight is chosen a priori against `REST_BIAS`**, in the same way and for the same
reason `SCAFFOLD_WEIGHT` is, so it cannot be tuned into the shape of a result:

- **4.0**. `sigmoid(4.0 − 2.5) = 0.82`, so arriving at a patch reliably crosses the 0.5
  action threshold on its own, without needing the visual channels to cooperate.
- It decays below threshold when the pulse falls under `2.5/4.0 = 0.625`, which at
  `food_call_decay_s = 4 s` is **1.5 s after arrival**. So it buys a *peck bout* on
  arrival, not permanent pecking — which is what real hens do and what
  `food_call_decay_s`'s own comment already argues for ("a bout, not continuous").
- It is **half** the visual crouch weight of 8.0, so seeing a hawk still dominates being
  on food. First-hand danger beats appetite, which is the correct ordering and the same
  one `SCAFFOLD_WEIGHT` is justified by.

**Arms**, matched seeds, 8 seeds, 16 hens, 30 min, `hawk_period_s = 60`:

| arm | arc | readout |
|---|---|---|
| baseline / frozen | current | `eta_out = 0` |
| **repaired / frozen** | `arrival_peck_weight = 4.0` | `eta_out = 0` |
| baseline / learning | current | instrumental |
| repaired / learning | `arrival_peck_weight = 4.0` | instrumental |

The camped oracle's **0.4223** from E111 is the reference ceiling; it is not re-run.

**Measured**: `P(peck | on food)` and `P(peck | off food)`; hunger; fraction at a patch
and feeding; `caught/dive`; the full ethogram at the repaired arc.

### Cost

~25 minutes.

---

## 6. Result

### 6a. The instrument falsifier fires — and it was my own inconsistency

8 seeds, 30 min. `scratchpad/e112_repair_peck.py`.

| arm | hunger | P(peck \| on food) | P(peck \| off) | at patch | feeding |
|---|---|---|---|---|---|
| baseline / frozen | 0.6332 | **41.8%** | 62.6% | 6.4% | 2.6% |
| **repaired / frozen** | 0.5761 | **75.3%** | 63.8% | 4.1% | 3.1% |
| baseline / learning | 0.6268 | 41.5% | 62.9% | 6.5% | 2.7% |
| repaired / learning | 0.5584 | 71.9% | 63.2% | 4.4% | 3.1% |

§4 required `P(peck | on food)` above **80%**. It reached **75.3%**, so **the instrument
falsifier fires** and by my own rule nothing downstream of it counts until the cause is
found.

The sign did reverse — 75.3% on food against 63.8% off, where the baseline had it
backwards — so the wiring does what it says. What caps it is not the wiring.

**Measured cause** (`scratchpad/e112b_stop_to_eat.py` and a dwell probe): **median dwell
at a patch is one step.** Mean 0.69 s, median 0.01 s. Half of all on-food episodes are a
single 10 ms step, and `world.step` sets the arrival pulse *after* that step's motor
output has already been computed — so on a one-step visit she cannot peck at all, however
the reflex is wired.

**She walks through the feeder.** `innate.py:305` says exactly this and
`peck_stops_walking` (E092) exists for exactly this, and has been off by default since it
was built. E111's camped oracle *stayed* on its patch; that was half of what it did, and
this experiment only built the other half.

### 6b. Post-hoc — the second half of the fix

**Labelled post-hoc: this arm was chosen after seeing §6a**, and it runs on a disjoint
seed block (8–15), which also replicates §6a's borderline headline.

| arm | hunger | P(peck \| on food) | at patch | feeding |
|---|---|---|---|---|
| baseline | 0.6557 | 39.7% | 6.4% | 2.5% |
| repaired peck | 0.6196 | 77.2% | 3.7% | 2.8% |
| **repaired + stops walking** | **0.5939** | **97.9%** | 3.3% | 3.3% |
| repaired + stops + learning | 0.5953 | 97.7% | 3.4% | 3.3% |

**With both halves the instrument clears completely: 39.7% → 97.9%.** The diagnosis was
right.

| contrast (hunger, lower is better) | | |
|---|---|---|
| repair alone vs baseline | −0.0360 ± 0.0223 | t=−1.62 |
| repair + stopping vs baseline | −0.0618 ± 0.0245 | t=−2.53 |
| what stopping adds on top | −0.0258 ± 0.0143 | t=−1.80 |
| **what learning adds on top of both** | **+0.0014 ± 0.0124** | **t=+0.12** |

**Nothing here is significant under this experiment's own multiplicity rule.** Four
contrasts, so the Bonferroni threshold at df=7 is **t≈3.5**; the largest is 2.53. And the
repair-alone effect does not replicate across blocks — −0.0571 (t=−2.28) then −0.0360
(t=−1.62), both short of even the uncorrected 2.365.

### 6c. How much of the gap is reachable by fixing the arc

Against E111's camped oracle at **0.4223**:

| | hunger | gap closed |
|---|---|---|
| baseline | 0.6332 / 0.6557 | — |
| repaired peck | 0.5761 / 0.6196 | 27% / 15% |
| **repaired + stops walking** | **0.5939** | **26%** |

**Roughly three quarters of the gap survives both innate repairs.**

### 6d. Regression

Inertness bit-identical at `arrival_peck_weight = 0.0`, 15/15 digests. Three guard tests,
including one asserting the flag changes **exactly one** entry of the reflex matrix — the
shape of bug E104's `lateral_pool` had — and one asserting a full arrival pulse crosses
the threshold alone while a half-decayed one does not.

## 7. Interpretation

**Take §4's interpretation guard in order.**

> If the repair closes **most** of the gap and learning still adds nothing, then what ten
> experiments were asking the learning rule to discover was a defect in the hand-written
> wiring. That is the uncomfortable outcome and it is the one I would report first.

**That did not happen.** Both innate repairs together close about a quarter of the gap,
and neither is significant under this experiment's own multiplicity correction.

> If the repair closes **little** of the gap, the arc was not the limitation, E111's
> headroom stands undiminished, and the nulls remain squarely about the rule.

**That is the outcome.** E111's headroom is real and largely intact.

**But the arc really is wrong, and that is worth keeping separately from the headline.**
A hen who pecks 39.7% of the time she is standing on food and 62.6% of the time she is
not is not a model of a chicken, whatever it does to the hunger metric. The fix is small,
justified a priori, and now measured: 39.7% → 97.9%. It is worth roughly 0.06 hunger
units, which is real and modest.

**What the remaining three quarters consists of, stated concretely.** The camped oracle
does one thing the repaired hen still cannot: it **stays**. Its hen picks a patch and
remains there for thirty minutes, taking food as it regrows. The repaired hen pecks
correctly when she happens to be on a patch and then wanders off — she is at a patch 3.3%
of the time against the oracle's 4.8%, despite pecking almost perfectly while there.

**Staying is a policy, not a reflex.** It requires holding "I am foraging here" across
seconds, against a hunger drive that keeps her walking, and no fixed stimulus→response
mapping produces it. **It is exactly what a learned pathway is for**, and exactly what
[E109](E109-what-the-rule-writes.md) showed this rule cannot express: persisting in a
place is a redirection of behaviour, and `dz_motor` confines the update to amplifying what
the arc already does.

So E112 does not rescue the learning rule and does not indict it further. It removes a
confound — "maybe the baseline was just broken" — and leaves the target sharper than
before: **the thing learning has never done, and the thing that would close most of the
remaining gap, is staying put.**

**Learning adds nothing on the repaired arc: +0.0014, t=+0.12.** That is the cleanest
version of H2's null in the project, because it is measured against a baseline that has
been repaired to the limit of what its own wiring can express.

## 8. Consequence

**Adopted, off by default.** `arrival_peck_weight = 0.0`. It is off despite being a
genuine correction, because turning it on would move every baseline in the tree and its
measured benefit does not survive multiplicity correction. The recommendation is that any
*future* baseline uses `arrival_peck_weight=4.0` **and** `peck_stops_walking=1.0`
together — separately, each is half a fix.

**`docs/hypothesis.md`.** H1 records that the innate peck reflex is mis-aimed and by how
much. H2's node records that repairing it does not change the null, and that the residual
gap is a *persistence* problem rather than a stimulus–response one.

**Not adopted.** Turning either flag on by default, and any claim that the repair improves
foraging — the effect fails to replicate and fails its own multiplicity threshold.

### Follow-ups

1. **The remaining gap has a name now: staying put.** If anything is tried next on H2, the
   target should be that, and any mechanism proposed for it should be pre-registered
   against E109's constraint — a rule that cannot redirect behaviour cannot produce
   persistence either.
2. **The trained-flock mute** (backlog §5, open since E032) is untouched by all of this
   and tests H0 rather than H2.
3. **E101/E102's permuted-gate control**, still outstanding from
   [E107](E107-red-team-review-2026-08-24.md).
