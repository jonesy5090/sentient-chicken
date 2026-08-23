# E091 — a call that stops pecking must also stop scratching, or she still cannot look up

*Sections 1–5 written and committed before anything was run. §2's measurements are
design inputs, taken before the design was fixed, and are labelled as such.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**, mechanism 1. Bears directly on **H1a**
(the environment creates an information asymmetry) and on the head-down gate, which
`CLAUDE.md` calls "the whole thesis in one line".

---

## 2. Question

A hen who hears a warning about a place needs to *look* at that place. The head-down gate
makes that conditional on what she is doing: `head_down = max(M_PECK, M_SCRATCH)`
(`spec.HEAD_DOWN_ACTIONS`), and `sensing.py` scales the aerial channel by
`(1 − head_down)`. So whether a call restores her vision depends entirely on which motor
channels the scaffold suppresses.

Measured (design input, hen on food, flockmate calling, hunger 0.5):

| call heard | M_PECK | M_SCRATCH | head_down | vision |
|---|---|---|---|---|
| **pre-E090** — gakel | 0.954 | 0.269 | 0.954 | 4.6% |
| **E090 adopted** — gakel | 0.102 | 0.269 | **0.269** | **73%** |
| E090 adopted — contact (control) | 0.998 | 0.269 | 0.998 | 0.2% |
| E090 adopted — aerial alarm | 0.993 | 0.080 | **0.993** | **1%** |

**E090 incidentally fixed the gakel case**, from 4.6% vision to 73%. That was not designed
for and was not noticed until it was asked about.

Two residuals remain.

**(a) The gakel scaffold suppresses `M_PECK` but not `M_SCRATCH`.** The *alarm* scaffold
suppresses both (`innate.py:256–257`); E083's redesign, which narrowed the gakel response
to pecking alone, never considered scratching. So `head_down` floors at the scratch level
rather than near zero. The arithmetic matches exactly: `M_SCRATCH` = sigmoid(0.5×3.0 +
REST_BIAS) = sigmoid(−1.0) = **0.269**.

**And scratching has no other consequence.** `M_SCRATCH` appears nowhere in `world.py` —
it does not feed her, move her, or deplete anything. Its *only* function in the model is
holding the head-down gate shut. Suppressing it costs nothing and buys vision.

**(b) The aerial alarm scaffold leaves a foraging hen with 1% of her vision.** It
suppresses both head-down channels, but at `SCAFFOLD_WEIGHT = 1.5` against a food drive of
7.0 plus a hunger term — the same saturation problem E089 found for the gakel call. She
crouches, because the call drives `M_CROUCH` directly, but she cannot look up:

| effective alarm weight | head_down | vision |
|---|---|---|
| **1.5 (current)** | **0.993** | **1%** |
| 5.0 | 0.841 | 16% |
| 7.0 | 0.437 | 56% |
| 9.0 | 0.102 | 90% |

**This experiment fixes (a) and only records (b).** E089's lesson was that changing two
reflexes at once confounds the regression gate, and `SCAFFOLD_WEIGHT` is load-bearing for
H4, H2f and E018/E036 — moving it would move recorded results and needs its own experiment
with an H2f reproduction gate.

**Does adding `M_SCRATCH` to the gakel scaffold restore the rest of her vision, at no
behavioural cost?**

---

## 3. Prediction

1. **`head_down` falls from 0.269 to ≈0.102** — the peck level — and vision from 73% to
   **≈90%**. The arithmetic: suppressed scratch is sigmoid(1.5 − 2.5 − 9.0) ≈ 0.00005, so
   `max(peck, scratch)` becomes peck alone.
2. **Nothing else moves.** `M_SCRATCH` has no world effect, so no feeding, movement or
   depletion measure can change. Ethogram stays 13/13.
3. **E090's conditional response is unchanged.** Scratch suppression does not touch
   `M_PECK`, so suppression at hunger 0.2 and 0.8 should be identical to E090's 96.7% and
   72.6%.
4. **The contact-call control stays blind** at ~0.2% vision. This must remain specific to
   the gakel call rather than becoming a response to hearing anything.

## 4. Falsifier

**Primary.** `head_down` under a gakel call does not fall below **0.15**, or vision does
not exceed **85%**.

**Regression falsifier.** Any ethogram assay changes state, or E090's suppression figures
move by more than 2 points. This touches the innate arc.

**Specificity falsifier.** Vision under the *contact* call rises above 5%. The scaffold
must stay tied to this call; a response to hearing anything is what the within-setup
contrast exists to catch.

**Default falsifier.** Behaviour changes with `gakel_scaffold=False`. Asserted
bit-identical.

---

## 5. Design

### The change

In `_add_gakel_scaffold`, add `w(spec.M_PECK, ...)`'s counterpart:

```python
w(spec.M_SCRATCH, gakel_call, -(peck_weight or SCAFFOLD_WEIGHT) * gain)
```

at the same weight, matching what the alarm scaffold already does for both channels. The
docstring's "suppression of ingestion only" reasoning (E083) is updated: the point was
never to leave *scratching* alone, it was to leave *locomotion* alone, and scratching is
not locomotion — it is the other half of the head-down posture.

### Guard

The existing `withdraw_on_hearing_a_gakel_call` assay gains a vision clause: hearing a
gakel call must leave `head_down` below 0.15, and hearing a *contact* call must not. This
runs at the configuration where it matters — a hen on food, which is the only place
`M_PECK` is high enough for the gate to be shut.

### Measurements

1. Head-down and vision under gakel, contact and alarm calls, at the adopted E090 weights.
2. Full ethogram, for the regression falsifier.
3. E090's suppression figures at hunger 0.2 and 0.8, for prediction 3.

### Cost

~2 minutes. This is a one-line change with a guard.

### Recorded, not fixed: the alarm scaffold

§2(b) is written into `docs/backlog.md` as its own item. It is a larger claim than it
looks — if hearing an alarm leaves a foraging hen blind, then the receiver cannot verify
what she was told, which is the precondition for learning what a call *means* (H2c), and
it sits directly on the asymmetry H1a is built from. It needs its own experiment, an H2f
reproduction as the regression gate, and it must not ride along with this one.

---

## 6. Result

Measured at the E090 adopted weights (W=9, H=4), hen on food, flockmate calling:

| condition | M_PECK | M_SCRATCH | head_down | vision |
|---|---|---|---|---|
| gakel, before E091 (hunger 0.5) | 0.102 | 0.269 | 0.269 | 73% |
| **gakel, after E091** (hunger 0.5) | 0.102 | **0.00007** | **0.102** | **90%** |
| gakel, after E091 (hunger 0.2, sated) | 0.033 | ~0 | **0.033** | **97%** |
| contact control, after E091 | 0.998 | 0.269 | 0.998 | 0% |

**All four predictions hold.** `head_down` falls to 0.102 at hunger 0.5 and 0.033 sated,
against a 0.15 falsifier; vision reaches 90–97% against an 85% bar; the contact control
stays fully blind, so the response remains specific to this call; and E090's suppression
figures are unchanged (96.7% sated, 72.6% starving) because scratch suppression does not
touch `M_PECK`.

Ethogram **13/13 at adopted weights** and **88 passed / 1 xfailed** at defaults *(corrected by E096: this line originally read 12/13, which was wrong — the tree's 13/13 is right)* — the gakel
assay remains the registered expected failure at `SCAFFOLD_WEIGHT` 1.5, correctly, since
E090's weights are not yet the default.

**The assay now tests both hunger ends**, implementing the specification E090 §8 recorded
*before* this was measured. A conditional response cannot be validated at one staged
hunger: sated is where the warning should win, starving is where she should override it.
The old assay staged hunger 0.8 only — an arbitrary choice when nothing depended on it,
and load-bearing once conditionality existed. It now requires ≥25% suppression when sated,
strictly less suppression when starving, `head_down < 0.15` under gakel and `> 0.5` under
contact.

## 7. Interpretation

**A warning she cannot look up from is useless, and one channel was holding the gate
shut.** E083 narrowed the gakel response to pecking so it would stop suppressing
*locomotion* — which was right, and which this does not undo. But scratching is not
locomotion; it is the other half of the head-down posture, and `M_SCRATCH` has no other
effect anywhere in the model. Suppressing it costs nothing and returns 17–24 points of
vision.

**E090 had already done most of this by accident.** Raising the gakel weight from 1.5 to
9.0 took vision from 4.6% to 73% as a side effect nobody designed or noticed. E091 is the
remaining quarter, found only because the question was asked directly.

## 8. Consequence

**Adopted**, and guarded: the assay's vision clause runs where it matters — a hen on food,
the only configuration where `M_PECK` is high enough for the gate to be shut.

**Recorded and deliberately not fixed: the aerial alarm scaffold leaves a foraging hen
with 1% of her vision.** It suppresses both head-down channels, but at `SCAFFOLD_WEIGHT`
1.5 against a food drive of 7.0 — E089's saturation problem again. She crouches, because
the call drives `M_CROUCH` directly, but she cannot look up to see what she was warned
about. Raising the effective weight to 7.0 would give 56%, to 9.0 gives 90%.

This is a bigger claim than it looks and it is now a backlog item in its own right. A
receiver who cannot verify what she was told cannot learn what a call *means*, which is
H2c; and it sits directly on the asymmetry H1a is built from. But `SCAFFOLD_WEIGHT` is
load-bearing for H4, H2f and E018/E036, so it needs its own experiment with an H2f
reproduction as the regression gate. E089's lesson — changing two reflexes at once
confounds the gate — applies.
