# E098 — repair the audience assay, then give `W_pred` a fair test

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2f** and **H3**, both of which are measured through
`run/audience.py`. Successor to [E097](E097-wpred-on-the-audience-task.md).

---

## 2. Question

E097 left three defects, all in the instrument rather than the hypothesis.

**(a) The assay cannot read what the rule learned.** Two independent routes:
`run/audience.py:101-112`'s `assay()` calls `simulate.rollout` **without passing the reared
`ps`**, so a fresh zero-filled trace state is built; and `run/simulate.py:102`'s
`if not pc.enabled: return` sits **above** `update_traces`, so `z_lag` could not update even
if it were passed. A `W_pred` projection trained on a centred lagged trace is therefore read
at test either from zeros or from instantaneous `rate(x)`, depending on `pred_enabled`.
**E071's error for the fourth time**, and this one silently nullifies the pathway.

**(b) The staging manufactures an audience effect in an unlearned hen.** E097's
plasticity-off arm shows intact DiD **+0.0650** collapsing to **+0.0020** when the audience
is muted. The staged hawk sits 5–9 m from the audience, inside `vision_range`, so they
alarm-call and "audience present" is also "aerial channel driven". Every audience number
this project has reported inherits it.

**(c) A `W_pred` null may be about the world, not the rule.** At `hawk_period_s=900`, 30
minutes of rearing contains roughly **two** hawk events. The mechanism E097 pre-registered
needs "flockmates in view precedes aerial alarm audio" to be learnable, and the pairing rate
has never been measured.

**Does a repaired assay, on a world that contains the contingency, let `W_pred` show an
audience effect?**

---

## 3. Prediction

1. **The trace fix is behaviourally inert where it should be.** Plasticity-off runs without
   `pred_enabled` are **bit-identical** before and after. Traces are state, not learning;
   updating them must change nothing that does not read them.
2. **The fix is not inert for `W_pred`.** With the reared `ps` carried and traces
   maintained, arm P's assay differs measurably from E097's — the projection now reads the
   signal it was trained on. No direction predicted.
3. **The pairing rate at `hawk_period_s=900` is too low to learn from**: fewer than 5
   flockmate-call events per rearing run. At 60 s it should exceed 20.
4. **Muted DiD ≥ +0.10 for `W_pred` on the repaired assay at the higher event rate** — the
   bar E097 set and could not fairly test. Held at **below** even odds: E097's arm P
   *suppressed* alarm calling by 61%, and while that is consistent with reading an untrained
   signal, it is not evidence the trained one would help.

## 4. Falsifier

**Inertness falsifier (a hard gate).** Any plasticity-off, non-`pred` result changes after
the trace fix. Then the fix is not a fix, it is a change to every recorded result, and it
must be reverted and redesigned. Checked by asserting bit-identity on the full suite and on
arm S's four assay numbers.

**Reachability falsifier.** Arm P's numbers are unchanged after the fix. The projection
would then still not be reading the trained signal, and (a) is not the defect it appears to
be.

**Primary.** Muted audience-specific DiD < +0.10 for `W_pred` on the repaired assay at a
rearing world where the pairing occurs >20 times. H2f's falsifier would then have been
fairly attempted and not met, and **that reading is licensed this time** because both §2(a)
and §2(c) will have been removed.

**Specificity falsifier.** Food-channel muted DiD ≥ +0.05.

---

## 5. Design

### The three repairs

1. **Move `update_traces` above the `pc.enabled` early return** in `run/simulate.py`, so
   trace state is maintained whenever the pathway that reads it is enabled, and only
   *weight* updates stay gated on `enabled`. Traces are state; learning is the weight
   change. Guarded by prediction 1's bit-identity assertion.
2. **`assay()` accepts and carries the reared `ps`.** Without it the fix above is moot.
3. **Report the muted contrast as the assay's primary number.** `AudienceResult` gains the
   muted quartet and `alarm_effect` becomes the muted one, with the intact figure retained
   as `alarm_effect_audible` for comparability with E057/E074/E096. This is a reporting
   change, not a new measurement — the muted contrast is the one that means what the name
   says.

### The measurement

**Part A — pairing rate.** Count, per rearing run, how often a flockmate's aerial call is
audible to the focal hen, at `hawk_period_s ∈ {900, 300, 60}`. This decides Part B's world
and is the check §2(c) says was never done.

**Part B — `W_pred`, fairly.** E097's four arms unchanged, on the repaired assay, at the
`hawk_period_s` Part A selects. 8 seeds, 30 min rearing, each brain assayed twice.

### Cost

Part A ~5 min. Part B ~25 min.

---

## 6. Result

### 6a. The first repair was a no-op, and it was measured rather than assumed

Repair 1 + 2 as landed — trace maintenance above the early return, `ps` carried into
`assay()` — **changed nothing.** Arm P is **8 of 8 bit-identical to `e097_cache.json`**,
intact and muted.

`assay()` calls `simulate.rollout` at its default `NO_PLASTICITY`, whose `pred_enabled` is
**False**. So `_one_step` leaves `pred_from=None` and `brain.step` sources the projection
from instantaneous `rate(x)` — *and* the newly-moved trace update is itself gated on
`pred_enabled`, so it is skipped too. **The reared `ps` rides in the scan carry unread.**

Both halves of the defect were real. The fix addressed neither, because it never asked how
the assay *sources* the prediction. The complete repair is a third change — `assay()` must
also accept the assay-time `pc` — and is now made. Learning still cannot occur: `W_pred` is
written only in `consolidate`, gated on `enabled`.

**Inertness re-verified after the completed repair: bit-identical, 89 passed / 1 xfailed.**

### 6b. Part A — the pairing rate, and something larger

| `hawk_period_s` | hawk events | paired with an audible alarm | % | **paired *and* she was blind** |
|---|---|---|---|---|
| 900 | 1.0 | 1.0 | 100% | — |
| 300 | 3.7 | 3.7 | 100% | — |
| 60 | 18.3 | 18.3 | 100% | **≈0.3** |

Prediction 3 is half-met: <5 at 900 ✓ (1.0), >20 at 60 ✗ (18.3, per-seed 16/21/18). On a
chorus threshold it reads 36.0 and clears. 60 s selected either way.

**But the pairing rate was the wrong thing to worry about.** It is **100% at every period**
— every hawk event produces an audible flockmate alarm. What varies is only how often hawks
arrive. And of 18.3 paired events, roughly **0.3** occur while the focal hen is blind.

**~~So in ~1.6% of hawk events does a call tell her anything she could not already see.~~** **STRUCK by [E099](E099-does-the-environment-supply-the-asymmetry.md).** This measured blindness at the hawk's *onset step*, which is near-zero by construction because nobody has called yet. Re-measured: **90.4%** of hawk events contain a moment when she is blind while a call sounds, and she is blind for **47%** of all call-time. The environment supplies the asymmetry.

### 6c. Part B — `W_pred` read through the signal it was trained on

8 seeds, 30 min, `hawk_period_s=60`. Muted is the primary measure; t(7) crit **2.365**.

| arm | DiD intact | DiD muted | ±SE | survives | t (muted) |
|---|---|---|---|---|---|
| S (no plasticity) | +0.0650 | +0.0020 | 0.0028 | 3% | 0.74 |
| H (`hebbian_readout`) | +0.2637 | +0.0580 | 0.0486 | 22% | 1.19 |
| P0 (`W_pred`, gain 0) | +0.1771 | **+0.0908** | 0.0393 | 51% | 2.31 |
| P (`W_pred`, gain 1) | +0.1746 | **+0.0815** | 0.0359 | 47% | 2.27 |

**Nothing is significant.** Every muted DiD sits below the bar.

- **Primary FIRES** — arm P muted **+0.0815 < +0.10**, and not distinguishable from zero.
- **Reachability CLEAR** on the corrected read (P's intact DiD −0.0101 → +0.1258), and it
  would have **FIRED** on the as-landed repair, which is what exposed 6a.
- **Specificity CLEAR** — food-channel muted effect **−0.0291**, negative.
- **Inertness CLEAR** — 8/8 bit-identical.
- **Arm H reproduces E096 exactly**: muted +0.0580 against +0.0577.

**And P0 ≈ P** (+0.0908 against +0.0815). E097's gain falsifier fires again: the arm whose
projection never shaped rearing produces *slightly more* of the effect. **Whatever this is,
it is not `W_pred`'s.**

### 6d. Two things the corrected read changes about E097

**The world mattered more than the fix.** Arm P's muted DiD moves −0.0236 → +0.1136 on the
*unrepaired* read simply by rearing at 60 s instead of 900 s. §2(c) was the larger defect.

**E097's numbers carried a spurious top-down boost.** Correcting the source drops
`alarm_alone` from ~0.49 to ~0.27: the centred lagged trace yields a far smaller
`relu(predicted)` than instantaneous `rate(x)`, so E097 was measuring a hen whose perception
was being inflated by a projection read through the wrong signal.

## 7. Interpretation

**H2f's falsifier has now been fairly attempted, and it is not met.** Both grounds E097
gave for withholding that reading are removed: the rule is read through the signal it was
trained on, and the world supplies the pairing 18 times per run instead of one. `W_pred`'s
muted audience effect is +0.0815, below the bar and not significant — **and identical to
the arm where the projection never shaped rearing at all.**

**No rule in this project has produced a targeted behavioural change.** `hebbian_readout`
produces a large one that is 78% a call relay. `W_pred` produces one indistinguishable from
its own gain-0 control. Both are consistent with rules that learn whatever correlation the
environment supplies rather than the contingency a hypothesis wants.

**The finding I did not expect is 6b's last line.** The project's founding premise is an
information asymmetry: a head-down hen cannot see the hawk, so a call is worth making.
~~Measured at the rearing configuration, ~1.6% of hawk events are ones where a call carries information she does not already have.~~ **STRUCK by E099** -- that figure measured blindness at the hawk's onset step. The environment *does* supply the asymmetry: **90.4%** of hawk events contain a blind-and-called moment, and she is blind for **47%** of call-time. This makes the learning results harder, not easier: no environmental excuse remains for them.

**My own error is the most instructive part of the repair.** I diagnosed two real defects,
fixed both, verified inertness, and shipped a change that did nothing — because I never
asked how the assay *sources* the prediction, only whether the state was present and
maintained. The bit-identity check that caught it was not one I designed; it was a
reachability falsifier written to test something else.

## 8. Consequence

**H2f: the falsifier is met and not satisfied.** The node's own test has been run properly
and the Pavlovian route does not produce an audience effect either. H2f stays `NOT
SUPPORTED` on the audience-conditional claim, and the "wrong *kind* of rule" hypothesis is
now **tested and negative** rather than untested.

**The assay is genuinely repaired** — `assay(p, cfg, n_hens, steps, ps, pc)` — with
inertness verified twice.

**The next question is not about a rule.** It is whether this environment supplies the
contingency any rule would need. Before another learning experiment, measure and if
necessary fix the asymmetry: at present a hen is almost never blind when a call arrives.
Candidates are a shorter `hear_range`, a wider arena, more head-down time, or a predator
class that is only sometimes announced.

**Recorded for the tree: `docs/hypothesis.md`'s H1a is `SUPPORTED` on head-down blindness
(64% of the time), and that is not the same claim as "calls carry information".** Those two
have been treated as one.
