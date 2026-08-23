# E097 — `W_pred` on the audience task: the rule H2f's falsifier actually named

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2f**, and specifically its registered falsifier, which has never
been tested. Successor to [E096](E096-red-team-review-2026-08.md).

---

## 2. Question

H2f's falsifier reads:

> *a rule closer to Pavlovian association (**e.g. sourced from `W_pred`**, already
> architecturally positioned for this per H2c) succeeds where the instrumental rule failed,
> on the same task and the same scaffold.*

What was implemented and scored against it was `hebbian_readout` — a constant modulator on
`W_out`, updated from `dz_motor × dz_slow`, both same-time traces (`tau_motor` 0.10,
`tau_slow` 0.20). It shares one property with the falsifier's description, *not
reward-gated*, and lacks the distinguishing one: there is no cue→outcome direction in it at
all.

**`W_pred` is the pathway the falsifier names and it has never been run on this task.** It
is a masked delta rule (`pred_err = observability × (obs − predicted)`), non-reward-gated,
sourced from `z_lag` at `tau_lag` 1.5 — the only genuine lag in the codebase — and it
writes onto the observation the *reflex arc* reads (`brain.py:85`), so a learned cue does
not have to recreate a behaviour, only the percept that already drives it.

**There is a specific mechanism by which it could produce an audience effect**, and stating
it in advance is what makes the falsifier meaningful: if "flockmates in view" reliably
precedes "aerial alarm audio" during rearing, `W_pred` can learn to *hallucinate the alarm
channel when flockmates are present*. `innate.py` wires `IDX_AERIAL → M_CALL_AERIAL` at
7.0, so that hallucination drives alarm calling. Cue → predicted outcome → innate response
is precisely Pavlovian, and it is available to `W_pred` and unavailable to
`hebbian_readout`.

**And E096 changes how this must be measured.** The audience assay is confounded: the
staged hawk sits 5–9 m from the audience, so they alarm-call, and 79% of H2f's
audience-specific effect is carried by that audio rather than by their presence. **The
primary measure here is therefore the *muted* contrast** — audience present but silent —
which is the quantity the audience effect was always supposed to be.

That is also the sharp test of the mechanism above. A prediction learned from "flockmates
present" is *internal*: it should survive muting, because she is generating the percept
herself rather than hearing it.

---

## 3. Prediction

1. **Muted audience-specific DiD ≥ +0.10.** With the audience present but silent, a
   `W_pred`-reared flock calls more than an alone flock, beyond the food-channel control.
   E096 measured `hebbian_readout` at **+0.0577** muted; the bar is roughly double that.
2. **Muted DiD is a larger fraction of intact DiD than `hebbian_readout`'s 21%.** If the
   effect is internally generated it should depend less on the audio.
3. **The food-channel control stays below +0.05 muted.** Specificity: the effect must be
   about calling in company, not about calling more in general.
4. **`pred_gain` matters.** At `pred_gain=0` the pathway is wired but cannot reach
   perception, so the DiD should fall back toward the fixed-hen baseline. This is the
   internal control that the effect is `W_pred`'s and not the scaffold's.

I hold prediction 1 at roughly even odds. The mechanism is available and the pathway is
built for it, but no learning rule in this project has yet produced a targeted behavioural
change, and `W_pred`'s own arc (E087/E088) showed it needs a well-conditioned readout to
carry anything.

## 4. Falsifier

**Primary.** Muted audience-specific DiD < **+0.10**. H2f's falsifier then has been
attempted and not met: the Pavlovian route does not produce an audience effect either, and
H2f should be read as `NOT SUPPORTED` on its own terms rather than `SUPPORTED, CONFOUNDED`
— because the confounded result would be the only thing holding it up and the rule it
actually named would have failed.

**Specificity falsifier.** Food-channel muted DiD ≥ +0.05. Indiscriminate elevation, which
is exactly what E058 used to reject H2c's apparent result.

**Gain falsifier.** The muted DiD at `pred_gain=0` is within 0.03 of the value at
`pred_gain>0`. The effect would then not be `W_pred`'s at all.

**Reproduction falsifier.** The `hebbian_readout` arm does not reproduce E096's muted
+0.0577 within 0.03. That arm is run alongside as a positive control on the harness; if it
does not reproduce, nothing else in the run is interpretable.

---

## 5. Design

Four arms, matched seeds, 8 seeds, 30 minutes' rearing at E057's configuration
(`food_deplete_rate=0.0`, auditory scaffold on):

| arm | rule | purpose |
|---|---|---|
| **S** | plasticity off | baseline |
| **H** | `hebbian_readout`, `readout_scaling_strength=0.3` | E096 reproduction / harness control |
| **P0** | `pred_enabled`, `pred_gain=0.0` | `W_pred` wired but unable to reach perception |
| **P** | `pred_enabled`, `pred_gain=1.0`, `pred_centred`, `pred_bar_freeze_s=60` | the falsifier's rule |

E088's adopted centring configuration is used for the `W_pred` arms, because E087 measured
the tracking baseline removing ~20 points of the signal `W_pred` reads, and running the
falsifier's rule through a readout known to be degraded would not be a fair test of it.

**Every arm is assayed twice on the same reared brain** — audio intact and audio muted at
test — and **the muted contrast is the primary measure**. Reporting both keeps
comparability with E057 while making the confound visible rather than inherited.

Reported per arm: `alarm_alone`, `alarm_audience`, `food_alone`, `food_audience`, each
intact and muted; the DiD of each; and the fraction surviving muting.

### Cost

~25 minutes.

---

## 6. Result

8 matched seeds, 30 min rearing, each brain assayed twice without re-rearing. 1280 s.

| arm | audio | alarm_alone | alarm_aud | food_alone | food_aud | **DiD** |
|---|---|---|---|---|---|---|
| **S** (no plasticity) | intact | 0.3196 | 0.3853 | 0.5005 | 0.5013 | **+0.0650** |
| S | MUTED | 0.3195 | 0.3224 | 0.5005 | 0.5014 | **+0.0020** |
| **H** (`hebbian_readout`) | intact | 0.4475 | 0.7496 | 0.5150 | 0.5277 | **+0.2894** |
| H | MUTED | 0.4206 | 0.5123 | 0.5143 | 0.5281 | **+0.0779** |
| **P0** (`W_pred`, gain 0) | intact | 0.2298 | 0.2397 | 0.5463 | 0.5519 | +0.0042 |
| P0 | MUTED | 0.2258 | 0.2088 | 0.5444 | 0.5511 | −0.0237 |
| **P** (`W_pred`, gain 1) | intact | 0.1248 | 0.1325 | 0.5441 | 0.5619 | −0.0101 |
| P | MUTED | 0.1247 | 0.1206 | 0.5423 | 0.5619 | **−0.0236** |

Paired, df=7, t crit 2.365. Muted DiD: S +0.0020 (t=0.74), **H +0.0779 (t=2.14)**,
P0 −0.0237 (t=0.89), **P −0.0236 (t=1.01)** — none significant.

**Primary falsifier FIRES.** Arm P's muted DiD is **−0.0236** — not merely short of +0.10
but wrong-signed. **Gain falsifier FIRES**: P and P0 differ by +0.0000. Specificity clear
(food +0.0196). **Reproduction clear**: H's muted +0.0779 against E096's +0.0577, inside
the window, so the harness works.

**The instrument is sound.** Arm S reproduces E057's published S row to four decimals
(0.3196 / 0.3853 / 0.5005 / 0.5013 against 0.3197 / 0.3852 / 0.5005 / 0.5013).

### 6b. Two defects that make the primary null uninformative

**(1) The rule was trained through one signal and tested through another.** During rearing
`W_pred` sources from centred, frozen-baseline `z_lag` (`tau_lag` 1.5). During the assay
plasticity is **off**, so `ps.z_lag` is never updated, `_one_step` passes `pred_from=None`,
and the projection is read out from **instantaneous, uncentred `rate(x)`**. §5 adopted
E088's centring specifically so the falsifier's rule would not be tested through a degraded
readout — and the readout at test is the uncentred one regardless.

**This is E071's error for the fourth time in this project** (E071 timescales, E082
centring, E093 fit-versus-runtime space, now train-versus-test source). A quantity measured
in one regime and read in another.

**(2) The rearing world may not contain the contingency.** At `hawk_period_s=900`, 30
minutes contains roughly **two hawk events**. §2's mechanism requires "flockmates in view
reliably precedes aerial alarm audio" to be *learnable*. Two co-occurrences is thin, and
nothing here measured the actual pairing rate. **A null from `W_pred` may be a statement
about the rearing world's event rate rather than about the rule.**

Also noted: `run/audience.py:assay` runs at `NO_PLASTICITY`'s default `pred_gain=1.0` in
every arm, so §3's prediction 4 ("at `pred_gain=0` the pathway cannot reach perception") is
not what the harness executes — gain 0 is a *rearing-time* property only. The gain falsifier
fires on both readings, so that verdict stands, but not for the reason §3 gave.

### 6c. The solid finding, and it is not about `W_pred`

**Arm H's muted effect is not significant on 8 seeds**: +0.0779 ± 0.0363, **t=2.14 against
2.365**, per-seed values from −0.0552 to +0.2346. E096's +0.0577 came from **3** seeds.

So the 21%-survives remnant that H2f's `SUPPORTED, CONFOUNDED` status rests on is, on a
proper sample, **not distinguishable from zero.**

**And the confound predates learning entirely.** The unlearned S arm shows an intact-audio
DiD of **+0.0650** collapsing to **+0.0020** muted. The staging drives an "audience effect"
in a hen who has learned nothing at all.

## 7. Interpretation

**H2f's falsifier has been attempted and the attempt is inconclusive.** The primary fired,
but §6b gives two reasons the null carries little information: the rule was tested through
a different signal than it was trained on, and the rearing world may not have contained the
contingency often enough to learn. `CLAUDE.md`'s standing rule applies — *a null is only
informative if the instrument could have shown a positive* — and neither condition was
checked before running. **This does not license reading H2f as `NOT SUPPORTED` on its own
terms**, which is what §4 said a primary firing would mean. That reading is withdrawn.

**What is established is about H2f itself, from the reproduction arm.** The audience effect
is real and large *with audio* (+0.2894) and **not significant without it** (t=2.14, n=8).
Combined with S's unlearned +0.0650 → +0.0020, the honest statement is:

> **The learned change is a response to hearing calls, not to having an audience.**

That is a *positive* statement about what the rule does, and it is not nothing:
`hen/innate.py:259-263` deliberately declined to wire a call relay, on the stated grounds
that one "would confound the audience assay". **Learning built the relay the innate arc
refused to build.** H2f asked whether the rule was the wrong *kind*; the answer visible
here is that this rule learns the correlation the environment actually supplies — flockmate
calls predict flockmate calls — rather than the contingency the hypothesis wanted.

**`W_pred` also suppresses alarm calling by 61%** (P's `alarm_alone` 0.1248 against S's
0.3196), which is the opposite of §2's predicted direction. Since `relu(predicted)` can only
*add* percepts, the suppression must come from added activity on channels whose reflex
weights oppose aerial calling. Whatever it learned is broad rather than the cue→outcome
association the falsifier described — consistent with §6b(1), since the projection was read
through an uncentred signal it was never trained against.

## 8. Consequence

**H2f moves to `NOT SUPPORTED` on the audience-conditional claim, and gains a new
supported one.** The audience-conditional reading has no support at n=8. What the rule
demonstrably learns is a call relay. The node needs rewriting around that rather than
around a status word.

**E097's own primary is recorded as inconclusive, not as a null.** The falsifier fired and
the fired verdict is withdrawn on the grounds in §6b. That is written here rather than
argued later.

**Two fixes before `W_pred` can be fairly tested, and they are the next experiment:**

1. **Make the assay read the signal the rule was trained on.** The projection must be
   sourced from the same centred, lagged trace at test as during rearing. This is a defect
   in `run/audience.py` / `run/simulate.py`'s plasticity-off path, not in the experiment.
2. **Measure the pairing rate before concluding anything from a `W_pred` null.** Count how
   often "flockmates in view" precedes "aerial alarm audio" during rearing, and if it is
   ~2 events, rear at a shorter `hawk_period_s` so the contingency exists to be learned.

**And the audience assay needs its confound fixed at the source.** S's +0.0650 → +0.0020
shows the staging manufactures an audience effect in an unlearned hen. Moving the staged
hawk out of the audience's `vision_range`, or muting by default, would make every future
audience number mean what it says.
