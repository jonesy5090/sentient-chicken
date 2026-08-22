# E086 — routing place cells to the hippocampus, and letting `W_pred` read it

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**, mechanism 2. Direct successor to
[E085](E085-repaired-instrument.md), which established that the blocker is the place
representation rather than the instrument.

---

## 2. Question

E085 measured position as linearly decodable from pallial state at about **four
percentage points above chance while the hen is moving** (54.3% on balanced-split seeds,
t=+2.95), against 84.6% when she is parked. `W_pred` is a linear readout, so that is the
relevant class, and four points is nowhere near enough to bind a place to a call.

The architecture review traced why, and the finding is that **the region built for this
job is not in the circuit**:

- `hen/regions.py:17` names the hippocampus "birds have a real one; place and spatial
  memory". E063 was written up as giving that region "its first real function".
- Measured: of the 64 units receiving place-cell afferents, **64 are in the sensory stub
  and 0 are in the hippocampus**. `W_in` writes only into `SENSORY` (`connectome.py:267`).
- `pred_src` is the pallium alone, units 64–319 (`connectome.py:355`). The hippocampus
  occupies 320–400 and is **excluded**, so even if place information reached it, `W_pred`
  could not source a prediction from it.

So place currently takes this path: 25 place channels, plus 25 gakel-place channels, plus
88 others, all funnelling through a **64-unit sensory stub** at 30% afferent density, then
one more hop (`sensory→pallium`, 0.30) to reach the units `W_pred` can read. The
hippocampus — 80 units, already wired `sensory→hippocampus` 0.20 and `hippocampus→pallium`
0.20 — sits beside that path contributing nothing spatial.

**Does giving place cells a direct hippocampal projection, and letting `W_pred` read the
hippocampus, make position legible while she moves?**

---

## 3. Prediction

1. **Held-out decodability rises above E085's 54.3%.** I predict **≥65%** on
   balanced-split seeds — short of E085's 70% gate, which I am *not* moving, but a large
   enough move to show the mechanism is the right one.
2. **Parked decodability does not fall.** The place code is already legible at rest
   (84.6%, E081); this must not trade one regime for the other.
3. **The ethogram is unchanged.** 13/13. This adds afferents and widens `pred_src`; it
   wires no behaviour and touches no reflex.
4. **The distance profile becomes decreasing** — nearest bin highest — which neither the
   parked-fit nor the live-fit discriminant has ever achieved (E083, E085).
5. **Off by default.** `place_to_hippocampus` joins the seven existing opt-in switches, so
   no prior experiment's baseline moves. This is the lesson E076/E077 cost.

I hold prediction 1 with moderate confidence. Widening `pred_src` and adding a direct
projection both raise position's share of readable variance, but the pallium is 256 units
against the hippocampus's 80, so the pooled readout may still be dominated by non-spatial
drive.

## 4. Falsifier

**Primary.** Held-out balanced accuracy on balanced-split seeds stays below **60%** — i.e.
less than half the distance from E085's 54.3% to the 70% gate. Routing place to its own
region would then not be the binding constraint, and fix (c) from the backlog — widening
the place population — would need trying before any further architectural change.

**Regression falsifier.** Parked decodability falls below 75% (E081 measured 84.6%), or
any ethogram assay fails. The change would then be trading regimes rather than adding
signal, and must be reverted.

**Baseline falsifier.** Any behaviour changes with `place_to_hippocampus=False`. The switch
must be inert when off, checked by asserting bit-identical connectomes.

**Dale falsifier.** The new projection violates Dale's law or the mask, checked by the
existing invariant tests at the configuration where the change appears.

---

## 5. Design

### The change, in three parts

**(a) Direct place afferents to the hippocampus.** `W_in` currently writes only into
`SENSORY`. Add rows for `HIPPOCAMPUS` carrying the place channels — self-location
(`PLACE_LO:PLACE_HI`) and testimony (`GAKEL_PLACE_LO:GAKEL_PLACE_HI`) — with the same
afferent statistics used everywhere else (gamma(2.0, 0.5), 30% density), so this is a
routing change and not a magnitude change.

The `shared_place_map` scaling is preserved: the testimony block keeps the same afferent
pattern as the self-location block, scaled by `testimony_gain`, so E064's "a place is the
same place however you learned of it" still holds in the new region.

**(b) `pred_src` extended to the hippocampus.** `W_pred` may then source predictions from
units that actually carry position. This is the half that makes (a) useful; either alone
is close to pointless.

**(c) Nothing else.** No innate place→motor weights, no place→pallium magnitude change, no
new behaviour. Which places are aversive stays entirely learned, which is the property
E063 was protecting and the reason T2-revised is not question-begging.

### Recorded risk: this re-creates the shape E008 fixed

`pred_src`'s own comment (`connectome.py:348`) says predictions come from the pallium
**never from the sensory stub**, because E008 found the first version circular — sourced
from a region carrying a percept *directly*, `W_pred` learned "when in hawk-state, predict
hawk". Giving the hippocampus direct place afferents and then adding it to `pred_src`
re-creates that structure: a `pred_src` region with direct sensory input.

Three reasons it is still the right change, and one thing it obliges:

1. **The association T2 needs is cross-modal.** `W_pred` writes onto the *gakel call*
   channel and sources from *place*. The gakel call is not an afferent of the hippocampus,
   so predicting it from position is not autoencoding. E008's failure was a channel
   predicting itself.
2. **The source is lagged and centred** — `z_lag − z_lag_bar`, not instantaneous rate
   (E008's own fix, plus E071's). That is the cue-to-outcome direction, not the
   simultaneous one.
3. **E086 does not learn anything.** It plants `W_pred` and measures decodability. The
   circularity risk is not exercised here at all.

**But it will be exercised the moment a learning run follows**, and `shared_place_map`
sharpens it: the testimony channels also feed the hippocampus, so `W_pred` could learn to
predict testimony-about-P from being-at-P, which is circular in exactly E008's sense and
would look like successful association. **Any learning experiment built on this must carry
an autoencoder control** — check that a learned `W_pred` does not simply reconstruct the
place channels from themselves — before its result is read. Recorded here so it is not
discovered later.

### Guard

A test asserting the hippocampus receives place afferents **and** is in `pred_src` when
the switch is on, and neither when off — at `n_hens=16`, the configuration these
experiments run at, per `CLAUDE.md`'s rule that guards run where the defect appears.

### The measurement

`scratchpad/e086_place_to_hippocampus.py`, reusing E085's Part C machinery unchanged so
the comparison is like-for-like: same estimator (difference-of-means), same held-out
protocol (fit on the selection run, evaluate on the test run of the same world), same
per-seed target selection, same 8 seeds, same radius.

Reported: held-out balanced accuracy with `place_to_hippocampus` off and on, overall and
on balanced-split seeds; the selectivity ratio; the distance profile; and a parked-state
decodability check reusing E081's protocol for prediction 2.

### Cost

~20 minutes, matching E085.

---

## 6. Result

8 seeds, both arms, 1088 s. **The off arm reproduces E085 exactly** — 59.6% overall,
54.3% on balanced-split seeds, and identical per-seed values (50.9, 50.6, 57.4, 53.7,
66.4, 84.9, 53.6, 59.6). The comparison is matched.

| | held-out acc | ratio |
|---|---|---|
| `place_to_hippocampus=False` | 59.6% | 1.70 |
| `place_to_hippocampus=True` | 61.8% | 1.38 |

On the six balanced-split seeds (subset taken from the **off** arm, so it is not chosen
on the treatment): **54.3% → 58.9%**, paired change **+4.6 ± 3.3 pts, t=+1.40**, not
significant against t(5)=2.571.

**Primary falsifier fires** (58.9% against a 60% threshold, 65% predicted).

But two other things moved decisively:

**Parked decodability 84.6% → 99.5%.** The off arm reproduces E081's 84.6% exactly; with
routing on, place is very nearly perfectly readable from `pred_src` at rest. The
regression falsifier is clear by a wide margin — this did not trade one regime for the
other.

**The distance profile is decreasing for the first time in this arc** (prediction 4):

| bin (m) | 0–1 | 1–2 | 2–3.3 | 3.3–5 | 5–7 | 7–10 | 10+ |
|---|---|---|---|---|---|---|---|
| off | 0.653 | 1.050 | 1.077 | 0.838 | 1.011 | 1.282 | 1.089 |
| **on** | **1.604** | 1.029 | 0.957 | 0.706 | 0.579 | 0.534 | 1.591 |

The innermost bin goes from the **lowest** of seven to the **highest**, and the profile
falls monotonically out to 7–10 m. Neither the parked-fit (E083) nor the live-fit (E085)
discriminant had ever achieved this. *(The 10+ m bin rises again to 1.591; not
investigated.)*

13/13 ethogram, 86/86 suite, guard test passing.

### 6b. Diagnostic — where the place signal is, and what removes it

*Post-hoc, not pre-registered.* `scratchpad/e086_where_is_the_signal.py`, on identical
data with routing on. Two candidate explanations for why the moving case barely improved:
**dilution** (`pred_src` is now 336 units of which only 80 carry place) and **temporal
filtering** (the runtime reads `z_lag − z_lag_bar`, not raw activity).

| readout | all seeds | balanced |
|---|---|---|
| `pred_src`, 336 units, lagged-centred | 61.8% | 63.5% |
| **hippocampus, 80 units, lagged-centred** | **69.2%** | **73.7%** |
| pallium, 256 units, lagged-centred | 56.4% | 54.4% |
| hippocampus, `z_lag` only (low-pass, τ 1.5 s) | **89.3%** | **90.0%** |
| hippocampus, raw rate (no filtering) | 90.8% | 90.7% |

**Both effects are real and one dominates.** Dilution costs ~10 points: pooling the 80
place-carrying units with 256 that carry almost none (pallium alone: 54.4%, near chance)
drags 73.7% down to 63.5%. **The centring costs ~20 points** — and the lag itself costs
almost nothing, 90.7% → 90.0%.

**I predicted the wrong culprit and the arithmetic that misled me is worth keeping.** I
reasoned the lag would smear position spatially: `tau_lag` is 1.5 s at a walk speed of
0.3 m/s, so 0.45 m of travel against a 6.67 m target disc — negligible, so I dismissed the
trace. That was right about smearing and wrong about the trace, because the damage is done
by the *other* half of it. `z_lag_bar` is a **20-second running mean**, and E085 measured
dwell times of **17–75 s**. The baseline therefore tracks position and subtracts it. It is
a high-pass filter with a corner sitting directly on the signal's own timescale.

## 7. Interpretation

**The routing change is correct and the architecture review's diagnosis was right — the
hippocampus was genuinely missing from the circuit.** Parked decodability going 84.6% →
99.5%, the distance profile inverting from worst-at-target to best-at-target, and
hippocampus-alone reaching 73.7% under movement where the pallium sits at 54.4%, all say
the same thing: place now has a home and it is legible there.

**What the primary falsifier caught is that `W_pred` cannot get at it.** The readout is
diluted across 336 units and then high-passed at a corner that removes it. Neither is a
fact about the representation; both are facts about the pathway that reads it. Which
means E085's conclusion — "mechanism 2 is insufficient as built" — needs narrowing rather
than withdrawing: **the place representation was insufficient, and now is not; the
readout is.**

**The centring is not a mistake, and that is what makes this a real architectural
tension.** E070 measured a planted place association predicting **1.0000 at its own place
and 0.9637 at a different one**, because `z_lag` is strictly positive with mean ~0.23 and
the across-stimulus signal is 3.7% of that DC baseline. Projected raw, DC dominates and
nothing is selective. E071 added the centring for exactly that reason and it worked. So
the pathway needs the DC removed **and** needs slow signals preserved, and the current
implementation cannot do both because it removes everything slower than 20 s.

**And the 20 s is not a considered choice.** `z_lag_bar` updates with
`a_b = cfg.dt / pc.baseline_tau_s` (`plasticity.py:355`) — the **same constant as the
reward baseline**. Two unrelated quantities share one time constant, and nothing in the
source states that as a decision. The reward baseline wants to track on the timescale of
reinforcement; the prediction-centring baseline wants to be slow compared to whatever the
prediction is about. There is no reason those should be equal, and here they are in direct
conflict.

**Prediction 1 was wrong in its number and right in its mechanism.** I predicted ≥65% and
recorded moderate confidence, reasoning that 80 hippocampal units against 256 pallial ones
might leave the pooled readout dominated by non-spatial drive. That is exactly what
happened — I simply did not anticipate the centring term on top of it.

## 8. Consequence

**Adopted, and staying on:** `place_to_hippocampus`. It is off by default per the
E076/E077 lesson, but it is now the recommended configuration for any T2 work, and the
guard test enforces both halves at `n_hens=16`.

**Narrowed, not withdrawn — E085's "mechanism 2 is insufficient as built".** The
representation is now sufficient when read directly (90.0% balanced, hippocampus, `z_lag`).
What is insufficient is `W_pred`'s access to it.

**Next, and it is one experiment: decouple the prediction-centring timescale from the
reward baseline.** Give `z_lag_bar` its own `pred_bar_tau_s`, defaulting to the current
value so nothing moves, and sweep it. The prediction is that decodability climbs from
73.7% toward the uncentred 90.0% as the corner moves below the signal, **while E070's
selectivity failure does not return** — that is the falsifier, and it must be measured,
not assumed, because it is the reason the centring exists. The two requirements are
separable in principle: a baseline slow enough to remove true DC but not fast enough to
track a 17–75 s dwell.

**Also worth doing, cheaply: restrict `W_pred`'s source for this association.** Sourcing
from the hippocampus alone rather than all 336 units buys ~10 points on its own and needs
no new mechanism. Whether that is principled or merely convenient deserves an argument
before it is adopted — a real pallium reads its hippocampus, it does not bypass everything
else — so it is recorded as available rather than recommended.

**Still standing from §5: any learning run on this needs an autoencoder control**, because
`shared_place_map` routes testimony into the same region `W_pred` now sources from.
