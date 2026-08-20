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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
