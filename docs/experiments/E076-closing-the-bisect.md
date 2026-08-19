# E076 — closing the bisect: both causes were mine

> **Diagnostic**, continuing [E075](E075-bisect-h2f-control.md)'s bisect with its
> fourth arm, plus the default decision that arm unblocked.

## 1. Parent hypothesis

**H2f**, whose food-channel control E074 found firing at t=10.04 where E057 reported it
null. E075 ruled out E067's `m_acc` fix and traced roughly half to E060's contamination,
leaving a residual just over threshold.

## 2. Question

E075 measured, but did not causally test, the remaining lead: E063's place-cell block is
**25.1% of all observation drive, active 100% of the time**. Does disabling it, on top of
contamination, restore E057's null?

## 3. Result

Fourth arm added to E075's design — 8 seeds, 30 min, paired, threshold t(7)=2.365:

| arm | general elevation | audience-specific | **food control** *(should be null)* |
|---|---|---|---|
| `current` | +0.1500, t=6.85 | +0.2242, t=42.46 | +0.1054, t=10.04 |
| `legacy_m` | +0.1547, t=6.99 | +0.2251, t=46.92 | +0.1014, t=9.10 |
| `no_contamination` | +0.1256, t=8.89 | +0.2116, t=31.74 | +0.0519, t=2.70 |
| **`no_contam_no_place`** | **+0.1318, t=8.41** | **+0.2407, t=37.94** | **−0.0293, t=0.98 — null** |

**Against E057's own recorded figures** (general +0.123, audience-specific +0.232, food
control null), the fourth arm reproduces to within noise. The control's point estimate
even goes slightly negative, which is what an honest null looks like.

**Checked while here, because E073 rested on it.** E073 concluded the pallium is
saturated in live operation at 0.7288 — but that was measured with place cells
contributing a quarter of observation drive. Without them: **0.6907**. Still deeply
saturated, so E073's conclusion survives; place cells account for ~0.04 of it, not the
phenomenon.

## 4. Interpretation

**The bisect closes, and both causes are additions I made in this session.**

- **E060** put contamination into `DEFAULT_COOP` with no opt-in — 32 sickness onsets per
  30-minute run, gakel calls, `CLS_SICK` visibility, 0.15× mobility, for every
  experiment since.
- **E063** added 25 permanently-active channels worth a quarter of all observation
  drive, likewise with no opt-in.

Neither is wrong as *machinery*. Both are wrong as *defaults*, and the project says so
explicitly: `legacy_audio`, `auditory_scaffold`, `pred_enabled`,
`readout_scaling_strength`, `gakel_scaffold` and `balanced_ei` are each off by default
with a comment explaining that enabling one silently would change the comparison basis
for everything before it. I wrote several of those comments while violating the rule two
files away.

**H2f itself was never damaged.** The audience-specific effect holds at t=31.7–46.9
across all four arms and is *largest* in the clean one. E057's result stands, fully
reproduced. What was damaged was the control that made it interpretable as *targeted*
rather than *indiscriminate* — and that is now repaired rather than explained away.

**Why the contamination half is mechanically plausible**: sickness makes hens call
(gakel), slow to 0.15× mobility, and become visually salient. The audience assay measures
calling with and without a listener present, so a world in which hens are periodically
incapacitated and vocalising changes both the flock's spatial distribution and its
acoustic background — the two things that assay is sensitive to. The place-cell half is
better explained by drive: a quarter more constantly-on input, into a network already
saturated.

## 5. Consequence

**Both defaults flipped to `False`.** This is the deliberate decision E075 declined to
make in passing, and the evidence now supports it directly: with both off, a known-good
result reproduces; with them on, its control breaks. T2's own experiments opt in
explicitly, which is what every other piece of optional machinery here already does.

Fixing the fallout was itself informative. Seven tests failed on the flip — all of them
tests *about* contamination or place cells, which correctly opted back in. But several
others would have **passed vacuously**, comparing two all-zero vectors and asserting they
matched. Those were opted in too. A flag that makes a test pass by emptying its subject
matter is worse than one that makes it fail.

**Ethogram 13/13, suite 86/86** with the flip in place.

**What this does not settle.** Whether contamination and place cells *should* eventually
be part of the baseline world is a real question and not this experiment's to answer. The
claim here is narrower: they must not arrive there silently, and the burden is on the
experiment that wants them to say so.

**A process note.** Four times this session a null or anomaly has traced to the
instrument rather than the bird, and this is the second where the instrument was
something I built. The bisect worked because it reverted against a known-good baseline
rather than reasoning about which change *ought* to matter — my own prediction (E067's
`m_acc`) was wrong, and the arm that mattered was one added only after the first two came
back flat.
