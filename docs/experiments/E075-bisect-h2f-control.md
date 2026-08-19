# E075 — what broke H2f's food-channel control?

> **Diagnostic, not pre-registered.** A bisect of a regression, following E067's own
> labelling. There is no hypothesis about the world here — only a question about which
> change to this codebase moved a number.

## 1. Parent hypothesis

**H2f**, whose status now carries a caveat.
[E074](E074-balanced-ei-adoption-gate.md) re-ran E057's contrast on current code and
found the **food-channel control firing at +0.1054, t=10.04** — 47% the size of the
audience-specific effect it exists to control for. E057 reported that control **null**,
and that null is what distinguished a *targeted* audience effect from *indiscriminate
elevation*; it is the same test E058 used to reject H2c's apparent result.

## 2. Question

Which change between E057 and now broke it? Three candidates, all recent:

1. **E067's `m_acc` fix** — changed `m`, the reward-modulation factor gating `W`'s
   update, from a single-step snapshot at the consolidation boundary to a mean over the
   window. `W` feeds the pallial states `W_out` reads, so this has a mechanistic route
   to the audience assay. **This one is mine.**
2. **`OBS_DIM` 74 → 138** — three new sensory blocks (`CLS_SICK` +
   `IDX_SICKNESS_ONSET`, place cells, the gakel location cue).
3. **`N_CALLS` 4 → 5** — the gakel call shifted every audio channel index.

## 3. Method

Candidate 1 first: cheapest to revert, most likely on mechanism, and the one I
introduced. Reverted behind `legacy_m_sampling`, a flag added for this purpose and
**never a default** — following `legacy_audio`'s precedent, which exists "purely so E021
can ask what the audio fix changed".

Two arms, otherwise identical to E057's contrast (8 seeds, 30 min, `auditory_scaffold=True`,
same `PlasticConfig` otherwise, paired seeds):

| arm | |
|---|---|
| `current` | as shipped |
| `legacy_m` | E067's fix reverted |

Reported per arm: general elevation, audience-specific difference-in-differences, and
the food control.

**Reading it.** If `legacy_m` restores a null food control, E067's fix is the cause and
the question becomes whether the fix or the control is the thing to keep. If it does
not, the cause is candidate 2 or 3 and the bisect continues — those are structural and
harder to revert, so a masking approach (zeroing the new blocks) would be next rather
than a true revert.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e075_bisect_h2f_control.py
```

## 4. Result

8 seeds, 30 min, paired, threshold t(7)=2.365. E057 reported the food control **null**.

| arm | general elevation | audience-specific | **food control** *(should be null)* |
|---|---|---|---|
| `current` | +0.1500, t=6.85 | +0.2242, t=42.46 | **+0.1054 ± 0.0105, t=10.04** |
| `legacy_m` (E067 reverted) | +0.1547, t=6.99 | +0.2251, t=46.92 | **+0.1014 ± 0.0111, t=9.10** |
| `no_contamination` (E060 reverted) | +0.1256, t=8.89 | +0.2116, t=31.74 | **+0.0519 ± 0.0192, t=2.70** |

**Candidate 1 (E067's `m_acc` fix) is ruled out.** Reverting it moves the control by
0.004 — nothing. All three metrics are near-identical between `current` and `legacy_m`.

That non-effect is mechanistically coherent rather than suspicious: H2f's reward is
dominated by continuous `d_drive`, where a single-step snapshot and a window mean are
nearly the same number. E067's 2% capture problem applies only to *discrete* events, and
this contrast runs at the 900 s hawk default where almost none arrive.

**Candidate 2 (E060's contamination) accounts for roughly half.** Disabling it halves
the control effect (+0.1054 → +0.0519) — but t=2.70 against a 2.365 threshold is still
significant, if marginally. **The bisect is partial: something else contributes.**

**Measured while looking for the residual, not yet causally tested:** E063's place-cell
block is **25.1% of all observation drive and active 100% of the time** — 25
permanently-on channels added to what had been an 88-channel observation. (The E064
gakel-location block is 25 more channels but sits at 0% except during calls.)

## 5. Interpretation

**The headline is not which candidate won. It is that two changes I made while building
T2's scaffold altered the shared substrate for every hypothesis, with no opt-in.**

- **E060 put contamination into `DEFAULT_COOP` unconditionally.** From E060 onward every
  experiment ran with hens being poisoned — 32 sickness onsets per 30-minute 16-hen run,
  gakel calls firing, `CLS_SICK` visible, mobility at 0.15× throughout.
- **E063 added a permanently-active block worth a quarter of all observation drive.**

This project's conventions exist precisely to prevent that. `legacy_audio`,
`auditory_scaffold`, `pred_enabled`, `readout_scaling_strength`, `gakel_scaffold`,
`balanced_ei` are all off by default, each with a comment explaining that turning it on
silently would change the comparison basis for everything before it. Both of these
skipped that, and four subsequent experiments were built on top without anyone noticing —
including my own E074 adoption gate, whose reference arm was contaminated by exactly this.

**What it does to H2f.** E057's audience-specific effect is not in question: it survives
every arm here at t=31.7–46.9. What is in question is the *control* that made it
interpretable. E057 could say the effect was targeted rather than indiscriminate because
the food channel stayed null; on current code it does not, and even with contamination
removed it clears threshold. Until the residual is explained, H2f's "predominantly
audience-conditional" framing rests on a control that no longer behaves as it did.

**Confidence.** The `m_acc` exoneration is clean — two arms, near-identical, with a
mechanism that explains why. The contamination result is solid in direction (halved) but
its remainder sits close enough to threshold that an 8-seed block cannot settle whether
the residual is a real smaller effect or noise. The place-drive figure is a measurement,
not a test: 25.1% is real, its *causal* contribution is unmeasured.

## 6. Consequence

**`contamination_enabled` added, defaulting to `True`.** The flag alone changes nothing.
Whether the default should flip is a real decision — contamination is now part of the
world every non-T2 experiment inherits — and it needs its own evidence rather than being
settled inside a bisect.

**H2f's node keeps its caveat, with the bisect's partial result recorded.** Not
downgraded: the audience effect is robust across all three arms. The caveat is about the
control.

**Next, in order:**
1. **Test the place-drive residual causally** — re-run the `no_contamination` arm with
   the place block masked to zero. If the control returns to null, E063's always-on drive
   is the remainder and both causes are mine.
2. **Then decide the defaults deliberately**, as one question rather than three: should
   contamination, place cells, and the gakel blocks be opt-in for experiments that are
   not about T2? That is a convention decision affecting every future comparison.
3. `balanced_ei` adoption stays blocked behind it — E074's gate cannot be read against a
   contaminated reference arm.

**A note on process, since it is the recurring theme.** This is the fourth time in this
session that a null or an anomaly traced to the instrument rather than the bird, and the
second where the instrument was something I had built. The pattern that catches it is
always the same — bisect against a reverted baseline rather than reasoning about which
change *should* matter. My own prediction here (E067's `m_acc`) was wrong, and the arm
that mattered was one I added only after the first two came back flat.
