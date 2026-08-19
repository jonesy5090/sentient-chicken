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

_Running._

## 5. Interpretation

_Pending §4._

## 6. Consequence

_Pending §4._
