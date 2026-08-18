# E046 — T1's other half: does per-hen vigilance fall as the flock grows?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T1** — shared vigilance. `SUPPORTED as a narrower claim than proposed` (E045): the
channel delivers a capacity-robust safety benefit with no intake cost, but not the
originally-predicted intake benefit. T1's design (`docs/backlog.md` §3) had a second,
never-tested half: *"per-hen vigilance falls as flock size rises."* E045 only swept
capacity; this sweeps flock size.

## 2. Question

The many-eyes literature predicts that as group size grows, each individual can safely
spend less time personally vigilant, because the *collective* probability that someone
is watching rises even if no one individual watches more. In this model, `head_down`
(fraction of time foraging, i.e. blind to the sky — already tracked by
`run/h4.py`'s `H4Result`) is the vigilance proxy: higher `head_down` means less
individual vigilance. **Does `head_down` rise with flock size specifically when the
channel is informative (L), and not when it isn't (C?)?**

## 3. Prediction

**Primary.** `head_down` increases with `n_hens` for L. The many-eyes mechanism only
works if a hen can trust that *someone* is watching and will warn her — which requires
the channel to carry real information. A flock hearing an uninformative channel (C?) has
no more reason to relax vigilance at 32 hens than at 4; if `head_down` rises with flock
size for C? *too*, and by a similar amount, the effect is about flock size or crowding
generically, not about the channel.

**Secondary.** Given T1's already-established finding (E045) that L's *safety* advantage
holds at every capacity tested, predict it also holds at every flock size — L's
`caught_itt` should stay below C?'s across the `n_hens` range, not just at the
standard 16.

## 4. Falsifier

**If `head_down` does not rise with flock size for L, or rises equally for both L and
C?**, the many-eyes mechanism this experiment was built to detect is not operating —
whatever safety benefit the channel provides (E045) is not coming from individuals
relaxing vigilance as the flock grows, and T1's "shared vigilance" framing needs
revising to something more accurate (e.g. a fixed per-hen response magnitude that simply
compounds with more potential warners, rather than a flock-size-dependent individual
strategy).

## 5. Design

`run/h4.py`'s existing machinery, matching E045's pattern exactly but sweeping `n_hens`
instead of `pallium_scale`. Two conditions (`L`, `C?`), both `scaffold=True`,
`pallium_scale=1.5` (H4's standard, held fixed — this experiment varies flock size, not
capacity). **Flock sizes**: {4, 8, 16, 32} — matching E017's already-established range
("holds 4–32 hens") so nothing here is testing an unfamiliar regime for the world model.

**World**: hawk every 20 s, 10 minutes, `food_deplete_rate=0`, otherwise
`spec.DEFAULT_COOP` defaults (arena size fixed regardless of flock size, matching every
other flock-size variation in this codebase).

**Metrics**: `head_down` (primary) and `caught_itt` (secondary), both already computed
by `run_condition`.

**Replicates**: 8 seeds per (flock size, condition) cell, matching this session's
established first-pass count.

**Command:**
```bash
python -m scratchpad.e046_t1_flocksize --seeds 8 --minutes 10 --hawk-period 20
```

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
