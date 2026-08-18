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

8 seeds per cell, 10 min, hawk every 20 s, `food_deplete_rate=0`, `pallium_scale=1.5`.
Wall clock 1028 s.

```
n_hens  cond    head_down  caught/dive
4       L          0.5953       0.1733
4       C?         0.6062       0.1717
8       L          0.5755       0.1441
8       C?         0.6048       0.1349
16      L          0.5686       0.1088
16      C?         0.5748       0.1309
32      L          0.5427       0.0614
32      C?         0.5409       0.1331

n_hens   head_down L  head_down C?   L-C? t    caught/dive L-C?     t
4            0.5953       0.6062       0.48        +0.0015       0.03
8            0.5755       0.6048       1.88        +0.0093       0.32
16           0.5686       0.5748       0.46        -0.0222       1.01
32           0.5427       0.5409       0.20        -0.0716  *    4.02

* clears threshold t=2.365 (df=7)

head_down vs n_hens, linear slope: L=-0.0017/hen  C?=-0.0025/hen
```

**The primary prediction is falsified, cleanly.** `head_down` *falls* with flock size
for both conditions (both slopes negative — hens forage *less*, not more, as the flock
grows) at similar rates for L and C? (−0.0017 vs −0.0025 per hen, same order of
magnitude, C? if anything declining slightly faster). There is no flock-size-dependent
vigilance relaxation specific to the informative channel — the registered falsifier
condition ("rises equally for both, or doesn't rise for L") fires, just with the sign
flipped from what was anticipated (falls for both, rather than rises for both).

**The secondary prediction is not supported uniformly either, and the pattern is the
interesting part.** L's safety advantage (E045's headline) is not detectable at small
flocks — indistinguishable from zero at `n_hens=4` (t=0.03) and `n_hens=8` (t=0.32),
where the point estimate is even slightly in C?'s favour — and only emerges, clearly and
significantly, at the largest flock size tested (`n_hens=32`, t=4.02). `n_hens=16`
(E045's own capacity-sweep flock size) sits in between, negative but not significant.

## 7. Interpretation

**Both foraging conditions get less foraging time as the flock grows, most plausibly
from a chorus effect that has nothing to do with which channel is informative.** More
hens means more total calls in the air regardless of condition (production is identical,
symmetric, and unaffected by whether the receiver's channel is L or C?), and this
codebase's audio channel combines in *power*, so a busier acoustic environment is a
louder one. The innate call-suppression reflex (stop pecking on hearing *any* call, part
of the auditory scaffold) does not distinguish a meaningful call from a decorrelated
one — it fires on loudness, not content. More hens, more calls, more interruptions,
less time foraging: symmetric across L and C? because the mechanism triggering it is
symmetric.

**This means "shared vigilance," as originally framed — individuals actively relaxing
their own watchfulness because they trust others are covering for them — is not what
this architecture can produce, and there is a structural reason it can't.** The reflex
arc has no policy to relax; behaviour is a fixed function of instantaneous input, not an
adaptive strategy conditioned on flock size or trust. There is no mechanism by which a
hen could "decide" to forage more because there happen to be more potential watchers.

**What the safety result suggests instead is a passive, statistical version of the same
underlying idea.** More hens means more independent chances that *someone* sees the hawk
in time to call — a property of the population, not of any individual's behaviour — and
that should matter more, not less, as the flock grows, purely by increasing the odds a
useful warning exists at all. That is consistent with the safety advantage being
undetectable at 4–8 hens (too few independent chances for the effect to separate from
noise) and clearly present at 32 (enough chances that it reliably does). This is
speculative — the mechanism was not directly measured, only its outward signature — but
it is a coherent story that requires no learning and no individual strategy change,
matching everything else this architecture is actually capable of.

## 8. Consequence

- **T1's "many eyes" framing is corrected, not merely unreplicated.** The vigilance
  half of the original prediction is falsified: individual foraging time does not
  increase with flock size for the informative channel, and the small decline observed
  is symmetric across conditions, most likely driven by call volume rather than call
  content. `docs/hypothesis.md`'s T1 node updated to state this plainly rather than
  leave it as an open item.
- **The safety benefit (E045) is now known to be flock-size-dependent, which E045 did
  not test** (it fixed `n_hens=16` throughout). This refines E045's own headline: the
  Pareto advantage is not uniform across flock sizes — it is essentially absent at
  small flocks and grows with the flock, in what looks like a threshold rather than
  a smooth trend across only 4 points.
- **Not run here, and the natural follow-up if this specific mechanism is worth
  pinning down further**: does raw call volume (rather than flock size specifically)
  predict the vigilance decline — e.g. artificially inflating or muting call rate
  independent of `n_hens` — to test the "chorus effect drives interruption, not
  content" explanation directly rather than inferring it from the symmetry between L
  and C?.
- **This closes T1's second registered prediction.** Between E045 and E046, both
  halves of T1's original design have now been tested to a settled conclusion: no
  intake benefit, no vigilance-relaxation mechanism, but a real and flock-size-dependent
  safety benefit that emerges from population-level statistics rather than any
  individual behavioural change.
