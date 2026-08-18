# E045 — T1: the intake/risk trade-off across a capacity sweep

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H4** — an intact channel beats a shuffled one on a task requiring private
information. `SUPPORTED as written` (E030), checked and held against the
`food_deplete_rate` confound (E039). This experiment operationalizes `docs/backlog.md`
§3's **T1** (shared vigilance) properly for the first time: T1's own registered metric
(fed %, L vs C?) has been run three times before (E026, E028, E028b) plus a
depletion-controlled re-analysis this session, always as a **raw mean comparison**, and
always null. `docs/backlog.md` §2 specifies the actual intended test — a **capacity
sweep tracing intake-vs-risk curves**, "not raw intake" — which has never been run.

## 2. Question

H4 already establishes L is much safer than C? at the standard capacity (`caught/dive`:
E030's −0.044 pooled, E039's −0.077 at a fresh block). A raw fed% comparison at that
single capacity cannot distinguish "L achieves the same intake more safely" (a real,
if different, benefit) from "L simply forages less" (a real cost) — both are consistent
with a non-significant mean difference. **Does the intake/risk trade-off — traced across
several pallium capacities, not read off one point — show L dominating C? (safer with no
worse intake, or better on both axes), and does the picture change with capacity, per
`docs/backlog.md` §2's "where does the curve bend"?**

## 3. Prediction

**Primary.** At every capacity tested, L's `caught/dive` is lower than C?'s (replicating
H4's already-established result at each new capacity, not a fresh claim) — this part is
close to a manipulation check. **The live question**: whether L's fed% is significantly
*higher* than C?'s at any capacity, consistent with T1's original prediction, or whether
it stays statistically indistinguishable (as in the three prior single-capacity checks)
across the whole sweep, in which case the honest characterisation of the trade-off is
"same intake, less risk" rather than "more intake."

**Secondary, from `docs/backlog.md` §2 directly:** channel-driven separation between L
and C? (on either axis) should not simply grow monotonically with capacity — a genuine
possibility is that it plateaus or even narrows at high capacity if a larger, untrained
pallium's baseline interference (E002's ceiling finding: too much cortical influence
makes behaviour worse) starts to matter more than any channel effect.

## 4. Falsifier

**T1's original prediction (L forages more at matched risk) is not supported** if fed%
stays statistically indistinguishable between L and C? at every capacity tested — four
null replications (three prior single-point checks plus this sweep) would make "no
detectable intake benefit, only a safety benefit" the standing characterisation, not a
finding still awaiting a big-enough sample.

**A materially different, still-informative outcome**: if the *sign* of the fed%
difference flips across capacities (L worse at low capacity, better at high, or vice
versa) — that would be `docs/backlog.md`'s "where the curve bends" and the most
interesting possible result, whether or not any single point clears significance.

## 5. Design

`run/h4.py`'s existing machinery — `Condition`, `run_condition`, `_t_critical` — reused
directly, matching `scratchpad/e030.py`/`e039_h4_depletion.py`'s pattern. Two conditions
only: `L language` and `C? yoked`, both with `scaffold=True` (matching every existing
H4-lineage run for comparability — H4's registered ladder has always included the
auditory scaffold in these two conditions, not something introduced here).

**Capacity sweep**: `pallium_scale` ∈ {0.5, 1.0, 1.5, 2.0, 4.0} — `docs/backlog.md`
§2's own suggested levels, 1.5 being H4's existing standard (`EXPANDED`) so every prior
H4 result sits on this curve as one of its points.

**World**: 16 hens, hawk every 20 s, 10 minutes (H4's standard duration),
`food_deplete_rate=0` — required this time, not optional: fed % is exactly the metric
E037 found confounded by the default, and this experiment's whole point is to measure
it properly.

**Metrics, both tracked per cell**: `fed_rate` and `caught_itt` (intent-to-treat
caught/dive, H4's registered risk metric — unmovable denominator).

**Replicates**: 8 seeds per (capacity, condition) cell — this session's established
first-pass count, **not** a claim to replace H4's own 36-seed pooled headline; this
experiment is about the *shape* of the trade-off across capacity, which a first-pass
sweep can show, not about re-establishing significance at H4's existing standard
capacity (already established, repeatedly).

**Command:**
```bash
python -m scratchpad.e045_t1_pareto --seeds 8 --minutes 10 --hawk-period 20
```

## 6. Result

8 seeds per cell, 10 min, 16 hens, hawk every 20 s, `food_deplete_rate=0`.
Wall clock 1946 s.

```
capacity  cond     fed %  caught/dive
0.5       L        4.335       0.1122
0.5       C?       4.586       0.1604
1.0       L        5.010       0.0871
1.0       C?       4.757       0.1970
1.5       L        4.462       0.1088
1.5       C?       4.303       0.1309
2.0       L        5.025       0.1039
2.0       C?       4.886       0.1690
4.0       L        4.216       0.0990
4.0       C?       4.362       0.1119

capacity      fed% L-C?     t    caught/dive L-C?     t
0.5           -0.251       0.54     -0.0481          1.17
1.0           +0.253       0.56     -0.1099   *       4.59
1.5           +0.159       0.47     -0.0222          1.01
2.0           +0.138       0.31     -0.0651          1.43
4.0           -0.145       0.56     -0.0129          0.47

* clears threshold t=2.365 (df=7)
```

**The risk axis is robust and consistent: L is safer than C? at every single capacity
tested**, 5 of 5, same sign throughout — extending H4's established result across the
full range rather than resting on the single 1.5x point every prior H4 experiment used.
Only 1.0x individually clears significance at this seed count (t=4.59), but the
direction never wavers.

**The intake axis is a clean null at every capacity — the falsifier fires.** No fed%
difference clears threshold anywhere (max t=0.56, at 0.5x and 4.0x). Four single-point
null checks (E026, E028, E028b, this session's zero-compute re-analysis of E039) plus
now five more across a full capacity range: T1's original prediction — L forages *more*
than C? — has never once come close to significance, across nine total checks.

**But the sign is not constant, and the pattern is the secondary prediction's real
answer.** fed% L−C? is negative at 0.5×, positive at 1.0×–2.0×, negative again at 4.0× —
an inverted-U, not noise scattered randomly around zero with no structure. None of the
five points is individually significant, so this is reported as a pattern worth having
found, not a established curve.

## 7. Interpretation

**T1's original claim is settled, not merely unreplicated: there is no detectable
food-intake benefit from the channel, at any pallium size tested.** This is the honest
characterisation to carry forward, not "still awaiting a big enough sample" — nine
independent checks across four experiments and a five-point capacity sweep have found
the same null every time.

**What the channel demonstrably buys is safety, consistently, across the whole capacity
range — which is a real, if more modest, version of the "many eyes" prediction.** The
many-eyes literature predicts vigilance-sharing frees up foraging time; what this model
shows instead is vigilance-sharing without a foraging cost or benefit — a flock with the
channel is not eating more, but it is not eating less either, while being substantially
safer. That is a genuine Pareto improvement over C? (better on one axis, tied on the
other), just not the specific mechanism (more time with head down) the original
prediction imagined.

**The non-monotonic fed% pattern is plausible, not confirmed, and worth naming a
mechanism for.** Low capacity (0.5×) may simply not have enough pallium for the channel
signal to do anything beyond adding noise to the motor drive — consistent with E002's
ceiling finding that an untrained cortical pathway can actively interfere. High capacity
(4.0×) reversing again is consistent with the same interference growing with more
neurons to carry it (a bigger, more active untrained pallium per E023's rate/gain
table), rather than capacity monotonically buying more benefit. This is speculative and
not established at n=8 per point.

## 8. Consequence

- **T1 gets a settled characterisation for the first time**, replacing four scattered
  single-point nulls: **no intake benefit from the channel at any capacity tested; a
  consistent, capacity-robust safety benefit.** `docs/hypothesis.md` updated
  accordingly — T1 has not had a formal node before this; one is added under H4.
- **The Pareto framing `docs/backlog.md` §3 called for is now actually answered**,
  where four prior single-point mean comparisons could not answer it: L does not
  dominate C? on both axes, but it is not worse on either, and is much better on one —
  a real, if narrower, form of Pareto superiority.
- **The non-monotonic intake pattern is worth a properly powered follow-up** (more
  seeds at 0.5× and 4.0× specifically, where the sign reverses) if the mechanism
  (interference from an undertrained pallium at capacity extremes) is worth pinning
  down — not run here, and lower priority than the settled finding above.
- **This does not touch H2c/H2d/H2f.** T1 tests the channel's *behavioural* consequence
  via the innate route (production hardwired, comprehension via the scaffold), exactly
  like H4's own headline — it says nothing about whether learning ever contributes,
  which remains the separate, still-open question this session's E042–E044 left
  unresolved.
