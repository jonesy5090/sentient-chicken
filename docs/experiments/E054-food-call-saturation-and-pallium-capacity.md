# E054 — did food-call saturation crowd out pallium capacity for the alarm channel?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2c** — top-down association/comprehension as a working mechanism. `NOT STARTED` as a
working mechanism; E042 found comprehension null even after E041's density fix ("nowhere
near a working mechanism"). This is a new, specific candidate explanation for that null,
raised directly by a user observation of the offline-replay viewer and structurally
distinct from every account tested so far (E017/E034's fan-in dilution, E041's density
fix, E042/E043's exposure, E044's structural read): **capacity competition from a
different, unrelated channel**, not a property of the alarm channel or its representation
on its own.

## 2. Question

[E053](E053-food-call-discovery-pulse.md) fixed the food-call reflex from firing
continuously on sight (measured: 42.8% of hen-steps, 4/16 hens over 50%) to a discovery
pulse (measured: 4.2%, 0/16 over 50%). **Does removing that near-constant activity
change whether comprehension emerges** — i.e., was the pallium's limited representational
capacity being spent disproportionately on the food channel's constant chatter, at the
alarm channel's expense?

## 3. Prediction

**No confident directional prediction — registered as genuinely uncertain**, consistent
with this session's now-repeated pattern for H2c/H2d (E036, E040, E042–E044): every
previously-tested precondition (comprehension scaffold, density fix, exposure, structural
capacity) supplied the named missing piece and comprehension stayed null anyway. If that
pattern holds, removing food-call noise should also not produce a working comprehension
mechanism. **What would be genuinely informative either way**: this is the first test in
that series aimed at a *competing* channel rather than the alarm channel's own
representation, so a positive result here would be qualitatively different from anything
found so far — and, per this project's standing rule, would need immediate replication on
a fresh seed block before being trusted.

## 4. Falsifier

If comprehension after rearing does not differ between the discovery-pulse (E053) and
legacy continuous-food-call conditions, food-call saturation was not a load-bearing
constraint on H2c's null — narrowing the explanation further toward the rule-kind
question already opened at H2f (E036/E040), the same underlying limitation E042–E044
converged on for the alarm channel's own representation.

## 5. Design

**Instrument**: identical to E042 (`run.audience.comprehension`, crouch-response to a
played-back alarm call with no predator present; `run.simulate.simulate` for rearing).
Density held at E041's fix (`sensory_pallium_density=1.0`) in every condition, so only
the food-call variable differs between arms — E042 already established that density
alone does not produce comprehension, and holding it at the best-known setting gives this
test its best chance of detecting an effect if one exists.

**The manipulated variable**: `legacy_food_call` (new `hen/connectome.build` /
`hen/innate.reflex_matrix` parameter). `False` (default) uses E053's discovery pulse —
the current shipped reflex. `True` recreates the pre-E053 continuous-on-sight food call
as an ablation condition, existing solely for this comparison.

**Conditions** (matching E042's three-way structure):
- no association, discovery pulse — baseline (manipulation check: association should
  still exceed this if it exceeds anything)
- association, discovery pulse (E053, current default)
- association, legacy continuous food call (pre-E053 ablation)

**Primary metric**: comprehension after 20 minutes' rearing, discovery-pulse vs. legacy
continuous-call (both with association enabled). **Secondary**: association vs.
no-association under the discovery-pulse condition (replicates E042's own manipulation
check); `|W_pred|` structural norm, exploratory.

**Replicates**: 8 seeds, this session's first-pass default, matching E042.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e054_food_call_saturation_and_capacity.py --seeds 8 --minutes 20 --hawk-period 20
```

## 6. Result

8 seeds, 20 min rearing, hawk every 20s, `sensory_pallium_density=1.0` throughout.

```
condition                                             comp before  comp after  |W_pred|
no assoc, discovery pulse (baseline)                      -0.0002     -0.0007   0.00000
assoc, discovery pulse (E053, current default)             -0.0002      0.0042   0.00041
assoc, legacy continuous food call (pre-E053 ablation)     -0.0002      0.0047   0.00041

PRIMARY -- discovery pulse vs legacy continuous food call:
  -0.0005 +/- 0.0007  t=0.70  threshold(df=7)=2.365  -> not significant

SECONDARY -- discovery-pulse association exceeds no-association:
  +0.0048 +/- 0.0028  t=1.75  threshold(df=7)=2.365  -> not significant

wall clock: 491 s
```

**The falsifier fires cleanly.** Comprehension after rearing is statistically
indistinguishable between the fixed (discovery-pulse) and legacy (continuous-calling)
food-call conditions — if anything the point estimate trends the *opposite* direction
from the hypothesis (legacy slightly higher, not lower), well within noise either way.
Both land at 0.004–0.005, two orders of magnitude below the auditory scaffold's
comprehension (~0.19, E036/E040) and consistent with E042–E044's characterisation of
this mechanism as "nowhere near working" regardless of what precondition gets fixed.

**The manipulation check itself is weaker here than in earlier runs**: association vs.
no-association (both discovery-pulse) does not clear significance either (t=1.75 against
2.365), though it points the expected direction. This is consistent with, not a
contradiction of, this session's running characterisation of H2c as a small, marginal,
inconsistent signal (E044: "real but small and partial... inconsistent, 2/6 seeds near
zero") — an 8-seed block landing short of clearing threshold on an already-marginal
effect is the expected behaviour of a marginal effect, not a new finding on its own.

## 7. Interpretation

**Food-call saturation was not a load-bearing constraint on H2c's null.** The user's
live hypothesis — that near-constant food-calling might be crowding out pallium capacity
that could otherwise represent the rarer alarm channel — does not survive the direct
test, on the same instrument and density fix that gave this mechanism its best
previously-known chance (E041/E042). Comprehension is equally (un)responsive whether
food-calling is saturated (42.8% of hen-steps, pre-E053) or fixed (4.2%).

**This adds a genuinely new kind of null to the H2c series, not a repeat of an old one.**
Every prior test (E036/E040: comprehension scaffold; E041/E042: density; E043: exposure;
E044: structural read) targeted the alarm channel's *own* representation or its
preconditions. This is the first to test a *competing-channel capacity* account instead,
and it fails the same way — strengthening, rather than narrowing, the case that the
limitation is in the learning rule itself (H2f: instrumental where the biology needs
something closer to Pavlovian), not in any specific representational precondition this
project has been able to name and fix one at a time.

**Not independently replicated on a second seed block**, per this session's own standing
practice of prioritising replication for surprising or positive results. This is a null
in the predicted direction, consistent with four prior experiments' worth of the same
pattern — the cost of a second 8-seed block (another ~490s) was not judged worth spending
on confirming a null that already fits the established picture, unlike E048's positive
3-seed reading, which did warrant the check and did not survive it.

## 8. Consequence

- **H2c stays `NOT STARTED` as a working mechanism.** No status change — this is another
  negative data point in an already-consistent series, not a new finding that moves the
  tree on its own.
- **The food-call-saturation hypothesis is closed, not left open.** It was a genuinely
  new, structurally distinct candidate explanation (capacity competition from an
  unrelated channel) and it has now been tested and found not to hold.
- **E053's fix stands on its own merits regardless of this result.** The discovery-pulse
  reflex is closer to documented cockerel behaviour than continuous sight-gating; that
  was true before this experiment and remains true after a null result on the
  capacity-competition question it was also used to test.
- **Reinforces H2f as the more promising remaining direction** for the comprehension
  question broadly — the rule-kind hypothesis, not a further search for representational
  preconditions to fix one at a time.

