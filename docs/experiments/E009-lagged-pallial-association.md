# E009 — lagged pallial association, with enough co-occurrence data

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.

## 2. Question

[E008](E008-top-down-association.md) tested an autoencoder by mistake: it mapped
`rate(t) → obs(t)`, so during a hawk event it learned to predict the hawk from the
hawk. Two corrections, then: source the prediction from a **lagged** trace (cue →
later outcome) restricted to the **pallium** (not the sensory stub, which carries the
percept directly and dominates), and run at raised predator density so the
co-occurrence data actually exists.

## 3. Prediction

Comprehension rises above zero once the association is directed and the data is
present. The no-cue baseline must stay flat.

## 4. Result

30 min rearing, single seed, exploration on.

| condition | hawks vs baseline | comprehension | crouch, no cue | \|W_pred\| |
|---|---|---|---|---|
| no association (control) | 1x | −0.0000 | 0.0778 | 0.0000 |
| association, normal | 1x | −0.0008 | 0.0700 | 0.0074 |
| association, 15x predators | 15x | 0.0001 | 0.0788 | 0.0500 |
| association, 90x predators | 90x | 0.0001 | **0.1466** | 0.0500 |

**Null again**, and the no-cue column is the informative one: at 90x predator density
baseline crouching nearly doubles (0.078 → 0.147) with no cue present at all. The
projection is learning something — but it is learning the *base rate* of aerial
threat, not a contingency on the call. That is the hallucination failure mode H2c
flagged, arriving as an unconditional bias rather than a cued response.

## 5. Diagnosis — the pallium barely represents the stimulus

Measuring the representation directly, with a settled network:

```
                mean|rate|   shift: hearing a call   shift: seeing a hawk
sensory stub      0.4218       0.0949  (22.5%)         0.0819  (19.4%)
pallium           0.8577       0.0277  ( 3.2%)         0.0235  ( 2.7%)

afferent weight, alarm-call channel : 20.53 across 18 neurons
afferent weight, aerial channel     : 17.65 across 19 neurons
```

**The input is fine.** The call reaches the brain with slightly *more* afferent weight
than the hawk channel, and moves the sensory stub by 22% of its resting activity.

**The pallium is saturated and nearly blind to the difference.** Mean rate 0.86 puts
it deep in the flat region of the sigmoid, where the slope is ~0.12 and every
distinction gets compressed. A call shifts it 3.2%; a hawk shifts it 2.7% — and the
two states differ from *each other* by only 0.008, under 1% of the mean.

No associative rule can be cue-specific when sourced from a representation that does
not distinguish the cues. That explains the base-rate learning exactly: with nothing
to condition on, the best available prediction *is* the base rate.

**A gain sweep confirms saturation is real but is not the whole story:**

| recurrent gain | mean pallial rate | call vs hawk separability |
|---|---|---|
| 0.90 *(current)* | 0.830 | 0.0077 |
| 0.60 | 0.212 | 0.0072 |
| 0.40 | 0.162 | 0.0034 |
| 0.15 | 0.131 | 0.0010 |

Dropping the gain to 0.6 fixes the operating regime — mean rate 0.21 instead of 0.83
— and *relative* separability improves about fourfold (0.9% → 3.4% of mean rate). But
absolute separability barely moves and falls off sharply below 0.6. The pallium's
representations of distinct stimuli are weakly separated at every gain tested,
because the two inputs project onto overlapping random subsets of the sensory stub
and nothing downstream decorrelates them.

**This is a representational problem, not an architectural one.** E002, E007, E008 and
E009 were all attempts to fix the *routing* of learning. The blocker is upstream of
routing: there is not enough information in the pallial state to route.

## 6. Consequence

- **H2c stays `NOT STARTED`.** Still not fairly tested — the mechanism cannot work on
  a representation this uninformative, so this is not evidence against it.
- **H2d opened**: the pallium does not form separable representations of distinct
  stimuli. This is now the blocker behind H2b, H2c and everything downstream.
- **Important caveat on an existing result.** H2's supported finding
  ([E004](E004-replication-at-twelve-seeds.md), t=3.93) was measured with the network
  in the saturated regime (mean pallial rate 0.83). Drive regulation apparently only
  needs coarse modulation, which a saturated network can still supply — but the result
  should be re-run once the operating point is fixed, and it might well get *stronger*.
  Recorded rather than quietly left.
- **Not changed yet**: the default gain stays 0.9. Changing it invalidates the
  comparison basis for every result so far, so it should be a deliberate re-baselining
  with H2 re-run, not a quiet tweak mid-investigation.
- **Method note, now earned four times**: measure the mechanism before the behaviour.
  The representation measurement took under a minute and identified a blocker that
  four behavioural experiments had been circling.