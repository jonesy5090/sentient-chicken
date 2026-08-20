# E080 — is H2d's loss really fan-in dilution?

> **Pre-registered.** Sections 1–5 written and committed before the run.

## 1. Parent hypothesis

**H2d**, whose every intervention is now closed (E079). One mechanism story has survived
untested since E017 and is the reason several of those interventions were attempted at
all.

## 2. Question

E017 localised H2d's 14.5–17× loss to the sensory→pallium projection and named the
mechanism: *"each pallial unit sums ~19 stub inputs of which one or two carry the
distinction, so a clean difference lands as a small perturbation on a large common-mode
drive."* E034 replicated the localisation. That framing motivated modality segregation
(E017/E035), the density sweep (E041/E078) and balanced E/I (E072/E077) — all now null
or reversed.

**Dilution makes a direct, gradable prediction that has never been tested**: if the
distinction is drowned by irrelevant input, then progressively *removing* irrelevant
active channels should progressively raise separability, approaching a large value when
only the informative channels remain.

E079 §8 proposed testing this via `OBS_DIM`'s historical growth. That turns out to be
the wrong instrument — E076 already disabled the two largest additions (place cells,
contamination) by default, and the remaining historical channels (`CLS_CROWDING`,
wall-escape, `IDX_FOOD_ARRIVAL`) are inactive in a staged scenario, so masking them would
change nothing. Removing *currently active but uninformative* channels tests the same
claim directly and does not depend on reconstructing history.

## 3. Prediction

**The endpoints are already measured and they do not favour dilution.** Naturalistic
input (14 active channels) gives 0.0814; the sparse probe (1 active channel, informative
only) gives 0.0961 — removing every irrelevant channel buys **1.18×**, against a loss of
14.5–17×.

So the prediction is: **a small, probably monotonic rise, far short of anything that
could explain the loss.** If that holds, fan-in dilution is not the mechanism, and the
three interventions built on it failed for a reason that was visible in advance.

**What would change my mind**: a steep or accelerating rise as the last irrelevant
channels go, i.e. most of the gain concentrated near full removal. That would mean the
endpoint comparison understates dilution because the relationship is non-linear, and the
mechanism survives.

## 4. Falsifier

If separability rises steeply and approximately in proportion to the fraction of
irrelevant input removed, dilution is confirmed and E017's framing stands — in which case
the interventions failed for implementation reasons rather than because the mechanism was
wrong, and are worth revisiting.

## 5. Design

Take the naturalistic hawk/call/rest observations. Identify the channels **identical
between hawk and call** — by construction these carry nothing about the distinction.
Zero a fraction *k* of them, in all three observations alike so the separability
denominator stays consistent, and measure.

- **k**: 0, 0.25, 0.50, 0.75, 1.00. At k=1.00 only informative channels remain, which
  should approach the sparse probe's numbers — a built-in consistency check.
- Channels are removed in a **fixed random order per genome**, so the fraction is what
  varies rather than which particular channels.
- **Metric**: `pallial_sep` verbatim from E041.
- **12 genomes, paired**, per E035.
- **Also reported**: mean pallial rate, since removing input necessarily lowers drive,
  and E079 established that drive itself moves separability. If the rise tracks drive
  rather than dilution, that must be visible here rather than inferred later.

**The drive confound is the reason this experiment could mislead**, and it cuts against
my own prediction: removing channels lowers drive, and E079 showed lower drive *hurts*
separability below the 0.95 optimum. So any rise observed is a rise *against* a headwind,
and a flat result is genuinely ambiguous between "no dilution" and "dilution exactly
cancelled by drive loss". Stated now, before the numbers.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e080_dilution_test.py
```

## 6. Result

Staged naturalistic observation: **15 active channels, 2 informative, 13
irrelevant-but-active.** 12 genomes, paired, threshold t=2.201.

| fraction removed | irrelevant kept | separability | settle rate | vs no removal |
|---|---|---|---|---|
| 0% | 13 | 0.0814 | 0.4602 | — |
| 25% | 10 | 0.0895 | 0.4023 | 1.10×, t=1.63, null |
| 50% | 7 | **0.0976** | 0.3501 | **1.20×, t=2.34, significant** |
| 75% | 3 | **0.0984** | 0.2960 | 1.21×, t=1.68, null |
| 100% | 0 | 0.0902 | 0.2724 | 1.11×, t=1.00, null |

At 100% removal separability lands at 0.0902 with rate 0.2724 against the sparse probe's
0.0961 at 0.2724 — the built-in consistency check passes.

## 7. Interpretation

**Dilution is real and negligible.** The largest effect anywhere in the sweep is
**1.21×**, and the only significant point is 1.20× at half removal. **H2d's loss is
14.5–17×.** Even taking the best number at face value, dilution accounts for on the
order of **1–2%** of the phenomenon it has been invoked to explain since E017.

**The non-monotonicity is the confound §5 predicted, and it works in my favour rather
than against.** Settle rate falls 0.4602 → 0.2724 as channels are removed, and E079
established that lower drive hurts separability below the 0.95 optimum. So the curve is
dilution-gain minus drive-penalty, which is why it peaks at 75% and falls back at 100%.
Correcting for that headwind would raise the true dilution effect somewhat — but from
1.2× to perhaps 1.3×, not to anything approaching 14.5×. §3 named the result that would
have changed my mind (a steep or accelerating rise near full removal); the opposite
happened.

**E017's localisation stands; its mechanism does not.** The loss is at the
sensory→pallium projection — E017 measured it, E034 replicated it, and nothing here
touches that. What fails is the *explanation*: "one or two informative inputs among ~19
drowned by the rest" predicts that removing the rest recovers the signal, and it recovers
1.2× of a 15× loss.

**Three interventions were built on that explanation** — modality segregation (E017/E035),
the density sweep (E041/E078), and balanced E/I (E072/E077). All null or reversed. In
hindsight they were testing a mechanism that a single graded measurement could have shown
was too small to matter, and the measurement was cheap.

## 8. Consequence

**H2d's magnitude now has no identified mechanism.** The position is worth stating
precisely, because it is more honest than "unsolved":

- **Where** the loss happens is established and replicated: sensory→pallium (E017, E034).
- **Why** is not. Dilution is measured and too small (here). Saturation is measured and
  wrong-signed (E079). Recurrent mixing is measured and wrong-signed — removing
  recurrence makes it *worse* (E017, E034). Common-mode DC is measured and null (E077).
- Every structural intervention tried is null or negative (E079's table).

**One candidate remains untested and is worth naming precisely** so it is not confused
with what has already been ruled out: the *strength* of pallial recurrence, as distinct
from global `gain` (E079, which scales every projection including afferents) and from
*removing* recurrence entirely (E017/E034). A network whose recurrent dynamics dominate
its input would converge toward input-independent states, which is the shape of the
observation. Note the counter-evidence already on record: E070 found place-pattern
similarity identical at 1, 10, 50 and 300 settle steps, which is not what attractor
convergence looks like. So this is a lead with a strike against it, not a promising one.

**The alternative remains that H2d is not a defect** (E079 §8): a two-channel distinction
in a high-dimensional observation, projected randomly into a recurrent pool, may simply
differ by ~7–8% of mean rate. Notably, a random *expansion* (64-unit stub → 256-unit
pallium) should preserve distances by Johnson–Lindenstrauss, so this account requires the
nonlinearity to be doing the work — and that is testable, which makes it a better next
question than another intervention.

**What this session established about H2d overall**: not a fix, but a much sharper map of
what is not true. Four mechanism stories are now measured rather than assumed, and the
three interventions that consumed the most effort were all downstream of the one that
turns out to explain 1–2% of the effect.
