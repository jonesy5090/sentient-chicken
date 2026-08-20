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

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
