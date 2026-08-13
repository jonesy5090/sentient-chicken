# E022 — second adversarial review, verified

> **Diagnostic, not pre-registered.** Re-measures claims made by an outside agent that
> read this repo with no context, commissioned via `/red-team` and asked the direction
> question: *what realistically needs to be solved next?* Recorded because two findings
> are structural, one is a defect in a guard test written the same day, and one of the
> review's headline numbers **does not replicate**.

## 1. Parent hypothesis

H2 and H4, plus the project's statistical practice.

## 2. Method

Every claim below re-measured from scratch in this repo, with my own scripts
(`scratchpad/verify_*.py`), not the reviewer's (`scratchpad/rt_*.py`, committed as its
evidence trail and **not** relied on).

## 3. Verified — adopt

### 3a. The pallium has no inhibitory neurons at all

`hen/connectome.py:103-104` assigns excitatory/inhibitory identity by **flat index** over
a **region-ordered** array. Regions are contiguous, so the 80% cut lands mid-arcopallium
and segregates E/I by region instead of mixing within each:

| region | excitatory |
|---|---|
| sensory | 100% |
| **pallium** | **100%** |
| hippocampus | 100% |
| arcopallium | 20.8% |
| hypothalamus | **0%** |
| motor stub | **0%** |

`hen/regions.py:79-81` documents `EXCITATORY_FRACTION = 0.8` as "fraction of neurons that
are excitatory", which reads as 80/20 throughout. It is not. Real avian pallium is
~20–30% GABAergic *throughout*, so this is also an undocumented departure from biology,
which the project's own rule forbids.

A purely excitatory recurrent pool is why the gain had to be held to two decimal places —
`hen/connectome.py:78-81` complains about exactly that without diagnosing it. **This is
the top structural item on the board.**

### 3b. The primary metric sits exactly on a knife edge

Hunger equilibrium is `h* = (1/hunger_fill_s) / (peck_food_rate · f)`. At the observed
feeding rate of 6.17%, `h* = 0.300` — and `coop/world.py:62` starts every hen at hunger
**0.30**. So "within-run hunger change" measures the *sign of* `f − 6.17%`.

Measured, fixed condition, 12 seeds, 10 min:

```
hunger change   mean -0.0124   sd 0.0660   range -0.131 .. +0.089
fed %           mean  6.66     sd 3.63     range  2.71 .. 14.87
corr(hunger change, fed %) = -0.939
```

**The nuisance spread (0.066) is larger than every effect the project has ever chased
(0.001–0.062).** And at −0.94 correlation the metric is a sign-flipping restatement of
`fed %`, which is already printed in the same table.

### 3c. Dale's law is not enforced on `W_out`

All 48 motor-stub neurons carry both signs. `hen/plasticity.py:335` clips symmetrically
with no sign constraint. `CLAUDE.md` states the invariant absolutely and
`test_dale_law_survives_learning` only checks `p.W`.

### 3d. E019's guard test would not have caught E019's defect

`tests/test_plasticity.py:298` perturbs each drive by an identical −0.01 and compares the
reward *response*. Under the broken code all four entered `d_drive` with the same
coefficient — so each scores exactly 25%, well under the 0.6 threshold, and **the guard
passes on the bug it was written for**.

E019's finding was *variance share in a rollout*: vigour dominated because vigour varied
(sd 0.23) while hunger barely moved. The test guards **sensitivity**; the defect was
**variance**. Written the same day, hours after diagnosing it.

### 3e. Two documentation claims are false

- `docs/experiments/E011` §6–8 are literally `_Pending._`, while `hypothesis.md:639`
  records a result from it that is load-bearing for invalidating E010.
- The README *body* still presents E013 as current ("destroyed 48% of the brain",
  "4.7% against 6.2%"), still says "**t=3.93.** Learning works", and still asserts
  "Nothing has been withdrawn" — false four times over. The *Current state* section was
  corrected earlier the same day; the body above it was not read.

## 4. Not verified — do not adopt

### The review's headline number does not replicate

It claimed food-patch layout accounts for **~80%** of the metric's variance, and that
pinning the layout cuts the spread from 0.070 to 0.031 — a 5× variance win it called
"nearly free" and ranked first.

Measured here, same 12 seeds, layout pinned versus varying:

| | sd (hunger change) | sd (fed %) |
|---|---|---|
| layout varies | 0.0660 | 3.63 |
| layout pinned | **0.0550** | 2.33 |

That is a **30% reduction in variance, not 80%** — and only **1.2×** in standard
deviation, against the 2.3× claimed. Pinning the layout is not the free 5× win the
review ranked first.

**Why the gap is itself informative:** the residual depends on *which* layout you pin,
and neither estimate has the seeds to say. At n≈12 an sd estimate carries roughly ±25%,
so the reviewer's 0.031 and my 0.0550 are not straightforwardly reconcilable and one of
them is a lucky draw.

**What survives:** the *diagnosis* (the metric is dominated by nuisance and sits on a
knife edge) is confirmed independently by 3b. The *prescription* (pin the layout) is not
supported at the claimed magnitude. Promoting `fed %` to primary follows from 3b on its
own and does not depend on the layout claim at all — which makes it the safe half of the
recommendation.

### Not assessed

The review's argument that **H4 does not require plasticity** is an argument, not a
measurement, and is treated separately in §5. Its claims about credit-window limits,
audience-assay confounds and the inert kin term are plausible and **not yet
re-measured**; they are recorded in the backlog as owed checks rather than acted on.

## 5. The sequencing argument

The review's strongest non-numerical claim: `docs/backlog.md:180` says phase 1 "blocks
everything below", and that is **wrong about H4**. H4 asks whether an intact channel
beats a shuffled one on a task requiring private information. Its prediction mentions no
learning. Production is innate and passes 7/7 probes; calls are audible since E019
(+0.908); comprehension can be made innate with the E018 scaffold (measured 0.189).

That is a complete, runnable H4 with **no plasticity anywhere** — and it tests the
project's actual thesis, which twenty-one experiments have not reached.

**Assessment: correct, and it is the most useful thing either review has produced.**
Phase 1 has been blocked for 21 experiments on a metric §3b shows cannot measure it, and
the headline experiment has been queued behind it the whole time for a dependency that
does not exist. The counter-argument — that an all-innate H4 tests a weaker claim than
H0 makes — is real but does not justify the ordering: a null there would be about the
coop, and that is worth knowing *before* more work on the rule.

## 6. Consequence

**Top of the backlog, in this order:**

1. **Fix the `dale` sampling** (3a). Two lines, invalidates every genome, needs a
   deliberate re-baselining. Highest structural value on the board.
2. **Promote `fed %` to primary, keep hunger change secondary** (3b). Does not depend on
   the unreplicated layout claim.
3. **Run H4 with no plasticity** (§5). Needs the capacity ladder wired into
   `run_condition`, which currently hardcodes `DEFAULT_REGIONS`, and a channel-shuffle
   path in `coop/sensing.py`. Neither exists.

**Fixed immediately** (cheap, unambiguous): the guard test (3d), the README body and the
E011 citation (3e).

**Recorded as owed, not adopted:** the credit-window argument, the audience-assay
confound, the inert kin term in E006, `W_pred`'s unconditional cost, and `strike_penalty`
still being a per-step rate.

**Abandon**, per the review and consistent with the record: H2a (six comparisons, never
significant, sign flipped in E020) and the E016 "last word" follow-up (a workaround for a
harm that no longer exists).
